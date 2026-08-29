"""Thin continuous-token adapter around the maintained Hugging Face Llama core."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import LlamaConfig, LlamaModel, PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput


class PretrainingConfig(PretrainedConfig):
    """Serializable configuration for the project-owned physical-token adapter."""

    model_type = "pretraining_trajectory"

    def __init__(
        self,
        *,
        hidden_size: int = 384,
        intermediate_size: int = 1024,
        num_hidden_layers: int = 12,
        attention_heads: int = 6,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1.0e-5,
        rope_theta: float = 10_000.0,
        initializer_range: float = 0.02,
        num_roles: int = 11,
        payload_dim: int = 8,
        action_horizon: int = 16,
        key_embedding: str = "sinusoid",
        max_keys: int = 64,
        action_loss_weight: float = 1.0,
        future_loss_weight: float = 0.5,
        token_abi_version: str = "physical-event-abi-0.2.0",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if hidden_size % attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if hidden_size % 2:
            raise ValueError("hidden_size must be even for deterministic Fourier keys")
        if key_embedding not in ("sinusoid", "learned"):
            raise ValueError("key_embedding must be 'sinusoid' or 'learned'")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.attention_heads = attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.initializer_range = initializer_range
        self.num_roles = num_roles
        self.payload_dim = payload_dim
        self.action_horizon = action_horizon
        self.key_embedding = key_embedding
        self.max_keys = max_keys
        self.action_loss_weight = action_loss_weight
        self.future_loss_weight = future_loss_weight
        self.token_abi_version = token_abi_version

    @classmethod
    def from_project_json(cls, path: str | Path) -> "PretrainingConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def llama_config(self) -> LlamaConfig:
        return LlamaConfig(
            vocab_size=1,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.attention_heads,
            num_key_value_heads=self.attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            hidden_act="silu",
            attention_bias=False,
            mlp_bias=False,
            initializer_range=self.initializer_range,
            tie_word_embeddings=False,
            use_cache=False,
        )


@dataclass
class PretrainingOutput(ModelOutput):
    loss: torch.Tensor | None = None
    action_loss: torch.Tensor | None = None
    future_loss: torch.Tensor | None = None
    action_predictions: torch.Tensor | None = None
    future_predictions: torch.Tensor | None = None


class PretrainingForTrajectoryPrediction(PreTrainedModel):
    """ICRT-derived trajectory learner with variable public readout queries.

    Transformer blocks, causal masking, RoPE, RMSNorm, and SwiGLU are owned by
    ``transformers.LlamaModel``. This class owns only continuous event adapters,
    variable action/future readouts, and an optional canonical-content input
    produced by external modality adapters.
    """

    config_class = PretrainingConfig
    base_model_prefix = "backbone"
    main_input_name = "role_ids"
    supports_gradient_checkpointing = True
    accepts_loss_kwargs = False

    def __init__(self, config: PretrainingConfig) -> None:
        super().__init__(config)
        h = config.hidden_size
        self.backbone = LlamaModel(config.llama_config())
        self.role_embedding = nn.Embedding(config.num_roles, h, padding_idx=0)
        self.payload_projector = nn.Linear(config.payload_dim, h, bias=False)
        half = h // 2
        key_frequencies = torch.exp(
            -math.log(10_000.0) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1)
        )
        # Persist this deterministic buffer. Transformers' low-memory
        # ``from_pretrained`` construction may allocate non-persistent buffers
        # without running their ordinary value construction, which would make
        # an otherwise exact checkpoint reload change the public-key encoding.
        self.register_buffer("key_frequencies", key_frequencies, persistent=True)

        # Probe arm only. The canonical `0.2.0` profile encodes public key
        # identity as a frozen Fourier code, so the world's only carrier of
        # "which channel/actuator is this" cannot adapt during training. The
        # learned arm replaces that code with a trainable table so a capacity
        # probe can attribute a multi-binding failure to the encoding rather
        # than to the budget. It is not part of the selected profile.
        self.key_table = (
            nn.Embedding(config.max_keys, h) if config.key_embedding == "learned" else None
        )

        self.action_head = nn.Linear(h, config.action_horizon)
        self.future_head = nn.Linear(h, 1)
        self.post_init()

        # The core is modality-free: it consumes continuous event embeddings and
        # never receives token ids, so LlamaModel's vocabulary table is
        # structurally dead weight rather than merely unused on some batches.
        # Left trainable it produces no gradient, and DDP then waits forever for
        # a reduction that never arrives. State that fact here instead of
        # relaxing the DDP contract with find_unused_parameters=True, which
        # would also hide a genuinely dead parameter appearing later.
        with torch.no_grad():
            self.backbone.embed_tokens.weight.zero_()
        self.backbone.embed_tokens.weight.requires_grad_(False)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def deterministic_key_embedding(self, key_ids: torch.Tensor) -> torch.Tensor:
        angles = key_ids.to(dtype=self.key_frequencies.dtype).unsqueeze(-1) * self.key_frequencies
        encoded = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        return encoded * self.config.initializer_range

    def key_identity_embedding(self, key_ids: torch.Tensor) -> torch.Tensor:
        """Public key identity under the configured encoding."""
        if self.key_table is None:
            return self.deterministic_key_embedding(key_ids)
        return self.key_table(key_ids.clamp_max(self.config.max_keys - 1))

    def embed_events(
        self,
        role_ids: torch.Tensor,
        key_ids: torch.Tensor,
        payloads: torch.Tensor,
        attention_mask: torch.Tensor,
        canonical_content_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = (
            self.role_embedding(role_ids)
            + self.payload_projector(payloads)
            + self.key_identity_embedding(key_ids).to(dtype=payloads.dtype)
        )
        if canonical_content_embeds is not None:
            if canonical_content_embeds.shape != embeddings.shape:
                raise ValueError(
                    "canonical_content_embeds must have shape "
                    "[batch, tokens, hidden_size] matching the event sequence"
                )
            embeddings = embeddings + canonical_content_embeds.to(dtype=embeddings.dtype)
        return embeddings * attention_mask.unsqueeze(-1).to(dtype=embeddings.dtype)

    @staticmethod
    def _masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=prediction.dtype)
        reduce_dimensions = tuple(range(1, mask.ndim))
        numerator = ((prediction - target.to(dtype=prediction.dtype)).abs() * mask).sum(
            dim=reduce_dimensions
        )
        denominator = mask.sum(dim=reduce_dimensions)
        valid = denominator > 0
        if not valid.any():
            return prediction.sum() * 0.0
        return (numerator[valid] / denominator[valid]).mean()

    def forward(
        self,
        role_ids: torch.Tensor,
        key_ids: torch.Tensor,
        position_ids: torch.Tensor,
        payloads: torch.Tensor,
        attention_mask: torch.Tensor,
        action_targets: torch.Tensor | None = None,
        action_target_mask: torch.Tensor | None = None,
        future_targets: torch.Tensor | None = None,
        future_target_mask: torch.Tensor | None = None,
        canonical_content_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> PretrainingOutput:
        inputs_embeds = self.embed_events(
            role_ids,
            key_ids,
            payloads,
            attention_mask,
            canonical_content_embeds,
        )
        hidden = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        action_predictions = torch.tanh(self.action_head(hidden))
        future_predictions = torch.tanh(self.future_head(hidden).squeeze(-1))

        action_loss = None
        future_loss = None
        total_loss = None
        if action_targets is not None and action_target_mask is not None:
            action_loss = self._masked_l1(action_predictions, action_targets, action_target_mask)
        if future_targets is not None and future_target_mask is not None:
            future_loss = self._masked_l1(future_predictions, future_targets, future_target_mask)
        if action_loss is not None or future_loss is not None:
            total_loss = hidden.sum() * 0.0
            if action_loss is not None:
                total_loss = total_loss + self.config.action_loss_weight * action_loss
            if future_loss is not None:
                total_loss = total_loss + self.config.future_loss_weight * future_loss

        return PretrainingOutput(
            loss=total_loss,
            action_loss=action_loss,
            future_loss=future_loss,
            action_predictions=action_predictions,
            future_predictions=future_predictions,
        )


def parameter_report(model: PretrainingForTrajectoryPrediction) -> dict[str, int]:
    groups = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "backbone": sum(parameter.numel() for parameter in model.backbone.parameters()),
    }
    groups["adapter_and_heads"] = groups["total"] - groups["backbone"]
    groups["frozen_unused_vocabulary"] = groups["total"] - groups["trainable"]
    return groups


def is_selected_profile_arm(config: PretrainingConfig) -> bool:
    """True for the canonical `0.2.0` profile, false for a probe arm.

    The drift guards below pin the selected parameterization exactly so an
    accidental architecture change cannot enter the lineage unnoticed. A
    deliberate probe arm must still be able to run, so it is identified here
    and exempted rather than forcing the pinned numbers to be edited.
    """
    return str(getattr(config, "key_embedding", "sinusoid")) == "sinusoid"


def assert_selected_parameter_report(
    report: dict[str, int], config: PretrainingConfig | None = None
) -> None:
    if config is not None and not is_selected_profile_arm(config):
        return
    expected = {
        "total": 21_257_489,
        # The vocabulary table is frozen; the modality-free core has no tokens.
        "trainable": 21_257_105,
        "backbone": 21_243_648,
        "adapter_and_heads": 13_841,
        "frozen_unused_vocabulary": 384,
    }
    if report != expected:
        raise AssertionError(f"selected parameterization drift: expected {expected}, got {report}")


def assert_selected_profile(config: PretrainingConfig) -> None:
    if not is_selected_profile_arm(config):
        return
    expected = {
        "hidden_size": 384,
        "intermediate_size": 1024,
        "num_hidden_layers": 12,
        "attention_heads": 6,
        "max_position_embeddings": 2048,
        "num_roles": 11,
        "payload_dim": 8,
        "action_horizon": 16,
    }
    actual = {key: getattr(config, key) for key in expected}
    if actual != expected:
        raise AssertionError(f"selected architecture drift: expected {expected}, got {actual}")
    supported_token_abis = {
        "physical-event-abi-0.2.0",
        "physical-event-abi-0.3.1",
    }
    if config.token_abi_version not in supported_token_abis:
        raise AssertionError(
            "selected interface drift: expected one of "
            f"{sorted(supported_token_abis)}, got {config.token_abi_version!r}"
        )
    llama = config.llama_config()
    if llama.num_key_value_heads != llama.num_attention_heads:
        raise AssertionError("GQA is not selected")
    rope_theta = getattr(llama, "rope_theta", None)
    if rope_theta is None:
        rope_theta = getattr(llama, "rope_parameters", {}).get("rope_theta")
    if rope_theta != 10_000.0 or llama.rms_norm_eps != 1.0e-5:
        raise AssertionError("RoPE/RMSNorm profile drift")
