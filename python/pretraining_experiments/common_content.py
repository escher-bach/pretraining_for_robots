"""Versioned common content-token boundary for embodied learners.

This module is deliberately separate from :mod:`model`.  ``model.py`` keeps
the historical Finite-G0 replay contract; this module defines the forward
architecture for new sensor realizations.  Raw modality adapters produce the
same ``[B, tokens, H]`` content tensor, while role, public key, physical time,
and masks remain explicit structural inputs to the temporal core.

The adapter registry is a convenience for wiring reusable modality families.
Its names select code at the boundary and are never encoded into a token.
There is no family-id input and no eight-float placeholder path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaModel, PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput


COMMON_CONTENT_VERSION = "common-content-1.0"


def _binary_mask(mask: torch.Tensor, *, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(mask.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(mask.shape)}")
    if mask.dtype.is_floating_point:
        if not torch.isfinite(mask).all() or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError(f"{name} must be finite and binary")
    elif not torch.all((mask == 0) | (mask == 1)):
        raise ValueError(f"{name} must be binary")
    return mask.to(dtype=torch.bool)


@dataclass
class AdapterBatch:
    """Content tokens and validity mask emitted by one raw modality adapter."""

    content_tokens: torch.Tensor
    content_mask: torch.Tensor

    @property
    def tokens(self) -> torch.Tensor:
        """Short alias used when composing several adapter outputs."""
        return self.content_tokens


@dataclass
class RawContentStream:
    """One public, already time-ordered raw stream before adaptation.

    ``position_ids`` is serialization order for causal attention.  It is
    deliberately separate from ``time_values`` (physical capture/control
    time), because asynchronous sensors can share a physical timestamp while
    still having an unambiguous causal serialization order.
    """

    adapter_name: str
    values: torch.Tensor
    role_ids: torch.Tensor
    key_ids: torch.Tensor
    time_values: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor | None = None
    available_at: torch.Tensor | None = None


class ContinuousContentAdapter(nn.Module):
    """Adapt a variable-length stream of fixed-width numeric records to ``H``.

    The token count is the input's second dimension and is intentionally not
    fixed by this class.  ``input_dim`` describes one modality realization;
    it is not a family identity and never reaches the temporal core.
    """

    def __init__(self, input_dim: int, hidden_size: int, *, bias: bool = True) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_size <= 0:
            raise ValueError("input_dim and hidden_size must be positive")
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.projector = nn.Linear(input_dim, hidden_size, bias=bias)

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor | None = None
    ) -> AdapterBatch:
        if values.ndim != 3 or values.shape[-1] != self.input_dim:
            raise ValueError(
                f"values must have shape [batch, tokens, {self.input_dim}], got {tuple(values.shape)}"
            )
        if not torch.isfinite(values).all():
            raise ValueError("values must be finite")
        valid = (
            torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
            if mask is None
            else _binary_mask(mask, name="adapter mask", shape=tuple(values.shape[:2]))
        )
        tokens = self.projector(values)
        # Invalid records are inert before they reach the temporal core.  The
        # explicit mask remains available for padding and availability logic.
        tokens = torch.where(valid.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        return AdapterBatch(tokens, valid)


class G0ContentAdapter(ContinuousContentAdapter):
    """The eight-float symbolic realization as one ordinary content adapter."""

    def __init__(self, hidden_size: int, *, bias: bool = False) -> None:
        super().__init__(8, hidden_size, bias=bias)


class SensorContentAdapter(ContinuousContentAdapter):
    """A coherent numeric sensor stream with a declared raw width."""


class GoalContentAdapter(ContinuousContentAdapter):
    """Encode public goal records through the same content boundary."""


class PastActionContentAdapter(ContinuousContentAdapter):
    """Encode previously executed embodiment actions as history content."""


class ControlContentAdapter(ContinuousContentAdapter):
    """Encode public boundary markers that carry no sensor payload."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__(1, hidden_size, bias=True)


class ContentAdapterRegistry(nn.Module):
    """Registry of reusable modality adapters.

    Registry keys are local wiring names only.  They are not passed as model
    inputs, not embedded, and cannot communicate a world or family identity.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = hidden_size
        self.adapters = nn.ModuleDict()

    def register(self, name: str, adapter: ContinuousContentAdapter) -> None:
        if not name or "." in name:
            raise ValueError("adapter names must be nonempty and contain no dots")
        if adapter.hidden_size != self.hidden_size:
            raise ValueError("adapter hidden_size must match the registry")
        if name in self.adapters:
            raise ValueError(f"adapter {name!r} is already registered")
        self.adapters[name] = adapter

    def forward(
        self, name: str, values: torch.Tensor, mask: torch.Tensor | None = None
    ) -> AdapterBatch:
        try:
            adapter = self.adapters[name]
        except KeyError as exc:
            raise KeyError(f"unknown content adapter {name!r}") from exc
        return adapter(values, mask)


@dataclass(frozen=True)
class AdapterSpec:
    """Primitive adapter construction record stored in a bundle checkpoint."""

    name: str
    adapter_type: str
    input_dim: int
    bias: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "adapter_type": self.adapter_type,
            "input_dim": self.input_dim,
            "bias": self.bias,
        }


@dataclass(frozen=True)
class EmbodimentActionSpec:
    """Action dimensions and physical conventions for one target embodiment."""

    action_dim: int
    names: tuple[str, ...]
    units: tuple[str, ...]
    coordinate_frame: str
    semantics: tuple[str, ...]
    lower_bounds: tuple[float, ...] | None = None
    upper_bounds: tuple[float, ...] | None = None
    control_dt: float | None = None

    def __post_init__(self) -> None:
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if len(self.names) != self.action_dim:
            raise ValueError("names must have one entry per action dimension")
        if len(self.units) != self.action_dim:
            raise ValueError("units must have one entry per action dimension")
        if len(self.semantics) != self.action_dim:
            raise ValueError("semantics must have one entry per action dimension")
        if not self.coordinate_frame:
            raise ValueError("coordinate_frame must be declared")
        if (self.lower_bounds is None) != (self.upper_bounds is None):
            raise ValueError("lower_bounds and upper_bounds must be supplied together")
        if self.lower_bounds is not None and (
            len(self.lower_bounds) != self.action_dim
            or len(self.upper_bounds or ()) != self.action_dim
            or any(lo > hi for lo, hi in zip(self.lower_bounds, self.upper_bounds or ()))
        ):
            raise ValueError("action bounds must have one ordered pair per dimension")
        if self.control_dt is not None and self.control_dt <= 0:
            raise ValueError("control_dt must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_dim": self.action_dim,
            "names": list(self.names),
            "units": list(self.units),
            "coordinate_frame": self.coordinate_frame,
            "semantics": list(self.semantics),
            "lower_bounds": list(self.lower_bounds) if self.lower_bounds is not None else None,
            "upper_bounds": list(self.upper_bounds) if self.upper_bounds is not None else None,
            "control_dt": self.control_dt,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EmbodimentActionSpec":
        return cls(
            action_dim=int(value["action_dim"]),
            names=tuple(value["names"]),
            units=tuple(value["units"]),
            coordinate_frame=str(value["coordinate_frame"]),
            semantics=tuple(value["semantics"]),
            lower_bounds=(
                tuple(float(item) for item in value["lower_bounds"])
                if value.get("lower_bounds") is not None
                else None
            ),
            upper_bounds=(
                tuple(float(item) for item in value["upper_bounds"])
                if value.get("upper_bounds") is not None
                else None
            ),
            control_dt=(float(value["control_dt"]) if value.get("control_dt") is not None else None),
        )


def adapter_from_spec(spec: AdapterSpec, hidden_size: int) -> ContinuousContentAdapter:
    """Construct only the small declared adapter vocabulary used by bundles."""
    constructors = {
        "g0": lambda: G0ContentAdapter(hidden_size, bias=spec.bias),
        "sensor": lambda: SensorContentAdapter(spec.input_dim, hidden_size, bias=spec.bias),
        "goal": lambda: GoalContentAdapter(spec.input_dim, hidden_size, bias=spec.bias),
        "past_action": lambda: PastActionContentAdapter(
            spec.input_dim, hidden_size, bias=spec.bias
        ),
        "control": lambda: ControlContentAdapter(hidden_size),
    }
    if spec.adapter_type not in constructors:
        raise ValueError(f"unsupported adapter_type {spec.adapter_type!r}")
    adapter = constructors[spec.adapter_type]()
    if spec.adapter_type == "g0" and spec.input_dim != 8:
        raise ValueError("the g0 adapter has fixed raw width 8")
    if spec.adapter_type == "control" and spec.input_dim != 1:
        raise ValueError("the control adapter has fixed raw width 1")
    return adapter


def registry_from_specs(
    hidden_size: int, specs: list[AdapterSpec]
) -> ContentAdapterRegistry:
    registry = ContentAdapterRegistry(hidden_size)
    for spec in specs:
        registry.register(spec.name, adapter_from_spec(spec, hidden_size))
    return registry


class CommonContentConfig(PretrainedConfig):
    """Serializable configuration for the versioned content-token core."""

    model_type = "common_content_temporal_core"

    def __init__(
        self,
        *,
        hidden_size: int = 384,
        intermediate_size: int = 1024,
        num_hidden_layers: int = 12,
        attention_heads: int = 6,
        max_position_embeddings: int = 2048,
        role_vocab_size: int = 32,
        key_vocab_size: int = 256,
        rms_norm_eps: float = 1.0e-5,
        rope_theta: float = 10_000.0,
        initializer_range: float = 0.02,
        boundary_version: str = COMMON_CONTENT_VERSION,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if hidden_size <= 0 or hidden_size % attention_heads:
            raise ValueError("hidden_size must be positive and divisible by attention_heads")
        if role_vocab_size <= 0 or key_vocab_size <= 0:
            raise ValueError("role_vocab_size and key_vocab_size must be positive")
        if boundary_version != COMMON_CONTENT_VERSION:
            raise ValueError(f"unsupported boundary_version {boundary_version!r}")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.attention_heads = attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.role_vocab_size = role_vocab_size
        self.key_vocab_size = key_vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.initializer_range = initializer_range
        self.boundary_version = boundary_version

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
class CommonContentOutput(ModelOutput):
    last_hidden_state: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None


class CommonContentLlama(PreTrainedModel):
    """Causal temporal core with one required canonical content input.

    ``content_tokens`` already came from a declared adapter.  Structural
    addressing is explicit and additive only after adaptation; the core has no
    payload projector, sidecar, family id, or fixed G0 action head.
    """

    config_class = CommonContentConfig
    base_model_prefix = "backbone"
    main_input_name = "content_tokens"
    supports_gradient_checkpointing = True
    accepts_loss_kwargs = False

    def __init__(self, config: CommonContentConfig) -> None:
        super().__init__(config)
        h = config.hidden_size
        self.backbone = LlamaModel(config.llama_config())
        self.role_embedding = nn.Embedding(config.role_vocab_size, h, padding_idx=0)
        self.key_embedding = nn.Embedding(config.key_vocab_size, h, padding_idx=0)
        self.time_projector = nn.Linear(1, h)
        self.availability_projector = nn.Linear(1, h)
        self.content_norm = nn.LayerNorm(h, eps=config.rms_norm_eps)
        self.post_init()
        # Llama's vocabulary is structurally unused because the core consumes
        # continuous embeddings. Freezing it keeps checkpoints explicit and
        # avoids a dead trainable parameter in distributed training.
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

    def _inputs(
        self,
        content_tokens: torch.Tensor,
        role_ids: torch.Tensor,
        key_ids: torch.Tensor,
        time_values: torch.Tensor,
        attention_mask: torch.Tensor,
        content_mask: torch.Tensor,
        position_ids: torch.Tensor,
        available_at: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if content_tokens.ndim != 3 or content_tokens.shape[-1] != self.config.hidden_size:
            raise ValueError(
                "content_tokens must have shape [batch, tokens, hidden_size]"
            )
        batch_shape = tuple(content_tokens.shape[:2])
        if available_at is None:
            # Older callers only supplied physical time.  Treat the record as
            # available at capture time while the trajectory route publishes
            # the explicit public availability timestamp.
            available_at = time_values
        for name, tensor in (
            ("role_ids", role_ids),
            ("key_ids", key_ids),
            ("time_values", time_values),
            ("available_at", available_at),
            ("attention_mask", attention_mask),
            ("content_mask", content_mask),
            ("position_ids", position_ids),
        ):
            if tuple(tensor.shape) != batch_shape and name != "time_values":
                raise ValueError(f"{name} must have shape {batch_shape}")
        if tuple(time_values.shape) != batch_shape:
            raise ValueError(f"time_values must have shape {batch_shape}")
        if tuple(available_at.shape) != batch_shape:
            raise ValueError(f"available_at must have shape {batch_shape}")
        if tuple(position_ids.shape) != batch_shape:
            raise ValueError(f"position_ids must have shape {batch_shape}")
        if (
            not torch.isfinite(content_tokens).all()
            or not torch.isfinite(time_values).all()
            or not torch.isfinite(available_at).all()
        ):
            raise ValueError("content_tokens, time_values, and available_at must be finite")
        if role_ids.dtype.is_floating_point or key_ids.dtype.is_floating_point:
            raise ValueError("role_ids and key_ids must be integer tensors")
        if position_ids.dtype.is_floating_point:
            raise ValueError("position_ids must be an integer tensor")
        if torch.any(role_ids < 0) or torch.any(role_ids >= self.config.role_vocab_size):
            raise ValueError("role_ids contain an out-of-range value")
        if torch.any(key_ids < 0) or torch.any(key_ids >= self.config.key_vocab_size):
            raise ValueError("key_ids contain an out-of-range value")
        if torch.any(position_ids < 0) or torch.any(position_ids >= self.config.max_position_embeddings):
            raise ValueError("position_ids contain an out-of-range value")
        active = _binary_mask(attention_mask, name="attention_mask", shape=batch_shape)
        content = _binary_mask(content_mask, name="content_mask", shape=batch_shape)
        if torch.any(content & ~active):
            raise ValueError("content_mask may only mark attended tokens")
        role = self.role_embedding(role_ids)
        key = self.key_embedding(key_ids)
        time = self.time_projector(time_values.unsqueeze(-1).to(dtype=content_tokens.dtype))
        availability = self.availability_projector(
            available_at.unsqueeze(-1).to(dtype=content_tokens.dtype)
        )
        inputs = self.content_norm(content_tokens) + role + key + time + availability
        # Content masks make unavailable records inert; attention masks control
        # what the causal core can read. Padding therefore cannot leak through
        # structural embeddings either.
        inputs = torch.where(active.unsqueeze(-1), inputs, torch.zeros_like(inputs))
        inputs = torch.where(content.unsqueeze(-1), inputs, torch.zeros_like(inputs))
        return inputs, active

    def forward(
        self,
        content_tokens: torch.Tensor,
        role_ids: torch.Tensor,
        key_ids: torch.Tensor,
        time_values: torch.Tensor,
        attention_mask: torch.Tensor,
        content_mask: torch.Tensor,
        position_ids: torch.Tensor,
        available_at: torch.Tensor | None = None,
        **_: Any,
    ) -> CommonContentOutput:
        inputs, active = self._inputs(
            content_tokens,
            role_ids,
            key_ids,
            time_values,
            attention_mask,
            content_mask,
            position_ids,
            available_at,
        )
        outputs = self.backbone(
            inputs_embeds=inputs,
            attention_mask=active,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        hidden = torch.where(active.unsqueeze(-1), hidden, torch.zeros_like(hidden))
        return CommonContentOutput(last_hidden_state=hidden, attention_mask=active)


class EmbodimentActionDecoder(nn.Module):
    """Decode temporal hidden states into one embodiment's action space.

    ``action_dim`` is declared by the target embodiment.  There is no fixed
    horizon or G0 slot layout in this decoder; temporal chunking and validity
    are represented by the input token sequence and ``action_mask``.
    """

    def __init__(
        self,
        hidden_size: int,
        action_spec: EmbodimentActionSpec | int | None = None,
        *,
        action_dim: int | None = None,
        intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        if action_spec is None:
            action_spec = action_dim
        elif action_dim is not None:
            raise ValueError("provide action_spec or action_dim, not both")
        if action_spec is None:
            raise TypeError("action_spec is required")
        if isinstance(action_spec, int):
            # Kept as a compact convenience for direct experiments. Bundles
            # require EmbodimentActionSpec so units/frame/semantics serialize.
            action_spec = EmbodimentActionSpec(
                action_dim=action_spec,
                names=tuple(f"action_{i}" for i in range(action_spec)),
                units=tuple("normalized" for _ in range(action_spec)),
                coordinate_frame="declared-by-caller",
                semantics=tuple("unspecified" for _ in range(action_spec)),
            )
        if not isinstance(action_spec, EmbodimentActionSpec):
            raise TypeError("action_spec must be EmbodimentActionSpec")
        if hidden_size <= 0 or action_spec.action_dim <= 0:
            raise ValueError("hidden_size and action_spec.action_dim must be positive")
        inner = intermediate_size or hidden_size
        self.action_spec = action_spec
        self.action_dim = action_spec.action_dim
        if action_spec.lower_bounds is None:
            self.register_buffer("lower_bounds", torch.empty(0), persistent=True)
            self.register_buffer("upper_bounds", torch.empty(0), persistent=True)
        else:
            self.register_buffer(
                "lower_bounds", torch.tensor(action_spec.lower_bounds, dtype=torch.float32), persistent=True
            )
            self.register_buffer(
                "upper_bounds", torch.tensor(action_spec.upper_bounds, dtype=torch.float32), persistent=True
            )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, inner),
            nn.SiLU(),
            nn.Linear(inner, action_spec.action_dim),
        )

    def forward(self, hidden_states: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, tokens, hidden_size]")
        actions = self.head(hidden_states)
        if self.lower_bounds.numel():
            midpoint = (self.lower_bounds + self.upper_bounds) / 2
            half_range = (self.upper_bounds - self.lower_bounds) / 2
            actions = midpoint + half_range * torch.tanh(actions)
        if action_mask is None:
            return actions
        if tuple(action_mask.shape) == tuple(hidden_states.shape[:2]):
            valid = _binary_mask(
                action_mask, name="action_mask", shape=tuple(hidden_states.shape[:2])
            ).unsqueeze(-1)
        elif tuple(action_mask.shape) == tuple(actions.shape):
            valid = _binary_mask(
                action_mask, name="action_mask", shape=tuple(actions.shape)
            )
        else:
            raise ValueError("action_mask must be [batch, tokens] or [batch, tokens, action_dim]")
        return torch.where(valid, actions, torch.zeros_like(actions))


class CommonContentBundleConfig(PretrainedConfig):
    """Checkpoint config that reconstructs core, adapters, and action decoder."""

    model_type = "common_content_bundle"

    def __init__(
        self,
        *,
        core_config: dict[str, Any] | None = None,
        adapter_specs: list[dict[str, Any]] | None = None,
        action_spec: dict[str, Any] | None = None,
        action_loss: str = "masked_smooth_l1",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.core_config = core_config or CommonContentConfig().to_dict()
        self.adapter_specs = adapter_specs or []
        self.action_spec = action_spec or EmbodimentActionSpec(
            1, ("action_0",), ("normalized",), "declared-by-caller", ("unspecified",)
        ).to_dict()
        if action_loss not in {"masked_smooth_l1", "mse"}:
            raise ValueError(f"unsupported action_loss {action_loss!r}")
        self.action_loss = action_loss


@dataclass
class CommonContentBundleOutput(ModelOutput):
    loss: torch.Tensor | None = None
    last_hidden_state: torch.Tensor | None = None
    actions: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None


class CommonContentBundle(PreTrainedModel):
    """Trainable and serializable core + adapters + embodiment decoder."""

    config_class = CommonContentBundleConfig
    base_model_prefix = "core"
    main_input_name = "content_tokens"

    def __init__(self, config: CommonContentBundleConfig) -> None:
        super().__init__(config)
        core_config = CommonContentConfig(**config.core_config)
        self.core = CommonContentLlama(core_config)
        specs = [AdapterSpec(**value) for value in config.adapter_specs]
        self.adapters = registry_from_specs(core_config.hidden_size, specs)
        self.action_spec = EmbodimentActionSpec.from_dict(config.action_spec)
        self.decoder = EmbodimentActionDecoder(core_config.hidden_size, self.action_spec)
        self.action_loss = config.action_loss
        # Register this composite PreTrainedModel with the installed
        # Transformers save/load machinery after nested modules exist.
        self.post_init()

    @classmethod
    def from_parts(
        cls,
        core_config: CommonContentConfig,
        adapter_specs: list[AdapterSpec],
        action_spec: EmbodimentActionSpec,
        action_loss: str = "masked_smooth_l1",
    ) -> "CommonContentBundle":
        config = CommonContentBundleConfig(
            core_config=core_config.to_dict(),
            adapter_specs=[spec.to_dict() for spec in adapter_specs],
            action_spec=action_spec.to_dict(),
            action_loss=action_loss,
        )
        return cls(config)

    def assemble_streams(
        self, streams: list[RawContentStream]
    ) -> dict[str, torch.Tensor]:
        """Adapt and concatenate public streams into one canonical batch.

        Streams can have different raw widths and token counts.  Their
        structure is carried beside content and concatenated in the caller's
        declared serialization order; no adapter name is included in the
        resulting tensors.
        """
        if not streams:
            raise ValueError("at least one raw content stream is required")
        token_parts: list[torch.Tensor] = []
        role_parts: list[torch.Tensor] = []
        key_parts: list[torch.Tensor] = []
        time_parts: list[torch.Tensor] = []
        availability_parts: list[torch.Tensor] = []
        position_parts: list[torch.Tensor] = []
        content_masks: list[torch.Tensor] = []
        attention_masks: list[torch.Tensor] = []
        batch_size: int | None = None
        for stream in streams:
            encoded = self.adapters(stream.adapter_name, stream.values)
            expected = tuple(encoded.tokens.shape[:2])
            for name, tensor in (
                ("role_ids", stream.role_ids),
                ("key_ids", stream.key_ids),
                ("time_values", stream.time_values),
                ("position_ids", stream.position_ids),
                ):
                if tuple(tensor.shape) != expected:
                    raise ValueError(
                        f"{name} for {stream.adapter_name!r} must have shape {expected}"
                    )
            stream_available = (
                stream.time_values
                if stream.available_at is None
                else stream.available_at
            )
            if tuple(stream_available.shape) != expected:
                raise ValueError(
                    f"available_at for {stream.adapter_name!r} must have shape {expected}"
                )
            if batch_size is None:
                batch_size = expected[0]
            elif expected[0] != batch_size:
                raise ValueError("all raw streams must have the same batch size")
            stream_attention = (
                encoded.content_mask
                if stream.attention_mask is None
                else _binary_mask(
                    stream.attention_mask,
                    name="stream attention_mask",
                    shape=expected,
                )
            )
            stream_content = encoded.content_mask & stream_attention
            token_parts.append(encoded.tokens)
            role_parts.append(stream.role_ids)
            key_parts.append(stream.key_ids)
            time_parts.append(stream.time_values)
            availability_parts.append(stream_available)
            position_parts.append(stream.position_ids)
            content_masks.append(stream_content)
            attention_masks.append(stream_attention)
        return {
            "content_tokens": torch.cat(token_parts, dim=1),
            "role_ids": torch.cat(role_parts, dim=1),
            "key_ids": torch.cat(key_parts, dim=1),
            "time_values": torch.cat(time_parts, dim=1),
            "available_at": torch.cat(availability_parts, dim=1),
            "position_ids": torch.cat(position_parts, dim=1),
            "content_mask": torch.cat(content_masks, dim=1),
            "attention_mask": torch.cat(attention_masks, dim=1),
        }

    def _packed_content(
        self,
        raw_values: torch.Tensor,
        adapter_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Adapt a padded tensor envelope without exposing routing to the core."""
        if raw_values.ndim != 3 or adapter_ids.shape != raw_values.shape[:2]:
            raise ValueError("raw_values must be [batch, tokens, width] and adapter_ids [batch, tokens]")
        if raw_values.shape[-1] < max(
            (adapter.input_dim for adapter in self.adapters.adapters.values()), default=0
        ):
            raise ValueError("raw_values is narrower than a registered adapter")
        valid = _binary_mask(
            attention_mask,
            name="attention_mask",
            shape=tuple(raw_values.shape[:2]),
        )
        if adapter_ids.dtype.is_floating_point:
            raise ValueError("adapter_ids must be integer routing codes")
        names = tuple(self.adapters.adapters.keys())
        tokens = torch.zeros(
            *raw_values.shape[:2],
            self.core.config.hidden_size,
            dtype=self.core.backbone.embed_tokens.weight.dtype,
            device=raw_values.device,
        )
        content_mask = torch.zeros_like(valid)
        for code, name in enumerate(names):
            adapter = self.adapters.adapters[name]
            selected = valid & (adapter_ids == code)
            values = raw_values[..., : adapter.input_dim]
            encoded = adapter(values, selected)
            tokens = tokens + encoded.tokens
            content_mask = content_mask | selected
        unknown = valid & ((adapter_ids < 0) | (adapter_ids >= len(names)))
        if torch.any(unknown):
            raise ValueError("adapter_ids contain an unknown registered-adapter code")
        return tokens, content_mask

    def _init_weights(self, module: nn.Module) -> None:
        # Nested maintained modules initialize themselves in their constructors.
        return None

    def _masked_action_loss(
        self,
        actions: torch.Tensor,
        action_targets: torch.Tensor,
        action_target_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_mask = _binary_mask(
            action_target_mask,
            name="action_target_mask",
            shape=tuple(actions.shape),
        ).to(dtype=actions.dtype)
        residual = actions - action_targets.to(dtype=actions.dtype)
        if self.action_loss == "masked_smooth_l1":
            element_loss = F.smooth_l1_loss(
                residual, torch.zeros_like(residual), reduction="none"
            )
        else:
            element_loss = residual.square()
        return (element_loss * target_mask).sum() / target_mask.sum().clamp_min(1.0)

    def forward(
        self,
        content_tokens: torch.Tensor | None = None,
        role_ids: torch.Tensor | None = None,
        key_ids: torch.Tensor | None = None,
        time_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        content_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        raw_values: torch.Tensor | None = None,
        adapter_ids: torch.Tensor | None = None,
        action_targets: torch.Tensor | None = None,
        action_target_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CommonContentBundleOutput:
        if raw_values is not None or adapter_ids is not None:
            if raw_values is None or adapter_ids is None:
                raise ValueError("raw_values and adapter_ids must be supplied together")
            if attention_mask is None:
                raise ValueError("attention_mask is required with packed raw values")
            content_tokens, content_mask = self._packed_content(
                raw_values, adapter_ids, attention_mask
            )
        required = {
            "role_ids": role_ids,
            "key_ids": key_ids,
            "time_values": time_values,
            "attention_mask": attention_mask,
            "content_mask": content_mask,
            "position_ids": position_ids,
            "content_tokens": content_tokens,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"missing common-content inputs: {', '.join(missing)}")
        core_output = self.core(
            content_tokens=content_tokens,
            role_ids=role_ids,
            key_ids=key_ids,
            time_values=time_values,
            attention_mask=attention_mask,
            content_mask=content_mask,
            position_ids=position_ids,
            **kwargs,
        )
        actions = self.decoder(core_output.last_hidden_state, action_mask)
        loss = None
        if action_targets is not None or action_target_mask is not None:
            if action_targets is None or action_target_mask is None:
                raise ValueError("action_targets and action_target_mask must be supplied together")
            if action_targets.shape != actions.shape or action_target_mask.shape != actions.shape:
                raise ValueError("action targets and mask must match decoded action shape")
            loss = self._masked_action_loss(actions, action_targets, action_target_mask)
        return CommonContentBundleOutput(
            loss=loss,
            last_hidden_state=core_output.last_hidden_state,
            actions=actions,
            attention_mask=core_output.attention_mask,
        )

    def forward_streams(
        self,
        streams: list[RawContentStream],
        *,
        action_mask: torch.Tensor | None = None,
        action_targets: torch.Tensor | None = None,
        action_target_mask: torch.Tensor | None = None,
    ) -> CommonContentBundleOutput:
        """Run raw public streams and optionally compute body-action MSE."""
        batch = self.assemble_streams(streams)
        output = self(action_mask=action_mask, **batch)
        if action_targets is None or action_target_mask is None:
            return output
        if action_targets.shape != output.actions.shape:
            raise ValueError("action_targets must match decoded action shape")
        if action_target_mask.shape != output.actions.shape:
            raise ValueError("action_target_mask must match decoded action shape")
        loss = self._masked_action_loss(output.actions, action_targets, action_target_mask)
        # ModelOutput is immutable only by convention; attach the optional
        # loss in a returned object while keeping the core output fields.
        return CommonContentBundleOutput(
            last_hidden_state=output.last_hidden_state,
            actions=output.actions,
            attention_mask=output.attention_mask,
            loss=loss,
        )
