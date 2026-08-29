"""Batched conversion from the Rust public-token boundary to torch tensors."""

from __future__ import annotations

from typing import Any

import torch

import pretraining_world_py


MODEL_FIELDS = (
    "role_ids",
    "key_ids",
    "position_ids",
    "payloads",
    "attention_mask",
    "action_targets",
    "action_target_mask",
    "future_targets",
    "future_target_mask",
)


def assert_world_model_compatibility(
    model_config: Any, *, profiled: bool | None = None
) -> None:
    """Fail before data generation when the model and Rust token ABI diverge."""
    versions = pretraining_world_py.versions()
    if profiled is None:
        profiled = model_config.token_abi_version == versions["profiled_token_abi"]
    expected = {
        "token_abi": model_config.token_abi_version,
        "role_count": model_config.num_roles,
        "payload_dim": model_config.payload_dim,
        "action_horizon": model_config.action_horizon,
    }
    actual = {
        "token_abi": versions["profiled_token_abi"] if profiled else versions["token_abi"],
        "role_count": versions["role_count"],
        "payload_dim": versions["payload_dim"],
        "action_horizon": versions["action_horizon"],
    }
    if actual != expected:
        raise RuntimeError(f"world/model ABI mismatch: expected {expected}, got {actual}")


def world_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "d_min": int(config["d_min"]),
        "d_max": int(config["d_max"]),
        "gain_min": float(config["gain_min"]),
        "gain_max": float(config["gain_max"]),
        "action_limit": float(config["action_limit"]),
        "calibration_pulse": float(config["calibration_pulse"]),
        "max_control_steps": int(config["max_control_steps"]),
        "profiled": bool(config.get("profiled", False)),
    }


def tensorize(raw: dict[str, Any], device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
    integer_fields = {"role_ids", "key_ids", "position_ids", "attention_mask"}
    tensors: dict[str, torch.Tensor] = {}
    for field in MODEL_FIELDS:
        dtype = torch.long if field in integer_fields else torch.float32
        tensors[field] = torch.tensor(raw[field], dtype=dtype, device=device)
    return tensors


def generate_torch_batch(
    *,
    seed: int,
    start_index: int,
    batch_size: int,
    max_tokens: int,
    world: dict[str, Any],
    device: torch.device | str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    raw = pretraining_world_py.generate_training_batch(
        seed=seed,
        start_index=start_index,
        batch_size=batch_size,
        max_tokens=max_tokens,
        **world_kwargs(world),
    )
    metadata = {key: value for key, value in raw.items() if key not in MODEL_FIELDS}
    return tensorize(raw, device), metadata


def _g0_extension_function(*names: str):
    """Return a declared G0 extension surface without guessing a fallback.

    The finite-G0 families have a different public profile from the legacy
    calibrated-monomial world.  Falling back to the latter would silently turn
    a seed-gate result into evidence about the wrong family, so absence of the
    rebuilt wheel is an explicit apparatus error.
    """
    for name in names:
        function = getattr(pretraining_world_py, name, None)
        if function is not None:
            return function
    joined = " or ".join(names)
    raise RuntimeError(
        f"installed pretraining_world_py lacks {joined}; rebuild the Rust extension"
    )


def generate_g0_mixed_torch_batch(
    *,
    families: list[str] | tuple[str, ...],
    weights: list[float] | tuple[float, ...],
    seed: int,
    start_index: int,
    batch_size: int,
    max_tokens: int,
    device: torch.device | str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Tensorize a Rust-owned, deduplicated finite-G0 mixture.

    ``families`` and ``weights`` are scheduler inputs only.  The returned
    model dictionary is restricted to ``MODEL_FIELDS`` so family identity,
    contract hashes, case labels, and generator indices cannot become learner
    features.
    """
    if not families or len(families) != len(weights):
        raise ValueError("families and weights must be nonempty and have equal length")
    if any(int(weight) != weight or int(weight) <= 0 for weight in weights):
        raise ValueError("every G0 mixture weight must be a positive integer")
    raw = _g0_extension_function("generate_g0_mixed_training_batch")(
        families=list(families),
        weights=[int(weight) for weight in weights],
        seed=int(seed),
        start_index=int(start_index),
        batch_size=int(batch_size),
        max_tokens=int(max_tokens),
    )
    missing = sorted(set(MODEL_FIELDS).difference(raw))
    if missing:
        raise RuntimeError(f"G0 batch is missing model fields: {missing}")
    metadata = {key: value for key, value in raw.items() if key not in MODEL_FIELDS}
    return tensorize(raw, device), metadata


def g0_corpus_manifest(
    *, families: list[str] | tuple[str, ...], max_tokens: int = 192
) -> dict[str, Any]:
    """Read the Rust manifest used for full-corpus evaluator-only scoring."""
    if not families:
        raise ValueError("families must be nonempty")
    if int(max_tokens) <= 0:
        raise ValueError("max_tokens must be positive")
    raw = _g0_extension_function("generate_g0_corpus_manifest", "g0_corpus_manifest")(
        families=list(families), max_tokens=int(max_tokens)
    )
    if not isinstance(raw, dict):
        raise RuntimeError("G0 corpus manifest must be a dictionary")
    return raw


def tensorize_rollout(raw: dict[str, Any], device: torch.device | str) -> dict[str, torch.Tensor]:
    return tensorize(raw, device)
