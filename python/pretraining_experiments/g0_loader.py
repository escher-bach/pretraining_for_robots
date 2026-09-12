"""Load the persisted finite-G0 corpus at the learner boundary.

The Rust corpus artifact intentionally stores evaluator and replay metadata
next to each learner transcript.  This module validates that artifact and
projects each episode to the tensor fields used by ``data.py`` plus loss-only
ActionQuery decision groups, as in the corrected R10 objective.
Metadata can be inspected separately, but is never returned by a dataset item
or a collated model batch.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, IO

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .data import MODEL_FIELDS, tensorize


CORPUS_SCHEMA_VERSION = 2
PAYLOAD_WIDTH = 8
ACTION_TARGET_WIDTH = 16
ROLE_COUNT = 11
CORPUS_MODEL_FIELDS = MODEL_FIELDS + ("action_decision_groups",)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _split_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("G0 example split must be a string")
    normalized = value.replace("-", "_").casefold()
    if normalized == "train":
        return "train"
    if normalized in {"heldout", "held_out"}:
        return "held_out"
    raise ValueError(f"unknown G0 example split {value!r}")


def _finite_vector(value: Any, width: int, field: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence of width {width}")
    if len(value) != width:
        raise ValueError(f"{field} must have width {width}, got {len(value)}")
    result: list[float] = []
    for item in value:
        result.append(_finite_number(item, field))
    return result


def _mask(value: Any, width: int, field: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence of width {width}")
    if len(value) != width:
        raise ValueError(f"{field} must have width {width}, got {len(value)}")
    result: list[float] = []
    for item in value:
        # Rust serializes these masks as bools.  Accepting integral 0/1 keeps
        # the loader compatible with ordinary JSON fixtures while rejecting
        # accidental target values or NaN masks.
        if isinstance(item, bool):
            result.append(float(item))
            continue
        if isinstance(item, (int, float)):
            try:
                number = float(item)
            except (OverflowError, ValueError):
                number = math.nan
            if math.isfinite(number) and number in (0.0, 1.0):
                result.append(number)
                continue
        raise ValueError(f"{field} must contain only boolean or 0/1 values")
    return result


def _scalar(value: Any, field: str) -> float:
    return _finite_number(value, field)


def _integer(value: Any, field: str, upper: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < 0 or (upper is not None and value >= upper):
        bound = f" in [0, {upper})" if upper is not None else " >= 0"
        raise ValueError(f"{field} must be{bound}")
    return int(value)


def _nonnegative_integer(value: Any, field: str) -> int:
    """Validate persisted evaluator metadata without exposing it to the model."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return int(value)


def _validate_token(token: Any, token_index: int) -> dict[str, Any]:
    if not isinstance(token, Mapping):
        raise ValueError(f"G0 token {token_index} must be an object")
    prefix = f"token {token_index}"
    required = {
        "role",
        "key",
        "event",
        "payload",
        "action_target",
        "action_mask",
        "future_target",
        "future_mask",
    }
    missing = sorted(required.difference(token))
    if missing:
        raise ValueError(f"{prefix} missing fields: {missing}")
    role = _integer(token["role"], f"{prefix}.role", ROLE_COUNT)
    key = _integer(token["key"], f"{prefix}.key", 1 << 16)
    event = _integer(token["event"], f"{prefix}.event", 1 << 16)
    payload = _finite_vector(token["payload"], PAYLOAD_WIDTH, f"{prefix}.payload")
    action_target = _finite_vector(
        token["action_target"], ACTION_TARGET_WIDTH, f"{prefix}.action_target"
    )
    action_mask = _mask(token["action_mask"], ACTION_TARGET_WIDTH, f"{prefix}.action_mask")
    future_mask_value = token["future_mask"]
    if isinstance(future_mask_value, bool):
        future_mask = float(future_mask_value)
    elif isinstance(future_mask_value, (int, float)):
        try:
            future_mask = float(future_mask_value)
        except (OverflowError, ValueError):
            future_mask = math.nan
        if not math.isfinite(future_mask) or future_mask not in (0.0, 1.0):
            raise ValueError(f"{prefix}.future_mask must be boolean or 0/1")
    else:
        raise ValueError(f"{prefix}.future_mask must be boolean or 0/1")
    return {
        "role": role,
        "key": key,
        "event": event,
        "payload": payload,
        "action_target": action_target,
        "action_mask": action_mask,
        "future_target": _scalar(token["future_target"], f"{prefix}.future_target"),
        "future_mask": future_mask,
    }


def _read_artifact(source: str | os.PathLike[str] | Mapping[str, Any] | IO[str]) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    if hasattr(source, "read"):
        value = json.load(source)  # type: ignore[arg-type]
    else:
        value = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("G0 corpus artifact must be a JSON object")
    return value


def _validate_artifact(source: str | os.PathLike[str] | Mapping[str, Any] | IO[str]) -> list[dict[str, Any]]:
    artifact = _read_artifact(source)
    version = artifact.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != CORPUS_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported G0 corpus schema_version {version!r}; expected {CORPUS_SCHEMA_VERSION}"
        )
    examples = artifact.get("examples")
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise ValueError("G0 corpus examples must be a sequence")
    validated: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        if not isinstance(example, Mapping):
            raise ValueError(f"G0 example {index} must be an object")
        split = _split_name(example.get("split"))
        tokens = example.get("tokens")
        if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)) or not tokens:
            raise ValueError(f"G0 example {index} tokens must be a nonempty sequence")
        validated_tokens = [_validate_token(token, token_index) for token_index, token in enumerate(tokens)]
        plan_length = _nonnegative_integer(
            example.get("plan_length"), f"G0 example {index}.plan_length"
        )
        metadata = {
            "split": split,
            "semantic_hash": example.get("semantic_hash"),
            "generator_seed": example.get("generator_seed"),
            "generator_index": example.get("generator_index"),
            "fingerprint": example.get("fingerprint"),
            # Plan length is evaluator/generator metadata.  It is deliberately
            # kept outside the learner-facing tensor projection below.
            "plan_length": plan_length,
        }
        validated.append({"tokens": validated_tokens, "metadata": metadata})
    return validated


def _tokens_to_tensors(tokens: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    raw = {
        "role_ids": [[token["role"] for token in tokens]],
        "key_ids": [[token["key"] for token in tokens]],
        "position_ids": [[token["event"] for token in tokens]],
        "payloads": [[token["payload"] for token in tokens]],
        "attention_mask": [[1] * len(tokens)],
        "action_targets": [[token["action_target"] for token in tokens]],
        "action_target_mask": [[token["action_mask"] for token in tokens]],
        "future_targets": [[token["future_target"] for token in tokens]],
        "future_target_mask": [[token["future_mask"] for token in tokens]],
    }
    result = {field: value.squeeze(0) for field, value in tensorize(raw).items()}
    # Same addressing as world-py's action_decision_groups: public event IDs
    # group alternatives; the model consumes these only after the backbone.
    result["action_decision_groups"] = torch.tensor(
        [token["event"] if token["role"] == 7 else -1 for token in tokens],
        dtype=torch.long,
    )
    return result


class G0CorpusDataset(Dataset[dict[str, torch.Tensor]]):
    """One learner-only tensor sequence per generated G0 episode."""

    def __init__(self, examples: Sequence[Mapping[str, Any]], split: str) -> None:
        self._records = [example for example in examples if example["metadata"]["split"] == split]
        self._tensors = [_tokens_to_tensors(record["tokens"]) for record in self._records]

    def __len__(self) -> int:
        return len(self._tensors)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # A clone prevents a training transform from mutating the cached
        # episode and consequently changing later epochs.
        return {field: value.clone() for field, value in self._tensors[index].items()}

    def metadata(self, index: int) -> dict[str, Any]:
        """Return evaluator metadata separately from learner tensors."""
        return dict(self._records[index]["metadata"])

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(int(tensor["attention_mask"].numel()) for tensor in self._tensors)


def load_g0_corpus(
    source: str | os.PathLike[str] | Mapping[str, Any] | IO[str],
    *,
    split: str = "train",
) -> G0CorpusDataset:
    """Validate and load ``pretraining-g0-corpus`` for one split."""
    selected_split = _split_name(split)
    return G0CorpusDataset(_validate_artifact(source), selected_split)


def collate_g0_episodes(
    episodes: Sequence[Mapping[str, torch.Tensor]],
    *,
    max_tokens: int | None = None,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Pad independent episodes into one model batch without concatenating them."""
    if not episodes:
        raise ValueError("cannot collate an empty G0 episode batch")
    missing = set(CORPUS_MODEL_FIELDS).difference(episodes[0])
    if missing:
        raise ValueError(f"G0 episode is missing model fields: {sorted(missing)}")
    lengths = [int(episode["attention_mask"].shape[0]) for episode in episodes]
    target_length = max(lengths)
    if max_tokens is not None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if target_length > max_tokens:
            raise ValueError(
                f"G0 episode length {target_length} exceeds max_tokens {max_tokens}; truncation is forbidden"
            )
        target_length = max_tokens
    result: dict[str, torch.Tensor] = {}
    for field in CORPUS_MODEL_FIELDS:
        values = []
        for episode, length in zip(episodes, lengths):
            if field not in episode:
                raise ValueError(f"G0 episode is missing model field {field!r}")
            value = episode[field]
            if value.shape[0] != length:
                raise ValueError(f"G0 field {field!r} has inconsistent episode length")
            values.append(value)
        padding_value = -1 if field == "action_decision_groups" else 0
        result[field] = pad_sequence(values, batch_first=True, padding_value=padding_value)
        if result[field].shape[1] < target_length:
            pad = [0] * (2 * (result[field].ndim - 1))
            pad[-1] = target_length - result[field].shape[1]
            result[field] = torch.nn.functional.pad(
                result[field], tuple(pad), value=padding_value
            )
    # A field-wise pad above is intentionally derived from the public mask;
    # no lengths or provenance values are added to the model dictionary.
    if device is not None:
        result = {field: value.to(device) for field, value in result.items()}
    return result


def g0_corpus_loader(
    source: str | os.PathLike[str] | Mapping[str, Any] | IO[str],
    *,
    split: str = "train",
    batch_size: int = 1,
    shuffle: bool = False,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Build a standard ``DataLoader`` over learner-only G0 episodes."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    dataset = load_g0_corpus(source, split=split)
    collate = partial(collate_g0_episodes, max_tokens=max_tokens)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, **kwargs)
