"""R10 finite-G0 seed-gate pilots and their immutable receipts.

This module is deliberately a source-family measurement surface.  It neither
loads nor scores the sealed transfer diagnostic, and a passing receipt is an
admission decision rather than a transfer claim.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
import tomllib
from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset
from transformers import Trainer, TrainerCallback, set_seed

from .data import (
    MODEL_FIELDS,
    assert_world_model_compatibility,
    generate_g0_mixed_torch_batch,
    g0_corpus_manifest,
    tensorize,
)
from .model import (
    PretrainingConfig,
    PretrainingForTrajectoryPrediction,
    assert_selected_parameter_report,
    assert_selected_profile,
    parameter_report,
)
from .train import training_arguments


FAMILIES = ("card04", "card03", "card02", "card05", "card06")
DISTINCT_EPISODE_COUNTS = {
    "card04": 16,
    "card03": 10,
    "card02": 9,
    "card05": 7,
    "card06": 32,
}
CONTRACT_HASHES = {
    "card04": "d975c3a646591ccf",
    "card03": "2442b372a18e1d66",
    "card02": "74b2d0da16ad3b31",
    "card05": "cbe39880124b9d2d",
    "card06": "76a08f38947c8cae",
}


def load_seed_gate_config(path: Path) -> dict[str, Any]:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    validate_seed_gate_config(config)
    return config


def validate_seed_gate_config(config: dict[str, Any]) -> None:
    """Reject drift from the fixed R10 scientific and execution contracts."""
    gate = config.get("seed_gate", {})
    run = config.get("run", {})
    model = config.get("model", {})
    expected = {
        "root_seed": 20260829,
        "family_order": list(FAMILIES),
        "max_updates": 64,
        "per_device_batch_size": 4,
        "learning_rate": 3.0e-4,
        "weight_decay": 0.1,
        "warmup_updates": 4,
        "sequence_length": 192,
        "evaluation_steps": [0, 16, 32, 48, 64],
        "timing_updates": 4,
        "required_macro_argmax": 0.80,
        "required_improvement": 0.25,
        "required_primary_case_kind_argmax": 0.60,
        "padding_strategy": "family_max_profiled_length_under_cap",
    }
    actual = {
        "root_seed": gate.get("root_seed"),
        "family_order": gate.get("family_order"),
        "max_updates": run.get("max_updates"),
        "per_device_batch_size": run.get("per_device_batch_size"),
        "learning_rate": run.get("learning_rate"),
        "weight_decay": run.get("weight_decay"),
        "warmup_updates": run.get("warmup_updates"),
        "sequence_length": model.get("sequence_length"),
        "evaluation_steps": gate.get("evaluation_steps"),
        "timing_updates": gate.get("timing_updates"),
        "required_macro_argmax": gate.get("required_macro_argmax"),
        "required_improvement": gate.get("required_improvement"),
        "required_primary_case_kind_argmax": gate.get("required_primary_case_kind_argmax"),
        "padding_strategy": gate.get("padding_strategy"),
    }
    if actual != expected:
        raise ValueError(f"R10 seed-gate contract drift: expected {expected}, got {actual}")
    execution = {
        "device": run.get("device"),
        "mixed_precision": run.get("mixed_precision"),
        "per_family_timeout_seconds": gate.get("per_family_timeout_seconds"),
        "total_timeout_seconds": gate.get("total_timeout_seconds"),
        "timing_updates_max_seconds": gate.get("timing_updates_max_seconds"),
        "timing_eval_max_seconds": gate.get("timing_eval_max_seconds"),
    }
    allowed_execution = {
        "cpu": {
            "device": "cpu", "mixed_precision": "no",
            "per_family_timeout_seconds": 90, "total_timeout_seconds": 540,
            "timing_updates_max_seconds": 3, "timing_eval_max_seconds": 10,
        },
        "cuda": {
            "device": "cuda", "mixed_precision": "fp16",
            "per_family_timeout_seconds": 120, "total_timeout_seconds": 720,
            "timing_updates_max_seconds": 10, "timing_eval_max_seconds": 10,
        },
    }
    if execution not in allowed_execution.values():
        raise ValueError(f"R10 seed-gate execution contract drift: {execution}")
    if gate.get("distinct_episode_counts") != DISTINCT_EPISODE_COUNTS:
        raise ValueError("R10 distinct public episode accounting drift")
    if gate.get("contract_hashes") != CONTRACT_HASHES:
        raise ValueError("R10 admitted contract hashes drift")
    if model.get("token_abi_version") != "physical-event-abi-0.3.1":
        raise ValueError("R10 must use the finite-G0 profiled token ABI")
    seeds = gate.get("family_seeds")
    expected_seeds = {
        family: {"init": 2026082901 + index, "train": 2026083001 + index, "eval": 2026083101 + index}
        for index, family in enumerate(FAMILIES)
    }
    if seeds != expected_seeds:
        raise ValueError("R10 per-family seed schedule drift")


def _manifest_batch(manifest: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Separate evaluator-only annotations from the standard model tensor ABI."""
    raw = manifest.get("batch", manifest)
    missing = sorted(set(MODEL_FIELDS).difference(raw))
    if missing:
        raise RuntimeError(f"G0 corpus manifest lacks model fields: {missing}")
    metadata = {key: value for key, value in manifest.items() if key not in MODEL_FIELDS and key != "batch"}
    metadata.update({key: value for key, value in raw.items() if key not in MODEL_FIELDS})
    return tensorize(raw), metadata


def compact_g0_corpus_manifest(
    family: str, sequence_cap: int
) -> tuple[dict[str, Any], int]:
    """Pad one family only to its audited maximum length under the fixed cap."""
    capped = g0_corpus_manifest(families=[family], max_tokens=sequence_cap)
    lengths = capped.get("lengths")
    if not isinstance(lengths, list) or not lengths:
        raise RuntimeError("G0 corpus manifest has no episode lengths")
    actual_tokens = max(int(length) for length in lengths)
    if actual_tokens <= 0 or actual_tokens > sequence_cap:
        raise RuntimeError(
            f"G0 family {family} requires {actual_tokens} tokens beyond cap {sequence_cap}"
        )
    if actual_tokens == sequence_cap:
        return capped, actual_tokens
    return (
        g0_corpus_manifest(families=[family], max_tokens=actual_tokens),
        actual_tokens,
    )


def _seed_gate_model_config(model_section: dict[str, Any]) -> PretrainingConfig:
    """Bind the selected core to the finite-G0 profiled ABI explicitly."""
    config = PretrainingConfig.from_project_json(Path(model_section["config"]))
    config.token_abi_version = str(model_section["token_abi_version"])
    assert_selected_profile(config)
    return config


def _episode_metadata(metadata: dict[str, Any], key: str, episodes: int) -> list[Any]:
    value = metadata.get(key)
    if not isinstance(value, list) or len(value) != episodes:
        raise RuntimeError(f"G0 corpus manifest requires one {key!r} value per episode")
    return value


@torch.no_grad()
def grouped_action_decision_argmax(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    families: list[str],
    case_kinds: list[str | list[str]],
    decision_groups: list[list[int | None]],
    primary_case_kinds: list[str] | None = None,
) -> dict[str, Any]:
    """Score one categorical choice across its query rows, not per row.

    Finite-G0 exposes alternatives as separate ActionQuery records and writes
    their selected/rejected target into slot zero.  Per-row threshold accuracy
    would reward predicting every alternative as rejected; this evaluator
    instead takes an argmax across each declared decision group.
    """
    if predictions.shape != targets.shape or targets.shape != target_mask.shape:
        raise ValueError("prediction, target, and mask shapes must agree")
    episodes, tokens, horizon = predictions.shape
    if horizon < 1 or len(families) != episodes or len(case_kinds) != episodes:
        raise ValueError("G0 metric metadata does not match the batch")
    if len(decision_groups) != episodes or any(len(row) != tokens for row in decision_groups):
        raise ValueError("action_decision_groups must match [episode, token]")
    if primary_case_kinds is None:
        primary_case_kinds = [
            str(value[0]) if isinstance(value, list) and value else str(value)
            for value in case_kinds
        ]
    if len(primary_case_kinds) != episodes:
        raise ValueError("primary_case_kinds must have one value per episode")

    by_kind: dict[str, list[bool]] = defaultdict(list)
    primary_by_kind: dict[str, list[bool]] = defaultdict(list)
    by_family: dict[str, list[bool]] = defaultdict(list)
    decision_count = 0
    for episode in range(episodes):
        local: dict[int, list[int]] = defaultdict(list)
        for token in range(tokens):
            if float(target_mask[episode, token, 0]) != 0.0:
                group = decision_groups[episode][token]
                if group is None or int(group) < 0:
                    raise ValueError("supervised action query lacks a decision group")
                local[int(group)].append(token)
        for positions in local.values():
            decision_count += 1
            predicted = max(positions, key=lambda position: float(predictions[episode, position, 0]))
            correct = float(targets[episode, predicted, 0]) > 0.0
            aliases = case_kinds[episode]
            labels = aliases if isinstance(aliases, list) else [aliases]
            if not labels:
                raise ValueError("each G0 episode must retain a case-kind label")
            for label in labels:
                by_kind[str(label)].append(correct)
            by_family[families[episode]].append(correct)
            primary_by_kind[primary_case_kinds[episode]].append(correct)

    def summarize(values: dict[str, list[bool]]) -> dict[str, dict[str, float | int]]:
        return {
            name: {"decisions": len(rows), "argmax": sum(rows) / len(rows)}
            for name, rows in sorted(values.items())
            if rows
        }

    per_kind = summarize(by_kind)
    if not per_kind:
        raise ValueError("G0 corpus has no supervised action decisions")
    per_primary = summarize(primary_by_kind)
    return {
        "decisions": decision_count,
        "macro_argmax": sum(float(row["argmax"]) for row in per_kind.values()) / len(per_kind),
        "by_case_kind": per_kind,
        "by_primary_case_kind": per_primary,
        "by_family": summarize(by_family),
    }


@torch.no_grad()
def evaluate_g0_corpus(model: torch.nn.Module, manifest: dict[str, Any]) -> dict[str, Any]:
    batch, metadata = _manifest_batch(manifest)
    episodes = int(batch["role_ids"].shape[0])
    families = [str(value) for value in _episode_metadata(metadata, "families", episodes)]
    aliases = _episode_metadata(metadata, "case_kinds", episodes)
    case_kinds = [
        [str(label) for label in value] if isinstance(value, list) else str(value)
        for value in aliases
    ]
    decision_groups = _episode_metadata(metadata, "action_decision_groups", episodes)
    primary = [str(value) for value in _episode_metadata(metadata, "primary_case_kinds", episodes)]
    device = next(model.parameters()).device
    model.eval()
    output = model(**{key: value.to(device) for key, value in batch.items()})
    return grouped_action_decision_argmax(
        output.action_predictions.detach().cpu(),
        batch["action_targets"],
        batch["action_target_mask"],
        families=families,
        case_kinds=case_kinds,
        decision_groups=decision_groups,
        primary_case_kinds=primary,
    )


class DistinctEpisodeDataset(IterableDataset):
    """Rust samples the deduplicated corpus; this class records actual cost."""

    def __init__(self, family: str, seed: int, max_tokens: int) -> None:
        self.family, self.seed, self.max_tokens = family, seed, max_tokens
        self.episode_presentations = 0
        self.action_query_targets = 0

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        index = 0
        while True:
            batch, _ = generate_g0_mixed_torch_batch(
                families=[self.family], weights=[1.0], seed=self.seed,
                start_index=index, batch_size=1, max_tokens=self.max_tokens,
            )
            self.episode_presentations += 1
            self.action_query_targets += int(batch["action_target_mask"][:, :, 0].sum().item())
            yield {name: batch[name][0] for name in MODEL_FIELDS}
            index += 1


@dataclass
class SeedGateCallback(TrainerCallback):
    model: torch.nn.Module
    manifest: dict[str, Any]
    evaluation_steps: set[int]
    started: float
    timeout_seconds: float
    evaluations: dict[int, dict[str, Any]]
    timed_out: bool = False

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.perf_counter() - self.started
        if elapsed > self.timeout_seconds:
            self.timed_out = True
            control.should_training_stop = True
            return control
        if state.global_step in self.evaluation_steps:
            self.evaluations[int(state.global_step)] = evaluate_g0_corpus(self.model, self.manifest)
        return control


def classify_seed_gate_pilot(
    *, result: dict[str, Any], gate: dict[str, Any]
) -> str:
    if not result.get("complete") or not result.get("finite") or not result.get("abi_and_bounds_ok"):
        return "unscored"
    if result.get("timed_out"):
        return "unscored"
    start = result["evaluations"][0]["macro_argmax"]
    final = result["evaluations"][int(gate["max_updates"])]["macro_argmax"]
    primary = result["evaluations"][int(gate["max_updates"])]["by_case_kind"]
    if (
        final >= float(gate["required_macro_argmax"])
        and final >= start + float(gate["required_improvement"])
        and primary
        and all(float(row["argmax"]) >= float(gate["required_primary_case_kind_argmax"]) for row in primary.values())
    ):
        return "admitted"
    return "inconclusive_not_admitted"


def seed_gate_timing_preflight(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    """Measure the fixed timing bounds before a family can be scored."""
    validate_seed_gate_config(config)
    gate, run, model_section = config["seed_gate"], config["run"], config["model"]
    family = FAMILIES[0]
    model_config = _seed_gate_model_config(model_section)
    assert_world_model_compatibility(model_config, profiled=True)
    use_cpu = str(run["device"]) == "cpu"
    if not use_cpu and not torch.cuda.is_available():
        raise RuntimeError("R10 CUDA seed-gate contract requires an available CUDA device")
    sequence_cap = int(model_section["sequence_length"])
    manifest, actual_tokens = compact_g0_corpus_manifest(family, sequence_cap)
    set_seed(int(gate["family_seeds"][family]["init"]))
    model = PretrainingForTrajectoryPrediction(model_config)
    if not use_cpu:
        model.to("cuda")

    started_eval = time.perf_counter()
    evaluate_g0_corpus(model, manifest)
    eval_seconds = time.perf_counter() - started_eval

    dataset = DistinctEpisodeDataset(
        family, int(gate["family_seeds"][family]["train"]), actual_tokens
    )
    started_updates = time.perf_counter()
    arguments = training_arguments(
        output_dir=output_root / "timing-preflight", run=run,
        max_steps=int(gate["timing_updates"]),
        per_device_batch_size=int(run["per_device_batch_size"]),
        warmup_steps=int(run["warmup_updates"]), save=False, use_cpu=use_cpu,
    )
    Trainer(model=model, args=arguments, train_dataset=dataset).train()
    update_seconds = time.perf_counter() - started_updates
    passed = (
        update_seconds <= float(gate["timing_updates_max_seconds"])
        and eval_seconds <= float(gate["timing_eval_max_seconds"])
    )
    result = {
        "passed": passed,
        "family": family,
        "device": str(run["device"]),
        "padding_strategy": gate["padding_strategy"],
        "sequence_cap": sequence_cap,
        "actual_tokens": actual_tokens,
        "updates": int(gate["timing_updates"]),
        "update_seconds": update_seconds,
        "update_limit_seconds": int(gate["timing_updates_max_seconds"]),
        "full_corpus_eval_seconds": eval_seconds,
        "full_corpus_eval_limit_seconds": int(gate["timing_eval_max_seconds"]),
        "classification": "preflight_passed" if passed else "unscored",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "timing-preflight-receipt.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def run_all_seed_gate_pilots(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    """Run the fixed R10 sequence, stopping before scoring after any hard cap."""
    validate_seed_gate_config(config)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    receipt: dict[str, Any] = {
        "row": "R10",
        "preflight": None,
        "pilots": [],
        "transfer_claim": False,
        "classification": "unscored",
    }
    if not bool(config["seed_gate"].get("r9_finalized", False)):
        receipt["classification"] = "blocked_r9_manifest_not_final"
        (output_root / "seed-gate-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        return receipt
    preflight = seed_gate_timing_preflight(config, output_root)
    receipt["preflight"] = preflight
    if not preflight["passed"]:
        (output_root / "seed-gate-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        return receipt
    for family in FAMILIES:
        if time.perf_counter() - started > float(config["seed_gate"]["total_timeout_seconds"]):
            receipt["classification"] = "unscored_total_timeout"
            break
        remaining = float(config["seed_gate"]["total_timeout_seconds"]) - (time.perf_counter() - started)
        receipt["pilots"].append(run_seed_gate_pilot(config, family, output_root, timeout_seconds=remaining))
    else:
        classifications = [pilot["classification"] for pilot in receipt["pilots"]]
        receipt["classification"] = "seed_gate_complete" if all(value == "admitted" for value in classifications) else "seed_gate_incomplete"
    receipt["elapsed_seconds"] = time.perf_counter() - started
    (output_root / "seed-gate-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed-contract R10 finite-G0 seed gate")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--family", choices=FAMILIES)
    args = parser.parse_args()
    if args.preflight_only and args.family:
        parser.error("--preflight-only and --family are mutually exclusive")
    config_path = args.config.resolve()
    config = load_seed_gate_config(config_path)
    configured_model = Path(config["model"]["config"])
    if not configured_model.is_absolute():
        config["model"]["config"] = str((config_path.parents[2] / configured_model).resolve())
    if args.preflight_only:
        result = seed_gate_timing_preflight(config, args.output_root.resolve())
    elif args.family:
        if not bool(config["seed_gate"].get("r9_finalized", False)):
            result = {"classification": "blocked_r9_manifest_not_final", "transfer_claim": False}
        else:
            preflight = seed_gate_timing_preflight(config, args.output_root.resolve())
            result = (
                run_seed_gate_pilot(config, args.family, args.output_root.resolve())
                if preflight["passed"]
                else {"classification": "unscored", "preflight": preflight, "transfer_claim": False}
            )
    else:
        result = run_all_seed_gate_pilots(config, args.output_root.resolve())
    print(json.dumps(result, sort_keys=True))


def run_seed_gate_pilot(
    config: dict[str, Any], family: str, output_root: Path, *, timeout_seconds: float | None = None
) -> dict[str, Any]:
    """Run one independently initialized selected-core family pilot."""
    validate_seed_gate_config(config)
    if family not in FAMILIES:
        raise ValueError(f"unknown R10 family {family!r}")
    gate, run, model_section = config["seed_gate"], config["run"], config["model"]
    family_seed = gate["family_seeds"][family]
    model_config = _seed_gate_model_config(model_section)
    assert_selected_parameter_report(parameter_report(PretrainingForTrajectoryPrediction(model_config)))
    assert_world_model_compatibility(model_config, profiled=True)
    use_cpu = str(run["device"]) == "cpu"
    if not use_cpu and not torch.cuda.is_available():
        raise RuntimeError("R10 CUDA seed-gate contract requires an available CUDA device")
    sequence_cap = int(model_section["sequence_length"])
    manifest, actual_tokens = compact_g0_corpus_manifest(family, sequence_cap)
    manifest_families = _episode_metadata(manifest, "families", len(manifest["role_ids"]))
    manifest_counts = _episode_metadata(manifest, "distinct_episode_counts", len(manifest_families))
    manifest_hashes = _episode_metadata(manifest, "contract_hashes", len(manifest_families))
    count_values = {int(value) for name, value in zip(manifest_families, manifest_counts) if name == family}
    hash_values = {str(value) for name, value in zip(manifest_families, manifest_hashes) if name == family}
    abi_and_bounds_ok = (
        count_values == {DISTINCT_EPISODE_COUNTS[family]}
        and hash_values == {CONTRACT_HASHES[family]}
        and manifest.get("token_abi_version") == "physical-event-abi-0.3.1"
        and manifest.get("interpretation_profile") == "finite-g0-discrete"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    set_seed(int(family_seed["init"]))
    model = PretrainingForTrajectoryPrediction(model_config)
    if not use_cpu:
        model.to("cuda")
    dataset = DistinctEpisodeDataset(family, int(family_seed["train"]), actual_tokens)
    started = time.perf_counter()
    evaluations = {0: evaluate_g0_corpus(model, manifest)}
    hard_timeout = min(float(gate["per_family_timeout_seconds"]), float(timeout_seconds) if timeout_seconds is not None else float("inf"))
    callback = SeedGateCallback(
        model=model, manifest=manifest, evaluation_steps=set(int(step) for step in gate["evaluation_steps"][1:]),
        started=started, timeout_seconds=hard_timeout, evaluations=evaluations,
    )
    arguments = training_arguments(
        output_dir=output_root / family, run=run, max_steps=int(run["max_updates"]),
        per_device_batch_size=int(run["per_device_batch_size"]), warmup_steps=int(run["warmup_updates"]),
        save=False, use_cpu=use_cpu,
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=dataset, callbacks=[callback])
    trainer.train()
    complete = int(trainer.state.global_step) == int(run["max_updates"])
    if complete and int(run["max_updates"]) not in evaluations:
        evaluations[int(run["max_updates"])] = evaluate_g0_corpus(model, manifest)
    result: dict[str, Any] = {
        "family": family, "contract_hash": CONTRACT_HASHES[family], "classification": "pending",
        "device": str(run["device"]),
        "seeds": dict(family_seed), "evaluation_support": "full_distinct_public_corpus",
        "padding_strategy": gate["padding_strategy"],
        "sequence_cap": sequence_cap, "actual_tokens": actual_tokens,
        "complete": complete, "finite": all(torch.isfinite(value).all().item() for value in model.parameters()),
        "abi_and_bounds_ok": abi_and_bounds_ok, "timed_out": callback.timed_out,
        "elapsed_seconds": time.perf_counter() - started, "evaluations": evaluations,
        "episode_presentations": dataset.episode_presentations,
        "scored_action_query_targets": dataset.action_query_targets,
        "transfer_claim": False,
    }
    result["classification"] = classify_seed_gate_pilot(result=result, gate={**gate, "max_updates": run["max_updates"]})
    (output_root / family / "seed-gate-receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
