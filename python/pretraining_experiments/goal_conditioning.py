"""Evaluate goal-sensitive behavior through the learner interface.

This module owns orchestration only. Rust owns the diagnostic dynamics,
learner-visible serialization, hidden goals, and the progress-classification
rule. The maintained Transformers model owns inference.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from typing import Any

import torch

import pretraining_world_py

from .data import tensorize
from .model import PretrainingConfig, PretrainingForTrajectoryPrediction


KINDS = (
    "witness",
    "fixed_goal_control",
    "state_predicts_goal_control",
    "hidden_goal_leakage_check",
    "renamed_witness",
)


def _score(summary: dict[str, Any], kind: str) -> dict[str, Any]:
    selected = [
        success
        for success, case_kind in zip(summary["success"], summary["case_kinds"])
        if case_kind == kind
    ]
    if not selected:
        raise ValueError(f"diagnostic has no cases of kind {kind!r}")
    successes = sum(bool(value) for value in selected)
    return {
        "successes": successes,
        "cases": len(selected),
        "rate": successes / len(selected),
    }


def _run_serialization(
    model: PretrainingForTrajectoryPrediction,
    *,
    device: torch.device,
    max_tokens: int,
    serialization_order: str,
    profiled: bool,
) -> dict[str, Any]:
    rollout = pretraining_world_py.GoalConditioningRolloutBatch(
        max_tokens=max_tokens,
        serialization_order=serialization_order,
        profiled=profiled,
    )
    model.eval()
    while not rollout.all_done():
        raw = rollout.learner_batch()
        tensors = tensorize(raw, device)
        with torch.no_grad():
            predictions = model(**tensors).action_predictions
        actions: list[list[float]] = []
        for row, (done, position) in enumerate(zip(raw["done"], raw["query_positions"])):
            if done:
                actions.append([])
            else:
                actions.append([float(predictions[row, position, 0].item())])
        rollout.step(actions)
    return dict(rollout.summary())


def _first_action(summary: dict[str, Any], case_id: str) -> int | None:
    index = summary["case_ids"].index(case_id)
    actions = summary["action_displacements"][index]
    return int(actions[0]) if actions else None


def _counterfactual_goal_sensitivity(summary: dict[str, Any]) -> float:
    left = _first_action(summary, "witness-left")
    right = _first_action(summary, "witness-right")
    return float(left == -1 and right == 1)


def _order_invariance(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_actions = dict(zip(left["case_ids"], left["action_displacements"]))
    right_actions = dict(zip(right["case_ids"], right["action_displacements"]))
    if left_actions.keys() != right_actions.keys():
        raise ValueError("serialization variants do not contain the same semantic cases")
    matches = sum(left_actions[case_id] == right_actions[case_id] for case_id in left_actions)
    return matches / len(left_actions)


def evaluate_model(
    model: PretrainingForTrajectoryPrediction,
    *,
    device: torch.device | str = "cpu",
    max_tokens: int = 64,
    headline_metric: float = 0.0,
    transfer: dict[str, float] | None = None,
    profiled: bool = False,
) -> dict[str, Any]:
    """Return classifier-compatible evidence plus representation diagnostics."""
    device = torch.device(device)
    model = model.to(device)
    canonical = _run_serialization(
        model,
        device=device,
        max_tokens=max_tokens,
        serialization_order="canonical",
        profiled=profiled,
    )
    permuted = _run_serialization(
        model,
        device=device,
        max_tokens=max_tokens,
        serialization_order="permuted",
        profiled=profiled,
    )
    diagnostics = {
        "witness": _score(canonical, "witness"),
        "fixed_goal_control": _score(canonical, "fixed_goal_control"),
        "state_predicts_goal_control": _score(canonical, "state_predicts_goal_control"),
        "hidden_goal_check": _score(canonical, "hidden_goal_leakage_check"),
        "renamed_witness": _score(canonical, "renamed_witness"),
        "counterfactual_goal_sensitivity": _counterfactual_goal_sensitivity(canonical),
        "presentation_order_invariance": _order_invariance(canonical, permuted),
    }
    checkpoint_evidence = {
        "headline_metric": float(headline_metric),
        "diagnostics": diagnostics,
        "transfer": transfer,
    }
    return {
        "checkpoint_evidence": checkpoint_evidence,
        "serialization_runs": {
            "canonical": canonical,
            "permuted": permuted,
        },
        "diagnostic_serialization_version": pretraining_world_py.versions()[
            "diagnostic_serialization"
        ],
        "token_abi_version": (
            pretraining_world_py.versions()["profiled_token_abi"]
            if profiled
            else pretraining_world_py.versions()["token_abi"]
        ),
        "interpretation": (
            "This is local diagnostic evidence. It is not project-level progress "
            "without a later held-out comparison against scratch and matched controls."
        ),
    }


def classify_against(
    previous: dict[str, Any],
    candidate: dict[str, Any],
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Use the Rust-owned decision rule; do not duplicate it in Python."""
    decision_json = pretraining_world_py.classify_goal_progress(
        previous_json=json.dumps(previous, sort_keys=True),
        candidate_json=json.dumps(candidate, sort_keys=True),
        thresholds_json=(json.dumps(thresholds, sort_keys=True) if thresholds else None),
    )
    return json.loads(decision_json)


def _train_probe_arm(
    config: PretrainingConfig,
    initial_state: dict[str, torch.Tensor],
    *,
    arm: str,
    updates: int,
    learning_rate: float,
    max_tokens: int,
) -> tuple[PretrainingForTrajectoryPrediction, dict[str, Any]]:
    raw = pretraining_world_py.generate_goal_conditioning_training_batch(
        arm=arm,
        max_tokens=max_tokens,
    )
    batch = tensorize(raw, "cpu")
    model = PretrainingForTrajectoryPrediction(config)
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.eval()
    with torch.no_grad():
        initial_loss = float(model(**batch).loss.item())
    started = time.perf_counter()
    model.train()
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        final_loss = float(model(**batch).loss.item())
    return model, {
        "arm": arm,
        "semantic_cases": int(raw["semantic_cases"]),
        "records_per_update": int(raw["records"]),
        "updates": updates,
        "supervised_records_processed": updates * int(raw["records"]),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "wall_seconds": elapsed,
        "records_per_second": updates * int(raw["records"]) / elapsed,
    }


def run_cpu_representation_probe(
    *,
    seed: int = 0,
    updates: int = 60,
    learning_rate: float = 2.0e-3,
    max_tokens: int = 64,
) -> dict[str, Any]:
    """Matched tiny-model probe of fixed versus renamed/reordered exposure."""
    if updates <= 0:
        raise ValueError("updates must be positive")
    torch.manual_seed(seed)
    config = PretrainingConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        attention_heads=4,
        max_position_embeddings=max_tokens,
        num_roles=11,
        payload_dim=8,
        action_horizon=16,
        token_abi_version="physical-event-abi-0.2.0",
    )
    initial = PretrainingForTrajectoryPrediction(config)
    initial_state = copy.deepcopy(initial.state_dict())
    baseline = evaluate_model(initial, max_tokens=max_tokens)
    arms: dict[str, Any] = {}
    for arm in ("fixed", "orbit"):
        trained, training = _train_probe_arm(
            config,
            initial_state,
            arm=arm,
            updates=updates,
            learning_rate=learning_rate,
            max_tokens=max_tokens,
        )
        evaluation = evaluate_model(trained, max_tokens=max_tokens)
        arms[arm] = {
            "training": training,
            "evaluation": evaluation,
            "decision_against_untrained": classify_against(
                baseline["checkpoint_evidence"],
                evaluation["checkpoint_evidence"],
            ),
        }
    return {
        "probe": "matched CPU representation probe",
        "seed": seed,
        "matched_conditions": {
            "same_initial_parameters": True,
            "same_semantic_cases": True,
            "same_records_per_update": True,
            "same_updates": updates,
            "same_learning_rate": learning_rate,
            "only_difference": (
                "the fixed arm repeats one presentation; the orbit arm replaces "
                "the duplicate with a consistently renamed and reordered presentation"
            ),
        },
        "untrained_baseline": baseline,
        "arms": arms,
        "interpretation": (
            "This probe tests representation sensitivity and local learnability. "
            "Neither arm establishes downstream transfer or project-level progress."
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run the goal-conditioning diagnostic on CPU through the real learner interface."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "artifacts" / "icrt-derived-small" / "model_config.json",
    )
    parser.add_argument("--model", type=Path, help="Optional save_pretrained model directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--headline-metric", type=float, default=0.0)
    parser.add_argument(
        "--profiled",
        action="store_true",
        help="prefix the learner-visible 0.3.0 interpretation declaration",
    )
    parser.add_argument(
        "--representation-probe",
        action="store_true",
        help="run the matched tiny-model fixed-versus-orbit CPU probe",
    )
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--previous-evidence", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.representation_probe:
        result = run_cpu_representation_probe(
            seed=args.seed,
            updates=args.updates,
            learning_rate=args.learning_rate,
            max_tokens=args.max_tokens,
        )
    else:
        torch.manual_seed(args.seed)
        if args.model:
            model = PretrainingForTrajectoryPrediction.from_pretrained(args.model)
            model_source = str(args.model.resolve())
        else:
            model = PretrainingForTrajectoryPrediction(PretrainingConfig.from_project_json(args.config))
            model_source = f"random initialization from {args.config.resolve()}"

        result = evaluate_model(
            model,
            device="cpu",
            max_tokens=args.max_tokens,
            headline_metric=args.headline_metric,
            profiled=args.profiled,
        )
        result["model_source"] = model_source
        result["random_seed"] = args.seed
        if args.previous_evidence:
            previous = _load_json(args.previous_evidence)
            if "checkpoint_evidence" in previous:
                previous = previous["checkpoint_evidence"]
            thresholds = _load_json(args.thresholds) if args.thresholds else None
            result["decision"] = classify_against(
                previous,
                result["checkpoint_evidence"],
                thresholds,
            )

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
