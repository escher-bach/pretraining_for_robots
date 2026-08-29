"""Evaluate the container process without collapsing distinct evidence levels.

Rust owns the process, hidden goals, transition rule, exhaustive oracle, and
serialization. This module only runs the maintained learner and summarizes
observable behavior. A single model evaluation is not a transfer comparison.
"""

from __future__ import annotations

import time
from typing import Any

import torch

import pretraining_world_py

from .data import tensorize
from .model import PretrainingForTrajectoryPrediction


KINDS = (
    "witness",
    "fixed_goal_control",
    "state_predicts_goal_control",
    "hidden_goal_leakage_check",
    "renamed_witness",
)


def _score(summary: dict[str, Any], kind: str) -> dict[str, float | int]:
    selected = [
        bool(success)
        for success, case_kind in zip(summary["success"], summary["case_kinds"])
        if case_kind == kind
    ]
    if not selected:
        raise ValueError(f"container suite has no cases of kind {kind!r}")
    successes = sum(selected)
    return {
        "successes": successes,
        "cases": len(selected),
        "rate": successes / len(selected),
    }


def first_action_diagnostic(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep first-choice sensitivity separate from complete success.

    An action can be a valid opening move and still be followed by a useless
    second move. This function therefore reports the opening contrast but does
    not use it as a substitute for the witness completion score.
    """
    rows = [
        {
            "case_id": case_id,
            "first_command": commands[0] if commands else None,
            "first_command_optimal": bool(optimal),
        }
        for case_id, case_kind, commands, optimal in zip(
            summary["case_ids"],
            summary["case_kinds"],
            summary["commands"],
            summary["first_action_optimal"],
        )
        if case_kind == "witness"
    ]
    if len(rows) != 2:
        raise ValueError(f"expected two witness cases, found {len(rows)}")
    commands = [row["first_command"] for row in rows]
    both_optimal = all(row["first_command_optimal"] for row in rows)
    changed_with_goal = None not in commands and commands[0] != commands[1]
    return {
        "cases": rows,
        "both_first_commands_optimal": both_optimal,
        "first_command_changed_with_goal": changed_with_goal,
        "passes_opening_contrast": both_optimal and changed_with_goal,
    }


def _order_invariance(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_commands = dict(zip(left["case_ids"], left["commands"]))
    right_commands = dict(zip(right["case_ids"], right["commands"]))
    if left_commands.keys() != right_commands.keys():
        raise ValueError("serialization variants do not contain the same semantic cases")
    matches = sum(
        left_commands[case_id] == right_commands[case_id]
        for case_id in left_commands
    )
    return matches / len(left_commands)


def _run_serialization(
    model: PretrainingForTrajectoryPrediction,
    *,
    device: torch.device,
    max_tokens: int,
    serialization_order: str,
) -> dict[str, Any]:
    rollout = pretraining_world_py.EvictionRolloutBatch(
        max_tokens=max_tokens,
        serialization_order=serialization_order,
        profiled=True,
    )
    model.eval()
    transitions = 0
    started = time.perf_counter()
    while not rollout.all_done():
        raw = rollout.learner_batch()
        if raw["token_abi_version"] != "physical-event-abi-0.3.1":
            raise RuntimeError("container evaluation requires the public profile declaration")
        tensors = tensorize(raw, device)
        with torch.no_grad():
            predictions = model(**tensors).action_predictions
        actions: list[list[float]] = []
        for row, (done, positions) in enumerate(
            zip(raw["done"], raw["query_positions"])
        ):
            if done:
                actions.append([])
                continue
            actions.append(
                [float(predictions[row, position, 0].item()) for position in positions]
            )
            transitions += 1
        rollout.step(actions)
    elapsed = time.perf_counter() - started
    summary = dict(rollout.summary())
    episodes = len(summary["case_ids"])
    summary["throughput"] = {
        "episodes": episodes,
        "transitions": transitions,
        "wall_seconds": elapsed,
        "episodes_per_second": episodes / elapsed,
        "transitions_per_second": transitions / elapsed,
    }
    return summary


def evaluate_model(
    model: PretrainingForTrajectoryPrediction,
    *,
    device: torch.device | str = "cpu",
    max_tokens: int = 64,
) -> dict[str, Any]:
    """Evaluate one checkpoint; do not infer transfer from this result alone."""
    device = torch.device(device)
    if model.config.token_abi_version != "physical-event-abi-0.3.1":
        raise ValueError(
            "container evaluation needs a model explicitly configured for "
            "physical-event-abi-0.3.1"
        )
    model = model.to(device)
    canonical = _run_serialization(
        model,
        device=device,
        max_tokens=max_tokens,
        serialization_order="canonical",
    )
    permuted = _run_serialization(
        model,
        device=device,
        max_tokens=max_tokens,
        serialization_order="permuted",
    )
    completion = {kind: _score(canonical, kind) for kind in KINDS}
    opening = first_action_diagnostic(canonical)
    order_invariance = _order_invariance(canonical, permuted)
    policy_evidence = {
        "witness": completion["witness"],
        "fixed_goal_control": completion["fixed_goal_control"],
        "state_predicts_goal_control": completion["state_predicts_goal_control"],
        "hidden_goal_check": completion["hidden_goal_leakage_check"],
        "renamed_witness": completion["renamed_witness"],
        "counterfactual_goal_sensitivity": float(opening["passes_opening_contrast"]),
        "presentation_order_invariance": order_invariance,
    }
    return {
        "complete_episode_success": completion,
        "first_action_diagnostic": opening,
        "presentation_order_invariance": order_invariance,
        "policy_evidence": policy_evidence,
        "hidden_goal_public_ceiling": 0.5,
        "serialization_runs": {
            "canonical": canonical,
            "permuted": permuted,
        },
        "process_version": pretraining_world_py.versions()["eviction_process"],
        "token_abi_version": pretraining_world_py.versions()["profiled_token_abi"],
        "interpretation": (
            "This is target-process behavior for one checkpoint. Meaningful project "
            "progress additionally requires matched learning curves against scratch "
            "and alternative pretraining, plus prior-skill retention. The compatible "
            "policy_evidence object may enter the existing progress classifier, which "
            "requires full witness success and the opening contrast together."
        ),
    }
