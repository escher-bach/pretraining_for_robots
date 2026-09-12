"""Compact CPU readiness audit for the first trajectory training system.

The audit intentionally treats the generator's private receipt as evaluator
evidence only.  Learner-facing checks use ``TrajectoryEpisode.learner_bundle``
and public histories; the receipt is used for rank, conditioning, clipping,
and reachability diagnostics that cannot be inferred from one public rollout.

Run from the repository root with::

    python tools/audit_training_system.py

The command always writes a compact JSON receipt and a human-readable trace to
``artifacts/first-training-system``.  A blocked receipt is a useful result:
it names the smallest repair needed before a first learner run.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product
import argparse
import inspect
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from pretraining_experiments.trajectory_world import (
    CalibratedReachConfig,
    CalibratedReachRollout,
    PublicEvent,
    PublicHistory,
    PublicHistoryTeacher,
    TrajectoryEpisode,
    corpus_fingerprint,
    generate_episode,
    teacher_public_only,
    fixed_reactive_action,
    WORLD_VERSION,
    TEACHER_VERSION,
)


OUTPUT_ROOT = ROOT / "artifacts" / "first-training-system"
ABS_TOL = 1.0e-7


def _finite_array(values: Iterable[float]) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("public vector is not finite and one-dimensional")
    return result


def _public_key_scan(value: Any, path: str = "") -> list[str]:
    """Return paths where learner-facing data uses a private field name."""

    forbidden = {
        "private_audit",
        "initial_state",
        "target_state",
        "sensor_matrix",
        "sensor_offset",
        "body_map",
        "dynamics_gain",
        "action_to_sensor_dt",
        "seed",
        "disturbance_start_step",
    }
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if str(key) in forbidden:
                found.append(child)
            found.extend(_public_key_scan(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_public_key_scan(item, f"{path}[{index}]"))
    return found


def _events_before(events: tuple[PublicEvent, ...], index: int) -> PublicHistory:
    return PublicHistory.from_events(events[: index + 1])


def _action_bounds(action_dim: int, config: CalibratedReachConfig) -> np.ndarray:
    return np.full((action_dim, 2), (-config.action_limit, config.action_limit), dtype=np.float64)


def _calibration_arrays(episode: TrajectoryEpisode) -> tuple[np.ndarray, np.ndarray]:
    actions = [event for event in episode.events if event.kind == "calibration_action"]
    observations = [event for event in episode.events if event.kind == "calibration_observation"]
    initial = next(event for event in episode.events if event.kind == "observation")
    if not actions or len(observations) != len(actions):
        raise ValueError(
            "calibration must publish one observation after every public pulse; "
            "the initial task observation is the prefix anchor"
        )
    action_matrix = np.asarray([event.action for event in actions], dtype=np.float64)
    observation_matrix = np.asarray(
        [initial.values, *[event.values for event in observations]], dtype=np.float64
    )
    if observation_matrix.shape[0] != action_matrix.shape[0] + 1:
        raise ValueError("calibration observation/action lengths do not align")
    return action_matrix, observation_matrix


def _teacher_targets(episode: TrajectoryEpisode, config: CalibratedReachConfig) -> list[np.ndarray]:
    teacher = PublicHistoryTeacher(control_gain=config.control_gain)
    targets = {target.query_event_id: target for target in episode.supervision}
    result: list[np.ndarray] = []
    for index, event in enumerate(episode.events):
        if event.kind != "action_query":
            continue
        target = targets.get(event.event_id)
        if target is None:
            raise ValueError(f"missing supervision for query {event.event_id}")
        bounds = _action_bounds(len(target.action), config)
        expected = teacher.action_for(_events_before(episode.events, index), bounds)
        actual = _finite_array(target.action)
        if actual.shape != expected.shape or not np.allclose(actual, expected, atol=ABS_TOL, rtol=ABS_TOL):
            raise ValueError(f"query {event.event_id} is not supervised by the public teacher")
        result.append(actual)
    if len(result) != len(episode.supervision):
        raise ValueError("supervision contains an unvisited query")
    return result


def _public_structure_checks(
    episode: TrajectoryEpisode, config: CalibratedReachConfig
) -> dict[str, Any]:
    if not episode.events or episode.events[0].kind != "reset":
        raise ValueError("public episode does not begin with reset")
    if episode.events[-1].kind != "episode_end":
        raise ValueError("public episode does not end with episode_end")
    public = episode.learner_bundle()
    leaked = _public_key_scan(public)
    if leaked:
        raise ValueError(f"private fields reached learner bundle: {leaked}")
    if "private_audit" in public:
        raise ValueError("learner bundle contains private_audit")
    for left, right in zip(episode.events, episode.events[1:]):
        if left.event_id >= right.event_id:
            raise ValueError("event ids are not strictly increasing")
        if right.time < left.time - ABS_TOL:
            raise ValueError("physical event time moved backwards")
    for event in episode.events:
        if event.available_at is not None and event.available_at < event.time - ABS_TOL:
            raise ValueError("event availability precedes physical time")
        if event.available_at is not None and abs(event.available_at - event.time) > ABS_TOL:
            raise ValueError(
                "the first-system learner path supports synchronous availability only; "
                "a delayed event requires an explicit availability mask"
            )
        if event.values:
            _finite_array(event.values)
        if event.action:
            action = _finite_array(event.action)
            if np.any(np.abs(action) > config.action_limit + ABS_TOL):
                raise ValueError(f"action event {event.event_id} exceeds declared bounds")
    for target in episode.supervision:
        action = _finite_array(target.action)
        if not np.all(np.asarray(target.action_valid, dtype=bool)):
            raise ValueError("first-run actions must be fully executable, not partially padded")
        if np.any(np.abs(action) > config.action_limit + ABS_TOL):
            raise ValueError(f"supervised action {target.query_event_id} exceeds bounds")
    return {
        "event_count": len(episode.events),
        "query_count": len(episode.supervision),
        "private_fields_in_learner_bundle": leaked,
        "public_teacher_signature": teacher_public_only(),
        "teacher_parameters": list(inspect.signature(PublicHistoryTeacher.action_for).parameters),
        "availability_contract": "synchronous_available_at_equals_physical_time",
    }


def _numerical_checks(episode: TrajectoryEpisode, config: CalibratedReachConfig) -> dict[str, Any]:
    private = episode.private_audit
    sensor = np.asarray(private["sensor_matrix"], dtype=np.float64)
    body = np.asarray(private["body_map"], dtype=np.float64)
    composite = np.asarray(private["action_to_sensor_dt"], dtype=np.float64)
    calibration, observations = _calibration_arrays(episode)
    deltas = np.diff(observations, axis=0)
    estimate, _, rank, singular_values = np.linalg.lstsq(calibration, deltas, rcond=None)
    residual = float(np.linalg.norm(calibration @ estimate - deltas))
    calibration_rank = int(np.linalg.matrix_rank(calibration))
    calibration_singular = np.linalg.svd(calibration, compute_uv=False)
    sensor_singular = np.linalg.svd(sensor, compute_uv=False)
    body_singular = np.linalg.svd(body, compute_uv=False)
    composite_singular = np.linalg.svd(composite, compute_uv=False)
    if sensor.shape[1] != 2 or int(np.linalg.matrix_rank(sensor)) != 2:
        raise ValueError("sensor map is not full column rank")
    if body.shape[0] != 2 or int(np.linalg.matrix_rank(body)) != 2:
        raise ValueError("body map is not full row rank")
    if calibration_rank != body.shape[1] or rank != body.shape[1]:
        raise ValueError("public calibration does not identify every actuator")
    if calibration_singular[-1] <= 1.0e-8:
        raise ValueError("calibration action matrix is numerically singular")
    if sensor_singular[0] / sensor_singular[-1] >= 3.5:
        raise ValueError("sensor map is poorly conditioned")
    if body_singular[0] / body_singular[-1] >= 4.0:
        raise ValueError("body map is poorly conditioned")
    # A four-actuator body deliberately has a two-dimensional nullspace.  Only
    # the two nonzero observable singular values participate in conditioning.
    if composite_singular[1] <= 1.0e-8:
        raise ValueError("action-to-sensor map has lost its two observable axes")
    if not np.allclose(estimate.T, composite, atol=1.0e-7, rtol=1.0e-6):
        raise ValueError("public least-squares calibration disagrees with audited composite map")
    # With a 4-actuator, 2D body the nullspace is a real semantic fact.  The
    # teacher declares minimum-L2 effort, so one pseudoinverse representative
    # is valid; the audit records the nullity rather than pretending the map is
    # invertible.
    nullity = int(composite.shape[1] - np.linalg.matrix_rank(composite))
    if int(private["action_dim"]) == 4 and nullity != 2:
        raise ValueError(f"four-actuator body expected nullity two, got {nullity}")
    if int(private["action_dim"]) == 2 and nullity != 0:
        raise ValueError(f"two-actuator body expected nullity zero, got {nullity}")
    calibration_clip = bool(np.max(np.abs(np.asarray([e.action for e in episode.events if e.kind == "calibration_action"]))) > config.action_limit + ABS_TOL)
    if calibration_clip:
        raise ValueError("calibration command was clipped")
    initial = np.asarray(next(event for event in episode.events if event.kind == "observation").values)
    goal_event = next(event for event in reversed(episode.events) if event.kind == "goal")
    goal = np.asarray(goal_event.values)
    if goal_event.key == "relative_goal":
        anchors = [
            event for event in episode.events
            if event.kind == "observation" and event.time <= goal_event.time + ABS_TOL
        ]
        goal = np.asarray(anchors[-1].values) + goal
    final = np.asarray(next(event for event in reversed(episode.events) if event.kind == "observation").values)
    initial_error = float(np.linalg.norm(goal - initial))
    final_error = float(np.linalg.norm(goal - final))
    if not np.isfinite(final_error) or final_error >= initial_error + ABS_TOL:
        raise ValueError(f"public teacher did not reduce terminal sensor error ({initial_error} -> {final_error})")
    sensor_rmse = final_error / np.sqrt(sensor.shape[0])
    initial_sensor_rmse = initial_error / np.sqrt(sensor.shape[0])
    return {
        "sensor_rank": int(np.linalg.matrix_rank(sensor)),
        "sensor_condition": float(sensor_singular[0] / sensor_singular[-1]),
        "body_rank": int(np.linalg.matrix_rank(body)),
        "body_condition": float(body_singular[0] / body_singular[-1]),
        "composite_rank": int(np.linalg.matrix_rank(composite)),
        "action_nullity": nullity,
        "calibration_rank": calibration_rank,
        "calibration_smallest_singular": float(calibration_singular[-1]),
        "calibration_residual": residual,
        "calibration_clipped": calibration_clip,
        "initial_sensor_error": initial_error,
        "final_sensor_error": final_error,
        "initial_sensor_rmse": float(initial_sensor_rmse),
        "final_sensor_rmse": float(sensor_rmse),
        "teacher_reduced_error": True,
    }


def _permute_episode(episode: TrajectoryEpisode, permutation: np.ndarray) -> TrajectoryEpisode:
    def permute_event(event: PublicEvent) -> PublicEvent:
        action = tuple(np.asarray(event.action)[permutation]) if event.action else event.action
        valid = tuple(np.asarray(event.action_valid)[permutation]) if event.action_valid else event.action_valid
        return replace(event, action=action, action_valid=valid)

    return replace(
        episode,
        events=tuple(permute_event(event) for event in episode.events),
        supervision=tuple(
            replace(
                target,
                action=tuple(np.asarray(target.action)[permutation]),
                action_valid=tuple(np.asarray(target.action_valid)[permutation]),
            )
            for target in episode.supervision
        ),
    )


def _sensor_permute_episode(episode: TrajectoryEpisode, permutation: np.ndarray) -> TrajectoryEpisode:
    def permute_event(event: PublicEvent) -> PublicEvent:
        if not event.values:
            return event
        return replace(
            event,
            values=tuple(np.asarray(event.values)[permutation]),
            value_valid=tuple(np.asarray(event.value_valid)[permutation]),
        )

    return replace(episode, events=tuple(permute_event(event) for event in episode.events))


def _information_ablations(episode: TrajectoryEpisode, config: CalibratedReachConfig) -> dict[str, bool]:
    """Check teacher API preconditions, not calibration's control necessity."""

    query_index = next(index for index, event in enumerate(episode.events) if event.kind == "action_query")
    history_events = list(episode.events[: query_index + 1])
    query = next(target for target in episode.supervision if target.query_event_id == episode.events[query_index].event_id)
    bounds = _action_bounds(len(query.action), config)
    teacher = PublicHistoryTeacher(control_gain=config.control_gain)

    def rejects(events: list[PublicEvent]) -> bool:
        try:
            teacher.action_for(PublicHistory.from_events(events), bounds)
        except (ValueError, np.linalg.LinAlgError):
            return True
        return False

    without_calibration = [
        event for event in history_events
        if event.kind not in {"calibration_action", "calibration_observation"}
    ]
    without_goal = [event for event in history_events if event.kind != "goal"]
    without_action = [event for event in history_events if event.kind != "calibration_action"]
    return {
        "calibration_ablation_rejected": rejects(without_calibration),
        "goal_ablation_rejected": rejects(without_goal),
        "calibration_action_ablation_rejected": rejects(without_action),
    }


def _signed_body_information_witness(config: CalibratedReachConfig) -> dict[str, Any]:
    """B and -B have identical observations under opposite public actions.

    This is an executable coordinate transformation of a generated trajectory,
    not a claim that a teacher's missing-field exception proves necessity.
    Removing calibration action content makes the first-query histories equal,
    while their public teachers require different body-local commands.
    """
    episode = generate_episode(4321, config=config, sensor_dim=8, action_dim=2, goal_mode="absolute")
    flipped = replace(
        episode,
        events=tuple(replace(event, action=tuple(-value for value in event.action)) for event in episode.events),
        supervision=tuple(replace(target, action=tuple(-value for value in target.action)) for target in episode.supervision),
    )
    original_actions = _teacher_targets(episode, config)
    flipped_actions = _teacher_targets(flipped, config)
    index = next(i for i, event in enumerate(episode.events) if event.kind == "action_query")
    def without_calibration_actions(value: TrajectoryEpisode) -> list[dict[str, Any]]:
        return [event.public_dict() for event in value.events[:index + 1] if event.kind != "calibration_action"]
    coarsened_equal = without_calibration_actions(episode) == without_calibration_actions(flipped)
    action_difference = float(np.linalg.norm(original_actions[0] - flipped_actions[0]))
    opposite_actions = bool(np.allclose(np.asarray(flipped_actions), -np.asarray(original_actions), atol=ABS_TOL))
    body = np.asarray(episode.private_audit["body_map"])
    same_effects = bool(np.allclose(body @ np.asarray(original_actions).T, (-body) @ np.asarray(flipped_actions).T, atol=ABS_TOL))
    return {
        "seed": 4321,
        "transformation": "body B -> -B; all public executed/calibration actions and loss labels negated",
        "same_observation_goal_history_without_calibration_actions": coarsened_equal,
        "opposite_public_teacher_actions": opposite_actions,
        "same_physical_action_effects": same_effects,
        "first_action_l2_difference": action_difference,
        "passed": coarsened_equal and opposite_actions and same_effects and action_difference > 1e-4,
        "interpretation": "Calibration action-observation pairing resolves a real first-decision ambiguity; no active exploration or transfer claim.",
    }


def _fixed_cohort_baselines() -> dict[str, Any]:
    """Measure the reviewed 32-cell cohort with public nonlearning policies."""
    config = CalibratedReachConfig(disturbance_scale=0.02)
    rows: list[dict[str, Any]] = []
    for index, (sensor_dim, action_dim, goal_mode, switch, disturbance) in enumerate(
        product((8, 10), (2, 4), ("absolute", "relative"), (-1, 3), (False, True))
    ):
        world = replace(config, goal_switch_step=None if switch < 0 else switch)
        seed = 100000 + 7919 * index
        for policy in ("fixed_reactive", "teacher", "inaction"):
            rollout = CalibratedReachRollout.from_seed(seed, config=world, sensor_dim=sensor_dim, action_dim=action_dim, goal_mode=goal_mode, disturbance=disturbance)
            rollout.reset()
            while not rollout.done:
                history = rollout.query()
                if policy == "teacher":
                    action = PublicHistoryTeacher(control_gain=world.control_gain).action_for(history, rollout.action_bounds)
                elif policy == "fixed_reactive":
                    action = fixed_reactive_action(history, rollout.action_bounds)
                else:
                    action = np.zeros(action_dim)
                rollout.step(action)
            metrics = rollout.privileged_metrics()
            rows.append({"cell_index": index, "seed": seed, "policy": policy, "sensor_dim": sensor_dim, "action_dim": action_dim, "goal_mode": goal_mode, "goal_switch_step": switch, "disturbance": disturbance, "physical_error": metrics["state_error_l2"], "success": metrics["success"]})
    summary = {}
    for policy in ("fixed_reactive", "teacher", "inaction"):
        selected = [row for row in rows if row["policy"] == policy]
        summary[policy] = {"episodes": len(selected), "successes": sum(row["success"] for row in selected), "mean_physical_error": float(np.mean([row["physical_error"] for row in selected])), "max_physical_error": max(row["physical_error"] for row in selected)}
    return {"world_version": WORLD_VERSION, "training_performed": False, "summary": summary, "rows": rows, "passed": summary["teacher"]["successes"] == 32 and summary["fixed_reactive"]["successes"] < summary["teacher"]["successes"], "interpretation": "Shortcut check, not an optimal calibration-blind policy bound."}


def _surgical_probes(
    episode: TrajectoryEpisode, config: CalibratedReachConfig
) -> dict[str, Any]:
    action_dim = int(episode.private_audit["action_dim"])
    permutation = np.arange(action_dim, dtype=np.int64)[::-1]
    permuted = _permute_episode(episode, permutation)
    original_targets = np.asarray(_teacher_targets(episode, config))
    permuted_targets = np.asarray(_teacher_targets(permuted, config))
    action_ok = bool(np.allclose(permuted_targets, original_targets[:, permutation], atol=ABS_TOL, rtol=ABS_TOL))

    sensor_dim = int(episode.private_audit["sensor_dim"])
    sensor_perm = np.arange(sensor_dim, dtype=np.int64)[::-1]
    sensor_permuted = _sensor_permute_episode(episode, sensor_perm)
    sensor_targets = np.asarray(_teacher_targets(sensor_permuted, config))
    sensor_ok = bool(np.allclose(sensor_targets, original_targets, atol=ABS_TOL, rtol=ABS_TOL))

    # Absolute and reset-relative encodings of the same physical target must
    # induce the same public control targets when the initial observation is
    # available.  This is a semantic frame check, not a learner shortcut test.
    paired = generate_episode(
        int(episode.private_audit["seed"]),
        config=config,
        sensor_dim=sensor_dim,
        action_dim=action_dim,
        goal_mode="absolute",
    )
    relative = generate_episode(
        int(episode.private_audit["seed"]),
        config=config,
        sensor_dim=sensor_dim,
        action_dim=action_dim,
        goal_mode="relative",
    )
    frame_ok = bool(
        np.allclose(
            np.asarray(_teacher_targets(paired, config)),
            np.asarray(_teacher_targets(relative, config)),
            atol=ABS_TOL,
            rtol=ABS_TOL,
        )
    )
    ablations = _information_ablations(episode, config)
    return {
        "action_permutation_invariant": action_ok,
        "sensor_row_permutation_invariant": sensor_ok,
        "absolute_relative_control_equivalent": frame_ok,
        **ablations,
        "action_permutation": permutation.tolist(),
        "sensor_permutation": sensor_perm.tolist(),
    }


def _paired_embodiment_probe(config: CalibratedReachConfig) -> dict[str, Any]:
    """Check that changing embodiment does not silently change the task."""

    seed = 4321
    two = generate_episode(seed, config=config, sensor_dim=8, action_dim=2, goal_mode="absolute")
    four = generate_episode(seed, config=config, sensor_dim=8, action_dim=4, goal_mode="absolute")
    two_private = two.private_audit
    four_private = four.private_audit
    initial_same = bool(np.allclose(two_private["initial_state"], four_private["initial_state"], atol=ABS_TOL, rtol=ABS_TOL))
    target_same = bool(np.allclose(two_private["target_state"], four_private["target_state"], atol=ABS_TOL, rtol=ABS_TOL))
    sensor_same = bool(
        np.allclose(two_private["sensor_matrix"], four_private["sensor_matrix"], atol=ABS_TOL, rtol=ABS_TOL)
        and np.allclose(two_private["sensor_offset"], four_private["sensor_offset"], atol=ABS_TOL, rtol=ABS_TOL)
    )
    two_body = np.asarray(two_private["body_map"], dtype=np.float64)
    four_body = np.asarray(four_private["body_map"], dtype=np.float64)
    body_different = not np.allclose(two_body, four_body[:, :2], atol=ABS_TOL, rtol=ABS_TOL)
    public_start_same = bool(
        np.allclose(
            np.asarray(next(event for event in two.events if event.kind == "observation").values),
            np.asarray(next(event for event in four.events if event.kind == "observation").values),
            atol=ABS_TOL,
            rtol=ABS_TOL,
        )
    )
    public_goal_same = bool(
        np.allclose(
            np.asarray(next(event for event in two.events if event.kind == "goal").values),
            np.asarray(next(event for event in four.events if event.kind == "goal").values),
            atol=ABS_TOL,
            rtol=ABS_TOL,
        )
    )
    return {
        "seed": seed,
        "same_initial_state": initial_same,
        "same_target_state": target_same,
        "same_sensor_transform": sensor_same,
        "different_body_effect": body_different,
        "same_public_start": public_start_same,
        "same_public_goal": public_goal_same,
        "environment_isolation": bool(initial_same and target_same and sensor_same and public_start_same and public_goal_same),
        "action_widths": [2, 4],
        "note": "The action target should differ after public calibration; a memoryless action-only baseline cannot infer this embodiment from an unchanged start/goal pair.",
    }


def _arm_probes(config: CalibratedReachConfig) -> dict[str, Any]:
    """Run one compact paired arm for goal switching and sensed disturbance."""

    armed = replace(config, disturbance_scale=0.02, goal_switch_step=5)
    episode = generate_episode(
        4333,
        config=armed,
        sensor_dim=8,
        action_dim=4,
        goal_mode="relative",
        disturbance=True,
    )
    structure = _public_structure_checks(episode, armed)
    targets = _teacher_targets(episode, armed)
    goals = [event for event in episode.events if event.kind == "goal"]
    switch_present = len(goals) >= 2 and goals[-1].time > goals[0].time
    disturbance_present = bool(episode.private_audit.get("disturbance_enabled"))
    disturbance_step = int(episode.private_audit.get("disturbance_start_step", armed.horizon))
    observations = [event for event in episode.events if event.kind == "observation"]
    changing_observation = any(
        not np.allclose(observations[index].values, observations[index - 1].values, atol=ABS_TOL, rtol=ABS_TOL)
        for index in range(1, len(observations))
    )
    final_goal = goals[-1]
    # The goal switch is published before the query and the resulting
    # observation at the same physical time.  Reconstruct a relative target
    # from public event order, rather than a timestamp comparison that would
    # accidentally read that post-action observation.
    goal_index = next(index for index, event in enumerate(episode.events) if event.event_id == final_goal.event_id)
    final_anchor = [event for event in episode.events[:goal_index] if event.kind == "observation"][-1]
    final_target = np.asarray(final_goal.values, dtype=np.float64)
    if final_goal.key == "relative_goal":
        final_target = np.asarray(final_anchor.values, dtype=np.float64) + final_target
    public_final_error = float(np.linalg.norm(np.asarray(observations[-1].values) - final_target))
    public_final_rmse = public_final_error / np.sqrt(len(final_target))
    private_receipt_matches_public = bool(
        np.isclose(
            float(episode.private_audit.get("teacher_final_sensor_error", np.nan)),
            public_final_error,
            atol=ABS_TOL,
            rtol=ABS_TOL,
        )
    )
    return {
        "seed": 4333,
        "goal_switch_present": switch_present,
        "disturbance_present": disturbance_present,
        "disturbance_start_step": disturbance_step,
        "teacher_targets_checked": len(targets),
        "public_observation_changes": changing_observation,
        "public_final_sensor_error": public_final_error,
        "public_final_sensor_rmse": public_final_rmse,
        "private_receipt_matches_public_target": private_receipt_matches_public,
        "private_fields_in_learner_bundle": structure["private_fields_in_learner_bundle"],
        "passed": bool(
            switch_present
            and disturbance_present
            and 0 <= disturbance_step < armed.horizon
            and changing_observation
            and private_receipt_matches_public
        ),
        "episode": episode,
    }


def _model_integration_probe() -> dict[str, Any]:
    """Run the smallest CPU Trainer save/resume smoke once the wrapper exists."""

    from pretraining_experiments.first_training import FirstTrainingConfig, build_first_system, run_cpu_smoke

    config = FirstTrainingConfig(
        dataset_size=2,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        attention_heads=2,
        smoke_steps=1,
        resumed_steps=2,
    )
    with tempfile.TemporaryDirectory(prefix="first-system-audit-") as directory:
        receipt = run_cpu_smoke(directory, config)
        state_keys = tuple(build_first_system(config).state_dict())
    required_groups = {
        "core": any(key.startswith("core.") for key in state_keys),
        "adapters": any(key.startswith("adapters.") for key in state_keys),
        "decoder": any(key.startswith("decoder.") for key in state_keys),
    }
    if not all(required_groups.values()):
        raise ValueError(f"model state is missing registered groups: {required_groups}")
    closed_loop = _closed_loop_model_probe(build_first_system(config))
    return {
        "status": "pass",
        "apparatus_only": bool(receipt.get("apparatus_only")),
        "first_steps": int(receipt["first_steps"]),
        "resumed_steps": int(receipt["resumed_steps"]),
        "trainable_parameters": int(receipt["trainable_parameters"]),
        "registered_state_groups": required_groups,
        "closed_loop": closed_loop,
    }


def _closed_loop_model_probe(model: Any) -> dict[str, Any]:
    """Drive a real public rollout from model actions, without labels or audit state.

    This is an apparatus check only.  The model is freshly initialized; the
    result proves that the common-content bundle can consume each growing
    public prefix, decode a bounded body-local action, and advance the actual
    environment.  It does not claim that the untrained policy regulates.
    """

    import torch

    from pretraining_experiments.trajectory_data import (
        MAX_RAW_WIDTH,
        ROLE_IDS,
        _event_route,
        _key_code,
        first_system_adapter_specs,
    )
    from pretraining_experiments.trajectory_world import CalibratedReachRollout

    adapter_codes = {spec.name: index for index, spec in enumerate(first_system_adapter_specs())}
    rollout = CalibratedReachRollout.from_seed(
        4344,
        sensor_dim=10,
        action_dim=4,
        goal_mode="relative",
        disturbance=False,
    )
    rollout.reset()
    model.eval()
    actions: list[list[float]] = []

    def public_batch(events: list[PublicEvent]) -> dict[str, torch.Tensor]:
        count = len(events)
        raw_values = torch.zeros(1, count, MAX_RAW_WIDTH, dtype=torch.float32)
        adapter_ids = torch.zeros(1, count, dtype=torch.long)
        role_ids = torch.zeros(1, count, dtype=torch.long)
        key_ids = torch.zeros(1, count, dtype=torch.long)
        time_values = torch.zeros(1, count, dtype=torch.float32)
        available_at = torch.zeros(1, count, dtype=torch.float32)
        position_ids = torch.arange(count, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones(1, count, dtype=torch.bool)
        action_mask = torch.ones(1, count, dtype=torch.bool)
        for index, event in enumerate(events):
            route, values = _event_route(event)
            raw_values[0, index, : len(values)] = torch.tensor(values, dtype=torch.float32)
            adapter_ids[0, index] = adapter_codes[route]
            role_ids[0, index] = ROLE_IDS[event.kind]
            key_ids[0, index] = _key_code(event.key)
            time_values[0, index] = float(event.time)
            available_at[0, index] = float(
                event.available_at if event.available_at is not None else event.time
            )
        return {
            "raw_values": raw_values,
            "adapter_ids": adapter_ids,
            "role_ids": role_ids,
            "key_ids": key_ids,
            "time_values": time_values,
            "available_at": available_at,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "action_mask": action_mask,
        }

    with torch.inference_mode():
        while not rollout.done:
            rollout.query()
            output = model(**public_batch(rollout.events))
            action = output.actions[0, -1, : rollout.environment.action_dim].detach().cpu().numpy()
            if action.shape != (rollout.environment.action_dim,) or not np.isfinite(action).all():
                raise ValueError("closed-loop model action is not finite or has the wrong width")
            if np.any(action < -0.5 - ABS_TOL) or np.any(action > 0.5 + ABS_TOL):
                raise ValueError("closed-loop model action exceeded the public body bounds")
            actions.append(action.tolist())
            rollout.step(action)
    episode = rollout.to_episode()
    if "private_audit" in episode.learner_bundle():
        raise ValueError("closed-loop learner bundle exposed private audit data")
    final_goal = next(event for event in reversed(episode.events) if event.kind == "goal")
    goal_index = episode.events.index(final_goal)
    anchor = next(
        event
        for event in reversed(episode.events[:goal_index])
        if event.kind == "observation"
    )
    target = np.asarray(final_goal.values, dtype=np.float64)
    if final_goal.key == "relative_goal":
        target = np.asarray(anchor.values, dtype=np.float64) + target
    final_observation = next(event for event in reversed(episode.events) if event.kind == "observation")
    return {
        "status": "pass",
        "apparatus_only": True,
        "seed": 4344,
        "sensor_dim": 10,
        "action_dim": 4,
        "steps": len(actions),
        "bounded_actions": True,
        "public_prefix_only": True,
        "final_sensor_error": float(
            np.linalg.norm(
                np.asarray(final_observation.values, dtype=np.float64) - target
            )
        ),
        "final_sensor_rmse": float(
            np.linalg.norm(np.asarray(final_observation.values, dtype=np.float64) - target)
            / np.sqrt(len(target))
        ),
        "note": "freshly initialized bundle; integration evidence only, not learner or regulation evidence",
    }


def _trace(episode: TrajectoryEpisode, numerical: dict[str, Any]) -> str:
    """Render reviewer metadata, public events, and supervision separately.

    The first version of this trace omitted calibration observations and put
    teacher targets beside the public query event.  That made a useful audit
    view look like the target was part of the learner-visible event.  Keep all
    three evidence layers explicit in the generated trace.
    """
    targets = {target.query_event_id: target for target in episode.supervision}
    lines = [
        "[reviewer metadata; excluded from public learner events]",
        f"world={episode.private_audit.get('sensor_dim')} sensor dims, {episode.private_audit.get('action_dim')} actions",
        f"seed={episode.private_audit.get('seed')} goal_mode={episode.private_audit.get('goal_mode')}",
        f"calibration_rank={numerical.get('calibration_rank')} nullity={numerical.get('action_nullity')}",
        f"sensor_error={numerical.get('initial_sensor_error')} -> {numerical.get('final_sensor_error')}",
        "calibration_scaffold=public signed pulse/observation pairs; t=0.00 is the acquisition boundary, not elapsed exploration",
        "",
        "[public event content]",
    ]
    for event in episode.events:
        if event.kind in {
            "reset",
            "goal",
            "calibration_action",
            "calibration_observation",
            "action_query",
            "action_executed",
            "observation",
            "episode_end",
        }:
            detail = ""
            if event.action:
                detail = " action=" + np.array2string(np.asarray(event.action), precision=4)
            elif event.values:
                detail = " values=" + np.array2string(np.asarray(event.values), precision=4)
            if event.kind == "goal":
                detail += f" frame={event.frame} key={event.key}"
            lines.append(f"{event.event_id:03d} t={event.time:5.2f} {event.kind}{detail}")
    lines.extend(("", "[reviewer supervision labels; excluded from public event content]"))
    for query_event_id in sorted(targets):
        target = targets[query_event_id]
        lines.append(
            f"query_event={query_event_id:03d} target="
            + np.array2string(np.asarray(target.action), precision=4)
            + f" valid={list(target.action_valid)}"
        )
    return "\n".join(lines) + "\n"


def _public_trace(episode: TrajectoryEpisode) -> str:
    """Write a compact public-only trace for a special readiness arm."""

    lines = [
        f"world={episode.private_audit.get('sensor_dim')} sensor dims, {episode.private_audit.get('action_dim')} actions",
        "trace=public events only; private receipt is excluded",
    ]
    for event in episode.events:
        detail = ""
        if event.action:
            detail = " action=" + np.array2string(np.asarray(event.action), precision=4)
        elif event.values:
            detail = " values=" + np.array2string(np.asarray(event.values), precision=4)
        if event.kind == "goal":
            detail += f" frame={event.frame} key={event.key}"
        lines.append(f"{event.event_id:03d} t={event.time:5.2f} {event.kind}{detail}")
    return "\n".join(lines) + "\n"


def _case(seed: int, sensor_dim: int, action_dim: int, goal_mode: str, config: CalibratedReachConfig) -> dict[str, Any]:
    episode = generate_episode(
        seed,
        config=config,
        sensor_dim=sensor_dim,
        action_dim=action_dim,
        goal_mode=goal_mode,
        disturbance=False,
    )
    structure = _public_structure_checks(episode, config)
    _teacher_targets(episode, config)
    numerical = _numerical_checks(episode, config)
    probes = _surgical_probes(episode, config)
    if not all(value if isinstance(value, bool) else bool(value) for value in probes.values()):
        raise ValueError(f"semantic permutation probe failed: {probes}")
    return {
        "seed": seed,
        "sensor_dim": sensor_dim,
        "action_dim": action_dim,
        "goal_mode": goal_mode,
        "structure": structure,
        "numerical": numerical,
        "probes": probes,
        "fingerprint": corpus_fingerprint(episode),
        "episode": episode,
    }


def run_audit(*, include_training: bool = True) -> dict[str, Any]:
    config = CalibratedReachConfig()
    cases: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seeds = iter(range(4200, 4208))
    for action_dim in config.action_dims:
        for sensor_dim in config.sensor_dims:
            for goal_mode in config.goal_modes:
                seed = next(seeds)
                try:
                    cases.append(_case(seed, sensor_dim, action_dim, goal_mode, config))
                except Exception as exc:  # receipt must survive a producer defect
                    failures.append({
                        "kind": "factorial_case",
                        "seed": seed,
                        "sensor_dim": sensor_dim,
                        "action_dim": action_dim,
                        "goal_mode": goal_mode,
                        "error": f"{type(exc).__name__}: {exc}",
                    })

    feature_gaps: list[str] = []
    try:
        arm_receipt = _arm_probes(config)
        if not arm_receipt["passed"]:
            feature_gaps.append("goal_switch_or_disturbance_arm_failed")
    except Exception as exc:
        arm_receipt = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        feature_gaps.append("goal_switch_or_disturbance_arm_failed")
    try:
        embodiment_receipt = _paired_embodiment_probe(config)
        if not embodiment_receipt["environment_isolation"]:
            feature_gaps.append("same_environment_embodiment_pair_failed")
    except Exception as exc:
        embodiment_receipt = {"environment_isolation": False, "error": f"{type(exc).__name__}: {exc}"}
        feature_gaps.append("same_environment_embodiment_pair_failed")

    try:
        information_witness = _signed_body_information_witness(config)
        baselines = _fixed_cohort_baselines()
        if not information_witness["passed"] or not baselines["passed"]:
            feature_gaps.append("calibration_information_or_shortcut_check_failed")
    except Exception as exc:
        information_witness = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        baselines = {"passed": False}
        feature_gaps.append("calibration_information_or_shortcut_check_failed")
    try:
        if include_training:
            model_integration = _model_integration_probe()
        else:
            from pretraining_experiments.first_training import FirstTrainingConfig, build_first_system
            import torch
            torch.set_num_threads(1)
            model_integration = {"status": "nonlearning_forward_pass", "training_performed": False, "checkpoint_continuation": "not_repeated_in_world_only_audit", "closed_loop": _closed_loop_model_probe(build_first_system(FirstTrainingConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, attention_heads=2)))}
    except Exception as exc:
        model_integration = {
            "status": "blocked",
            "error": f"{type(exc).__name__}: {exc}",
        }
        failures.append({"kind": "model_integration", "error": model_integration["error"]})

    receipt: dict[str, Any] = {
        "schema_version": 2,
        "status": "pass" if not failures and not feature_gaps else "blocked",
        "world_version": WORLD_VERSION,
        "teacher_version": TEACHER_VERSION,
        "training_performed": include_training,
        "cpu_only": True,
        "factorial_cases_requested": 8,
        "factorial_cases_passed": len(cases),
        "failures": failures,
        "feature_gaps": feature_gaps,
        "arm_probe": {key: value for key, value in arm_receipt.items() if key != "episode"},
        "paired_embodiment_probe": embodiment_receipt,
        "calibration_information_witness": information_witness,
        "fixed_cohort_baselines": baselines,
        "teacher_ablation_interpretation": "Missing-field rejection proves a teacher API precondition only; necessity evidence is separate.",
        "cases": [
            {key: value for key, value in case.items() if key != "episode"}
            for case in cases
        ],
        "model_integration": model_integration,
        "scientific_split": {
            "status": "pending_scientific_runner",
            "audit_seed_block": [4200, 4207],
            "smoke_training_size": 2,
            "held_out_evaluation": "not_run_by_boundary_audit",
            "note": "This receipt proves world and apparatus seams; it is not a source-versus-held-out learner result.",
        },
        "learner_contract": {
            "availability": "synchronous_only",
            "success_comparison": "report_raw_sensor_l2_and_l2_over_sqrt_sensor_width",
            "world_config_propagation": "pending_scientific_toml_runner",
        },
    }
    if cases:
        receipt["trace_case"] = {
            "seed": cases[0]["seed"],
            "sensor_dim": cases[0]["sensor_dim"],
            "action_dim": cases[0]["action_dim"],
            "goal_mode": cases[0]["goal_mode"],
        }
        receipt["trace_text"] = _trace(cases[0]["episode"], cases[0]["numerical"])
    if isinstance(arm_receipt, dict) and "episode" in arm_receipt:
        receipt["arm_trace_text"] = _public_trace(arm_receipt["episode"])
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-only", action="store_true", help="Run world checks and a nonlearning model forward; do not optimize weights.")
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        receipt = run_audit(include_training=not args.world_only)
    except Exception as exc:  # preserve a compact blocker even on import/runtime failure
        receipt = {
            "schema_version": 2,
            "status": "blocked",
            "cpu_only": True,
            "factorial_cases_requested": 8,
            "factorial_cases_passed": 0,
            "failures": [{"kind": "audit_runtime", "error": f"{type(exc).__name__}: {exc}"}],
            "feature_gaps": [],
            "model_integration": {"status": "pending"},
        }
    trace = str(receipt.pop("trace_text", "No episode was generated; inspect failures in audit.json.\n"))
    arm_trace = str(receipt.pop("arm_trace_text", "No arm episode was generated; inspect failures in audit.json.\n"))
    receipt_name = "audit-world-only.json" if args.world_only else "audit.json"
    (OUTPUT_ROOT / receipt_name).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    suffix = "-world-only" if args.world_only else ""
    (OUTPUT_ROOT / f"episode-trace{suffix}.txt").write_text(trace, encoding="utf-8")
    (OUTPUT_ROOT / f"arm-probe-trace{suffix}.txt").write_text(arm_trace, encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
