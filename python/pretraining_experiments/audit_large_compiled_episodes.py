"""Reproduce selected large-run compiler episodes for final review.

The public artifact contains only the learner bundle.  Process IR, physical
metrics, and audit predicates are written to a separate reviewer file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .process_runtime import ProcessIR, TimedPublicProcessTeacher, generate_composed_episode, load_compiled_processes
from .trajectory_world import PublicHistory


FAMILY_NAMES = {
    "compiled_reaching": "Reaching",
    "compiled_actuator_lag": "ActuatorLag",
    "compiled_goal_switch": "GoalSwitch",
    "compiled_disturbance": "Disturbance",
    "compiled_actuator_lag_goal_switch": "ActuatorLagGoalSwitch",
    "compiled_actuator_lag_disturbance": "ActuatorLagDisturbance",
    "compiled_goal_switch_disturbance": "GoalSwitchDisturbance",
    "compiled_lag_goal_switch_disturbance": "LagGoalSwitchDisturbance",
}


def _matching_process(programs: tuple[dict[str, object], ...], family: str, sensor_dim: int, action_dim: int) -> dict[str, object]:
    wanted = FAMILY_NAMES[family]
    matches = [
        item["process"]
        for item in programs
        if str(item["process"].get("family", "")).split("::")[-1] == wanted
        and int(item["process"].get("sensor_dim", -1)) == sensor_dim
        and int(item["process"].get("action_dim", -1)) == action_dim
    ]
    if not matches:
        raise ValueError(f"compiled manifest has no {family}/{sensor_dim}/{action_dim} process")
    return matches[0]


def _episode_checks(episode: Any, ir: ProcessIR, *, goal_mode: str, seed: int) -> dict[str, object]:
    events = episode.events
    public = episode.public_input()
    encoded_public = json.dumps(public, sort_keys=True, separators=(",", ":"))
    calibration_actions = [event for event in events if event.kind == "calibration_action"]
    calibration_observations = [event for event in events if event.kind == "calibration_observation"]
    goals = [event for event in events if event.kind == "goal"]
    queries = [event for event in events if event.kind == "action_query"]
    actions = [event for event in events if event.kind == "action_executed"]
    forbidden = ("initial_state", "sensor_matrix", "body_map", "disturbance_start_step", "process_ir", "seed")
    times = [event.time for event in events]
    switch_time = None
    if ir.goal_switch_step is not None:
        # Calibration consumes 2 * action_dim physical intervals before step 0.
        switch_time = (2 * ir.action_dim + ir.goal_switch_step) * ir.dt
    first_query_index = next(index for index, event in enumerate(events) if event.kind == "action_query")
    first_query_history = PublicHistory.from_events(events[: first_query_index + 1])
    fit_teacher = TimedPublicProcessTeacher()
    _, _, identified_alpha = fit_teacher._fit(first_query_history, ir.action_dim, np.full((ir.action_dim, 2), (-0.5, 0.5)))
    executed_values = np.asarray([event.action for event in actions], dtype=np.float64)
    saturation_count = int(np.sum(np.isclose(np.abs(executed_values), 0.5, atol=1.0e-10))) if len(executed_values) else 0
    final_error = float(episode.private_audit["teacher_physical_metrics"]["state_error_l2"])
    checks: dict[str, object] = {
        "seed": seed,
        "goal_mode": goal_mode,
        "public_sha256": hashlib.sha256(encoded_public.encode("utf-8")).hexdigest(),
        "public_forbidden_fields_absent": all(field not in encoded_public for field in forbidden),
        "public_event_count": len(public),
        "supervision_count": len(episode.supervision),
        "horizon_matches_supervision": len(episode.supervision) == ir.horizon,
        "event_ids_contiguous": [event.event_id for event in events] == list(range(len(events))),
        "event_times_monotone": all(left <= right for left, right in zip(times, times[1:])),
        "calibration_action_count": len(calibration_actions),
        "calibration_observation_count": len(calibration_observations),
        "calibration_times_strict": all(left < right for left, right in zip([event.time for event in calibration_observations], [event.time for event in calibration_observations][1:])),
        "calibration_before_first_query": bool(calibration_observations) and calibration_observations[-1].time <= queries[0].time,
        "goal_count": len(goals),
        "goal_switch_time_expected": switch_time,
        "goal_switch_time_observed": (goals[-1].time if len(goals) > 1 else None),
        "goal_switch_causal": (len(goals) == 1 if ir.goal_switch_step is None else len(goals) == 2 and bool(np.isclose(goals[-1].time, switch_time))),
        "query_action_alignment": len(queries) == len(actions) == len(episode.supervision),
        "action_bounds_respected": all(all(-0.5 <= value <= 0.5 for value in event.action) for event in actions),
        "public_query_has_no_label": all("action" not in event for event in public if event["kind"] == "action_query"),
        "all_public_values_finite": all(np.isfinite(value) for event in events for value in (*event.values, *event.action)),
        "goal_reachable_within_horizon_at_0.05": float(episode.private_audit["teacher_physical_metrics"]["state_error_l2"]) <= 0.05,
        "goal_margin_to_0.05": 0.05 - final_error,
        "identified_lag_response_alpha": float(identified_alpha),
        "compiled_lag_response_alpha": (float(ir.actuator_lag) if ir.uses_lag else 1.0),
        "action_saturation_count": saturation_count,
        "action_saturation_fraction": float(saturation_count / max(executed_values.size, 1)),
        "physical_metrics": episode.private_audit["teacher_physical_metrics"],
        "interface_fields": ["observation", "goal", "action_query", "action_executed", "embodiment dimensions", "timed public history"],
        "transition_table_absent_from_public_input": all("transition_table" not in event for event in public),
    }
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled-processes", type=Path, required=True)
    parser.add_argument("--scientific-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    programs = load_compiled_processes(args.compiled_processes)
    receipt = json.loads(args.scientific_receipt.read_text(encoding="utf-8"))
    final_rows = receipt["closed_loop"][-1]["rows"]
    failures = [row for row in final_rows if row["policy"] == "teacher" and not row["success"]]
    if len(failures) != 2:
        raise ValueError(f"expected two final teacher failures, found {len(failures)}")
    requests = [
        {"label": "teacher_failure_1", "row": failures[0]},
        {"label": "teacher_failure_2", "row": failures[1]},
        {
            "label": "representative_lag_switch_disturbance",
            "row": {
                "process_family": "compiled_lag_goal_switch_disturbance",
                "sensor_dim": 8,
                "body_dim": 2,
                "goal_mode": "relative",
                "seed": 20260913,
            },
        },
    ]
    public_cases: list[dict[str, object]] = []
    reviewer_cases: list[dict[str, object]] = []
    for request in requests:
        row = request["row"]
        family = str(row["process_family"])
        sensor_dim = int(row["sensor_dim"])
        action_dim = int(row["body_dim"])
        goal_mode = str(row["goal_mode"])
        seed = int(row["seed"])
        process = _matching_process(programs, family, sensor_dim, action_dim)
        ir = ProcessIR.from_compiled_process(process)
        episode = generate_composed_episode(seed, ir=ir, sensor_dim=sensor_dim, action_dim=action_dim, goal_mode=goal_mode)
        public_cases.append({
            "case_id": request["label"],
            "world_schema": "trajectory-public-events-v2",
            "events": list(episode.public_input()),
            "supervision": [target.private_dict() for target in episode.supervision],
        })
        reviewer_cases.append({
            "case_id": request["label"],
            "source_row": row,
            "compiled_process_name": ir.name,
            "compiled_process_contract": ir.canonical(),
            "compiled_contract_hash": ir.contract_hash,
            "checks": _episode_checks(episode, ir, goal_mode=goal_mode, seed=seed),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "generated-episodes-public.json").write_text(
        json.dumps({"schema_version": "generated-public-episodes-v1", "cases": public_cases}, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "generated-episodes-reviewer.json").write_text(
        json.dumps({
            "schema_version": "generated-episode-review-v1",
            "source_compiled_processes": str(args.compiled_processes),
            "source_scientific_receipt": str(args.scientific_receipt),
            "teacher_version": "timed-public-process-teacher-v1",
            "public_interface_note": "The learner-facing path contains timed observations, goals, executed actions, action queries, and declared embodiment widths. It does not contain a transition table, latent state, private process graph, generator seed, vision encoder, language encoder, or full robotic corpus adapter.",
            "cases": reviewer_cases,
        }, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"cases": len(public_cases), "output_dir": str(args.output_dir), "teacher_failures_reproduced": len(failures)}, sort_keys=True))


if __name__ == "__main__":
    main()
