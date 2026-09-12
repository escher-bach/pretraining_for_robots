"""Emit a compact semantic receipt for one composed compiler/runtime world."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from .trajectory_world import ProcessIR, compose_processes, generate_composed_episode


def _load_compiled(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.compiled_json:
        payload = json.loads(Path(args.compiled_json).read_text(encoding="utf-8"))
    else:
        repo = Path(__file__).resolve().parents[2]
        payload = json.loads(
            subprocess.check_output(
                [
                    "cargo",
                    "run",
                    "-q",
                    "-p",
                    "pretraining-term-compiler",
                    "--bin",
                    "process-export",
                    "--",
                    "--seed",
                    str(args.export_seed),
                    "--count",
                    str(args.export_count),
                ],
                cwd=repo,
                text=True,
            )
        )
    if not isinstance(payload, list) or len(payload) < 3:
        raise ValueError("compiled audit requires at least three exported process programs")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--sensor-dim", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--compiled-json", type=Path)
    parser.add_argument("--export-seed", type=int, default=20260913)
    parser.add_argument("--export-count", type=int, default=3)
    args = parser.parse_args()
    programs = _load_compiled(args)
    audits = []
    for index, generated in enumerate(programs):
        ir = ProcessIR.from_compiled_process(generated)
        episode = generate_composed_episode(
            args.seed + index,
            ir=ir,
            sensor_dim=ir.sensor_dim or args.sensor_dim,
            action_dim=ir.action_dim or args.action_dim,
            goal_mode="relative",
        )
        public = episode.public_input()
        audits.append({
            "name": ir.name,
            "contract_hash": ir.contract_hash,
            "process": ir.canonical(),
            "event_count": len(episode.events),
            "supervision_count": len(episode.supervision),
            "calibration_times": sorted({event.time for event in episode.events if event.kind == "calibration_action"}),
            "goal_count": len([event for event in episode.events if event.kind == "goal"]),
            "teacher_metrics": episode.private_audit["teacher_physical_metrics"],
            "public_forbidden_fields": {
                field: field in json.dumps(public)
                for field in ("initial_state", "sensor_matrix", "body_map", "seed", "disturbance_start_step", "process_ir")
            },
        })
    receipt = {
        "schema_version": "composed-process-audit-v2",
        "source": "rust-process-export",
        "export_count": len(programs),
        "programs": audits,
    }
    print(json.dumps(receipt, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
