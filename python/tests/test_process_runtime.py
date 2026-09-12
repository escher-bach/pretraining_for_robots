from __future__ import annotations

import json
import subprocess
import unittest

import numpy as np

from pretraining_experiments.trajectory_world import (
    ProcessIR,
    ProcessWorldFactory,
    TimedPublicProcessTeacher,
    compose_processes,
    generate_composed_episode,
    process_public_only,
)


class ProcessRuntimeTests(unittest.TestCase):
    def _compiled_programs(self):
        raw = subprocess.check_output(
            ["cargo", "run", "-q", "-p", "pretraining-term-compiler", "--bin", "process-export", "--", "--seed", "7", "--count", "3"],
            text=True,
        )
        return json.loads(raw)

    def test_rust_export_round_trips_into_one_python_runtime(self) -> None:
        programs = self._compiled_programs()
        self.assertEqual(len(programs), 3)
        for generated in programs:
            ir = ProcessIR.from_compiled_process(generated)
            self.assertEqual(ir.sensor_dim, generated["process"]["sensor_dim"])
            self.assertEqual(ir.action_dim, generated["process"]["action_dim"])
            episode = generate_composed_episode(
                31,
                ir=ir,
                sensor_dim=ir.sensor_dim,
                action_dim=ir.action_dim,
            )
            self.assertEqual(len(episode.supervision), ir.horizon)
            self.assertTrue(episode.private_audit["teacher_physical_metrics"]["success"])

    def test_compiled_lowering_values_are_faithful_and_inconsistent_runtime_is_rejected(self) -> None:
        program = self._compiled_programs()[0]["process"]
        program["nodes"] = [
            dict(node, kind=(dict(node["kind"], Plant=dict(node["kind"]["Plant"], gain=1.75)) if "Plant" in node["kind"] else node["kind"]))
            for node in program["nodes"]
        ]
        program["runtime_private"]["dynamics_gain"] = 1.75
        ir = ProcessIR.from_compiled_process(program)
        self.assertEqual(ir.plant_gain, 1.75)
        environment = __import__("pretraining_experiments.process_runtime", fromlist=["ProcessEnvironment"]).ProcessEnvironment.from_seed(
            4, ir=ir, sensor_dim=ir.sensor_dim, action_dim=ir.action_dim
        )
        self.assertEqual(environment.dynamics_gain, 1.75)
        program["runtime_private"]["dynamics_gain"] = 1.0
        with self.assertRaises(ValueError):
            ProcessIR.from_compiled_process(program)
    def test_composition_is_validated_and_hashes_canonically(self) -> None:
        left = compose_processes("reach", "actuator_lag", "goal_switch", goal_switch_step=3)
        right = ProcessIR(components=("reach", "actuator_lag", "goal_switch"), goal_switch_step=3)
        self.assertEqual(left.contract_hash, right.contract_hash)
        with self.assertRaises(ValueError):
            compose_processes("reach", "unknown")
        with self.assertRaises(ValueError):
            compose_processes("reach", "goal_switch")

    def test_timed_public_calibration_identifies_lag_and_controls_composition(self) -> None:
        ir = compose_processes(
            "reach", "actuator_lag", "goal_switch", "disturbance",
            horizon=10, goal_switch_step=4, disturbance_scale=0.08,
        )
        episode = generate_composed_episode(17, ir=ir, sensor_dim=10, action_dim=4, goal_mode="relative")
        self.assertTrue(episode.private_audit["teacher_physical_metrics"]["success"])
        calibration = [event for event in episode.events if event.kind == "calibration_action"]
        self.assertEqual(len(calibration), 8)
        self.assertGreater(len({event.time for event in calibration}), 1)
        self.assertEqual(len([event for event in episode.events if event.kind == "goal"]), 2)
        self.assertEqual(len(episode.supervision), ir.horizon)

    def test_public_bundle_excludes_private_process_parameters(self) -> None:
        episode = generate_composed_episode(19, ir=compose_processes("reach", "actuator_lag"), sensor_dim=8, action_dim=2)
        encoded = json.dumps(episode.learner_bundle(), sort_keys=True)
        for forbidden in ("initial_state", "sensor_matrix", "body_map", "disturbance_start_step", "seed", "process_ir"):
            self.assertNotIn(forbidden, encoded)
        self.assertTrue(process_public_only())
        self.assertTrue(np.isfinite(np.asarray(episode.supervision[0].action)).all())

    def test_teacher_api_does_not_accept_environment_or_latent_state(self) -> None:
        self.assertEqual(tuple(__import__("inspect").signature(TimedPublicProcessTeacher.action_for).parameters), ("self", "history", "action_bounds"))

    def test_one_factory_generates_crossed_embodiments(self) -> None:
        factory = ProcessWorldFactory(base=compose_processes("reach", "actuator_lag"))
        episodes = factory.generate(101, 8)
        self.assertEqual(len(episodes), 8)
        self.assertGreaterEqual(len({episode.private_audit["sensor_dim"] for episode in episodes}), 2)
        self.assertGreaterEqual(len({episode.private_audit["action_dim"] for episode in episodes}), 2)
        self.assertTrue(all(episode.private_audit["teacher_physical_metrics"]["success"] for episode in episodes))


if __name__ == "__main__":
    unittest.main()
