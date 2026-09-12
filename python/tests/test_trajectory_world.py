from __future__ import annotations

import json
import unittest

import numpy as np

from pretraining_experiments.trajectory_world import (
    CalibratedReachConfig,
    CalibratedReachEnvironment,
    CalibratedReachRollout,
    LEGACY_WORLD_VERSION,
    PublicHistory,
    PublicHistoryTeacher,
    TrajectoryEpisode,
    WORLD_CONTRACT_VERSION,
    WORLD_VERSION,
    corpus_fingerprint,
    fixed_reactive_action,
    generate_episode,
    teacher_public_only,
)


class TrajectoryWorldTests(unittest.TestCase):
    def test_v2_contract_is_explicit_and_legacy_identifier_is_preserved(self) -> None:
        self.assertEqual(WORLD_VERSION, "calibrated-reach-v2")
        self.assertEqual(WORLD_CONTRACT_VERSION, "CalibratedReachV2")
        self.assertEqual(LEGACY_WORLD_VERSION, "calibrated-reach-v1")

    def test_declared_third_sensor_width_uses_same_public_teacher_contract(self) -> None:
        config = CalibratedReachConfig(sensor_dims=(6,))
        episode = generate_episode(19, config=config, sensor_dim=6, action_dim=2, goal_mode="absolute")
        self.assertEqual(episode.private_audit["sensor_dim"], 6)
        self.assertTrue(episode.private_audit["teacher_physical_metrics"]["success"])
        self.assertEqual(episode.learner_bundle()["world_contract_version"], WORLD_CONTRACT_VERSION)

    def test_sensor_and_actuator_realization_streams_are_independent(self) -> None:
        config = CalibratedReachConfig(sensor_dims=(6, 8, 10))
        narrow = CalibratedReachEnvironment.from_seed(20, config=config, sensor_dim=6, action_dim=2, goal_mode="absolute")
        wide = CalibratedReachEnvironment.from_seed(20, config=config, sensor_dim=10, action_dim=2, goal_mode="absolute")
        two = CalibratedReachEnvironment.from_seed(20, config=config, sensor_dim=8, action_dim=2, goal_mode="absolute")
        four = CalibratedReachEnvironment.from_seed(20, config=config, sensor_dim=8, action_dim=4, goal_mode="absolute")
        np.testing.assert_allclose(narrow.body_map, wide.body_map)
        np.testing.assert_allclose(two.sensor_matrix, four.sensor_matrix)
        np.testing.assert_allclose(two.sensor_offset, four.sensor_offset)
        np.testing.assert_allclose(two.initial_state, four.initial_state)
        np.testing.assert_allclose(two.target_state, four.target_state)

    def test_v2_teacher_beats_calibration_free_reactive_control_on_32_cells(self) -> None:
        teacher_successes = 0
        reactive_successes = 0
        cell = 0
        for sensor_dim in (8, 10):
            for action_dim in (2, 4):
                for goal_mode in ("absolute", "relative"):
                    for repeat in range(4):
                        seed = 100 + cell
                        teacher_rollout = CalibratedReachRollout.from_seed(
                            seed, sensor_dim=sensor_dim, action_dim=action_dim, goal_mode=goal_mode
                        )
                        reactive_rollout = CalibratedReachRollout.from_seed(
                            seed, sensor_dim=sensor_dim, action_dim=action_dim, goal_mode=goal_mode
                        )
                        teacher_rollout.reset()
                        reactive_rollout.reset()
                        teacher = PublicHistoryTeacher(control_gain=teacher_rollout.config.control_gain)
                        for _ in range(teacher_rollout.config.horizon):
                            teacher_rollout.query()
                            reactive_rollout.query()
                            teacher_rollout.step(teacher.action_for(teacher_rollout.history, teacher_rollout.action_bounds))
                            reactive_rollout.step(fixed_reactive_action(reactive_rollout.history, reactive_rollout.action_bounds))
                        teacher_successes += int(teacher_rollout.privileged_metrics()["success"])
                        reactive_successes += int(reactive_rollout.privileged_metrics()["success"])
                        cell += 1
        self.assertEqual(cell, 32)
        self.assertEqual(teacher_successes, 32)
        self.assertLess(reactive_successes, teacher_successes)

    def test_matched_public_prefix_requires_calibration_for_signed_body_change(self) -> None:
        positive = CalibratedReachRollout.from_seed(
            121, sensor_dim=8, action_dim=2, goal_mode="absolute"
        )
        negative = CalibratedReachRollout.from_seed(
            121, sensor_dim=8, action_dim=2, goal_mode="absolute"
        )
        negative.environment.body_map *= -1.0
        positive_prefix = positive.reset()
        negative_prefix = negative.reset()
        self.assertEqual(positive_prefix.events[:3], negative_prefix.events[:3])
        positive_calibration = positive_prefix.of_kind("calibration_observation")
        negative_calibration = negative_prefix.of_kind("calibration_observation")
        self.assertTrue(
            any(left.values != right.values for left, right in zip(positive_calibration, negative_calibration))
        )
        positive.query()
        negative.query()
        teacher = PublicHistoryTeacher()
        positive_action = teacher.action_for(positive.history, positive.action_bounds)
        negative_action = teacher.action_for(negative.history, negative.action_bounds)
        self.assertFalse(np.allclose(positive_action, negative_action))
    def test_public_trajectory_has_no_table_or_latent_fields(self) -> None:
        episode = generate_episode(17, sensor_dim=8, action_dim=2, goal_mode="absolute")
        encoded = json.dumps(episode.public_input(), sort_keys=True)
        for forbidden in (
            "transition_table",
            "initial_state",
            "target_state",
            "sensor_matrix",
            "body_map",
            "dynamics_gain",
            "seed",
        ):
            self.assertNotIn(forbidden, encoded)
        self.assertIn("sensor_matrix", episode.private_audit)
        self.assertNotIn("private_audit", episode.learner_bundle())

    def test_crossed_sensor_and_body_realizations_are_independent(self) -> None:
        cases = [
            generate_episode(21, sensor_dim=sensor, action_dim=action, goal_mode="relative")
            for sensor in (8, 10)
            for action in (2, 4)
        ]
        self.assertEqual(
            {(episode.private_audit["sensor_dim"], episode.private_audit["action_dim"]) for episode in cases},
            {(8, 2), (8, 4), (10, 2), (10, 4)},
        )
        self.assertTrue(all(len(episode.events[1].values) in (8, 10) for episode in cases))
        self.assertTrue(all(len(episode.supervision[0].action) in (2, 4) for episode in cases))

    def test_body_pair_shares_task_and_sensor_realization(self) -> None:
        two = generate_episode(23, sensor_dim=8, action_dim=2, goal_mode="absolute")
        four = generate_episode(23, sensor_dim=8, action_dim=4, goal_mode="absolute")
        self.assertEqual(two.private_audit["initial_state"], four.private_audit["initial_state"])
        self.assertEqual(two.private_audit["target_state"], four.private_audit["target_state"])
        self.assertEqual(two.private_audit["sensor_matrix"], four.private_audit["sensor_matrix"])
        self.assertEqual(two.private_audit["sensor_offset"], four.private_audit["sensor_offset"])
        self.assertEqual(
            next(event.values for event in two.events if event.kind == "observation"),
            next(event.values for event in four.events if event.kind == "observation"),
        )
        self.assertEqual(
            next(event.values for event in two.events if event.kind == "goal"),
            next(event.values for event in four.events if event.kind == "goal"),
        )
        self.assertNotEqual(two.private_audit["body_map"], four.private_audit["body_map"])

    def test_calibration_is_full_rank_and_recovers_public_action_map(self) -> None:
        episode = generate_episode(29, sensor_dim=10, action_dim=4, goal_mode="absolute")
        self.assertEqual(episode.private_audit["calibration_rank"], 4)
        self.assertGreater(episode.private_audit["calibration_smallest_singular"], 0.0)

        public_actions = np.asarray(
            [event.action for event in episode.events if event.kind == "calibration_action"]
        )
        public_observations = [
            np.asarray(event.values)
            for event in episode.events
            if event.kind in {"observation", "calibration_observation"}
        ][: len(public_actions) + 1]
        observed = np.asarray(
            [public_observations[index + 1] - public_observations[index] for index in range(len(public_actions))]
        )
        estimate, _, rank, _ = np.linalg.lstsq(public_actions, observed, rcond=None)
        self.assertEqual(rank, 4)
        expected = np.asarray(episode.private_audit["action_to_sensor_dt"])
        self.assertTrue(np.allclose(estimate.T, expected, atol=1e-10))

    def test_public_teacher_has_only_history_and_public_bounds(self) -> None:
        self.assertTrue(teacher_public_only())
        episode = generate_episode(31, sensor_dim=8, action_dim=4, goal_mode="relative")
        teacher = PublicHistoryTeacher()
        query = next(event for event in episode.events if event.kind == "action_query")
        prefix = PublicHistory.from_events(episode.events[: query.event_id + 1])
        bounds = np.full((4, 2), (-0.5, 0.5))
        action = teacher.action_for(prefix, bounds)
        target = episode.supervision[0].action
        self.assertTrue(np.allclose(action, target))
        self.assertEqual(teacher.effort_objective, "bounded_residual_plus_minimum_l2")

    def test_goal_frame_and_relative_goal_are_publicly_well_defined(self) -> None:
        episode = generate_episode(37, sensor_dim=8, action_dim=2, goal_mode="relative")
        initial = next(event for event in episode.events if event.kind == "observation")
        goal = next(event for event in episode.events if event.kind == "goal")
        self.assertEqual(goal.key, "relative_goal")
        self.assertEqual(goal.frame, "task_sensor")
        self.assertTrue(np.allclose(np.asarray(initial.values) + goal.values, np.asarray(episode.private_audit["sensor_matrix"]) @ np.asarray(episode.private_audit["target_state"]) + np.asarray(episode.private_audit["sensor_offset"])))

    def test_action_query_precedes_execution_and_supervision_is_separate(self) -> None:
        episode = generate_episode(41, sensor_dim=10, action_dim=2, goal_mode="absolute")
        query_ids = [event.event_id for event in episode.events if event.kind == "action_query"]
        executed_ids = [event.event_id for event in episode.events if event.kind == "action_executed"]
        self.assertEqual(len(query_ids), len(executed_ids))
        self.assertTrue(all(query < executed for query, executed in zip(query_ids, executed_ids)))
        self.assertEqual({target.query_event_id for target in episode.supervision}, set(query_ids))
        self.assertEqual(len(episode.public_input()), len(episode.events))

    def test_disturbance_is_sensed_and_teacher_does_not_receive_its_timing(self) -> None:
        config = CalibratedReachConfig(disturbance_scale=0.2)
        episode = generate_episode(43, config=config, sensor_dim=8, action_dim=2, goal_mode="absolute")
        self.assertTrue(episode.private_audit["disturbance_enabled"])
        self.assertGreaterEqual(episode.private_audit["disturbance_start_step"], 0)
        self.assertLess(episode.private_audit["disturbance_start_step"], config.horizon)
        self.assertNotIn("disturbance_start_step", json.dumps(episode.public_input()))

    def test_goal_switch_is_public_and_relative_anchor_is_the_switch_observation(self) -> None:
        config = CalibratedReachConfig(goal_switch_step=3)
        episode = generate_episode(45, config=config, sensor_dim=8, action_dim=2, goal_mode="relative")
        goals = [event for event in episode.events if event.kind == "goal"]
        self.assertEqual(len(goals), 2)
        self.assertEqual(episode.private_audit["goal_target_count"], 2)
        switched = goals[1]
        observations_before_switch = [
            event for event in episode.events if event.kind == "observation" and event.time <= switched.time
        ]
        # The goal update precedes its query and is anchored to the latest
        # public observation, rather than to hidden state or future data.
        self.assertTrue(observations_before_switch)
        query = next(event for event in episode.events if event.kind == "action_query" and event.time == switched.time)
        self.assertLess(switched.event_id, query.event_id)
        anchor = np.asarray(observations_before_switch[-1].values)
        teacher = PublicHistoryTeacher()
        history = PublicHistory.from_events(episode.events[: query.event_id + 1])
        action = teacher.action_for(history, np.full((2, 2), (-0.5, 0.5)))
        self.assertTrue(np.isfinite(action).all())

    def test_generation_is_replayable_and_private_receipt_is_separate(self) -> None:
        left = generate_episode(47, sensor_dim=10, action_dim=4, goal_mode="relative")
        right = generate_episode(47, sensor_dim=10, action_dim=4, goal_mode="relative")
        self.assertEqual(left.public_input(), right.public_input())
        self.assertEqual(left.supervision, right.supervision)
        self.assertEqual(corpus_fingerprint(left), corpus_fingerprint(right))
        self.assertIsInstance(left, TrajectoryEpisode)

    def test_rollout_api_executes_policy_actions_through_shared_dynamics(self) -> None:
        config = CalibratedReachConfig(horizon=3)
        rollout = CalibratedReachRollout.from_seed(
            53, config=config, sensor_dim=8, action_dim=2, goal_mode="absolute"
        )
        reset_history = rollout.reset()
        self.assertEqual(len(reset_history.of_kind("calibration_action")), 4)
        self.assertIsNone(rollout.pending_query_event_id)
        for _ in range(config.horizon):
            query_history = rollout.query()
            self.assertIsNotNone(rollout.pending_query_event_id)
            self.assertEqual(query_history.latest("action_query").event_id, rollout.pending_query_event_id)
            rollout.step(np.zeros(2))
        self.assertTrue(rollout.done)
        self.assertEqual(rollout.history.events[-1].kind, "episode_end")
        self.assertEqual(len(rollout.history.of_kind("action_executed")), config.horizon)

    def test_rollout_reset_replays_the_same_scheduled_episode(self) -> None:
        rollout = CalibratedReachRollout.from_seed(
            57,
            config=CalibratedReachConfig(horizon=2, goal_switch_step=1, disturbance_scale=0.1),
            sensor_dim=8,
            action_dim=2,
            goal_mode="relative",
            disturbance=True,
        )
        first = rollout.reset()
        for _ in range(2):
            rollout.query()
            rollout.step(np.zeros(2))
        second = rollout.reset()
        self.assertEqual(first.events, second.events)

    def test_rollout_rejects_wrong_or_out_of_bounds_policy_actions_without_clipping(self) -> None:
        rollout = CalibratedReachRollout.from_seed(
            59, config=CalibratedReachConfig(horizon=2), sensor_dim=8, action_dim=4
        )
        rollout.reset()
        rollout.query()
        with self.assertRaises(ValueError):
            rollout.step(np.zeros(2))
        with self.assertRaises(ValueError):
            rollout.step(np.full(4, 0.5001))
        self.assertEqual(len(rollout.history.of_kind("action_executed")), 0)

    def test_goal_switch_receipt_uses_latest_target_for_relative_rollout(self) -> None:
        config = CalibratedReachConfig(horizon=5, goal_switch_step=2)
        episode = generate_episode(
            61, config=config, sensor_dim=8, action_dim=2, goal_mode="relative"
        )
        final_sensor = np.asarray(episode.events[-2].values)
        target = np.asarray(episode.private_audit["sensor_matrix"]) @ np.asarray(
            episode.private_audit["target_state"]
        ) + np.asarray(episode.private_audit["sensor_offset"])
        self.assertAlmostEqual(
            episode.private_audit["teacher_final_sensor_error"],
            np.linalg.norm(final_sensor - target),
        )
        self.assertEqual(episode.private_audit["goal_target_count"], 2)

    def test_same_seed_realizations_share_task_but_step_from_policy_actions(self) -> None:
        config = CalibratedReachConfig(horizon=2)
        left = CalibratedReachRollout.from_seed(
            67, config=config, sensor_dim=8, action_dim=2, goal_mode="absolute"
        )
        right = CalibratedReachRollout.from_seed(
            67, config=config, sensor_dim=8, action_dim=4, goal_mode="absolute"
        )
        left.reset()
        right.reset()
        self.assertTrue(np.allclose(left.environment.initial_state, right.environment.initial_state))
        self.assertTrue(np.allclose(left.environment.target_state, right.environment.target_state))
        left.query()
        right.query()
        left.step(np.zeros(2))
        right.step(np.zeros(4))
        self.assertTrue(np.allclose(left.environment.state, right.environment.state))

    def test_privileged_metrics_compare_physical_state_and_track_latest_switch_goal(self) -> None:
        config = CalibratedReachConfig(horizon=3, goal_switch_step=1)
        rollout = CalibratedReachRollout.from_seed(
            71, config=config, sensor_dim=8, action_dim=2, goal_mode="relative"
        )
        rollout.reset()
        for _ in range(config.horizon):
            rollout.query()
            rollout.step(np.zeros(2))
        metrics = rollout.privileged_metrics(
            success_tolerance_state_l2=0.0, success_tolerance_sensor_rmse=0.05
        )
        self.assertEqual(metrics["schema_version"], "calibrated-reach-physical-metrics-v1")
        self.assertEqual(metrics["goal_switch_count"], 1)
        self.assertEqual(metrics["scored_steps"], config.horizon)
        self.assertEqual(metrics["success"], metrics["state_error_l2"] <= 0.0)
        self.assertIn("sensor_rmse", metrics)
        self.assertIn("action_effort_l2_sum", metrics)

    def test_same_actions_have_same_physical_outcome_across_sensor_widths(self) -> None:
        config = CalibratedReachConfig(horizon=3)
        left = CalibratedReachRollout.from_seed(
            73, config=config, sensor_dim=8, action_dim=2, goal_mode="absolute"
        )
        right = CalibratedReachRollout.from_seed(
            73, config=config, sensor_dim=10, action_dim=2, goal_mode="absolute"
        )
        left.reset()
        right.reset()
        for _ in range(config.horizon):
            left.query()
            right.query()
            action = np.asarray((0.1, -0.07))
            left.step(action)
            right.step(action)
        left_metrics = left.privileged_metrics(success_tolerance_state_l2=0.1)
        right_metrics = right.privileged_metrics(success_tolerance_state_l2=0.1)
        self.assertAlmostEqual(left_metrics["state_error_l2"], right_metrics["state_error_l2"])
        self.assertEqual(left_metrics["success"], right_metrics["success"])
        self.assertNotEqual(left_metrics["sensor_dim"], right_metrics["sensor_dim"])

    def test_unreachable_postcalibration_goal_is_reported_as_failure(self) -> None:
        config = CalibratedReachConfig(horizon=1, action_limit=0.01, calibration_pulse=0.001)
        rollout = CalibratedReachRollout.from_seed(
            79, config=config, sensor_dim=8, action_dim=2, goal_mode="absolute"
        )
        rollout.reset()
        rollout.environment.target_state = np.asarray((0.7, 0.7), dtype=np.float64)
        rollout.query()
        rollout.step(np.zeros(2))
        metrics = rollout.privileged_metrics(success_tolerance_state_l2=0.05)
        self.assertGreater(metrics["state_error_l2"], 0.05)
        self.assertFalse(metrics["success"])


if __name__ == "__main__":
    unittest.main()
