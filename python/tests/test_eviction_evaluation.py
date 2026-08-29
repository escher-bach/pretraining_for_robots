from __future__ import annotations

import unittest

import torch

import pretraining_world_py

from pretraining_experiments.data import tensorize
from pretraining_experiments.eviction_evaluation import (
    evaluate_model,
    first_action_diagnostic,
)
from pretraining_experiments.goal_conditioning import classify_against
from pretraining_experiments.model import PretrainingConfig, PretrainingForTrajectoryPrediction


def tiny_profiled_config() -> PretrainingConfig:
    return PretrainingConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        attention_heads=4,
        max_position_embeddings=64,
        num_roles=11,
        payload_dim=8,
        action_horizon=16,
        token_abi_version="physical-event-abi-0.3.1",
    )


class EvictionLearnerBoundaryTests(unittest.TestCase):
    def test_profile_header_reaches_the_actual_model_tensors(self) -> None:
        raw = pretraining_world_py.EvictionRolloutBatch(
            max_tokens=64,
            serialization_order="canonical",
        ).learner_batch()
        tensors = tensorize(raw)
        self.assertEqual(raw["token_abi_version"], "physical-event-abi-0.3.1")
        self.assertEqual(tuple(tensors["role_ids"].shape), (9, 64))
        self.assertTrue(torch.all(tensors["role_ids"][:, 0] == 4))
        self.assertTrue(torch.all(tensors["key_ids"][:, 0] == 2))
        self.assertTrue(torch.all(tensors["position_ids"][:, 0] == 0))
        self.assertTrue(
            torch.equal(
                tensors["payloads"][0, 0],
                torch.tensor([0.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            )
        )
        self.assertEqual(float(tensors["action_target_mask"][:, 0].sum()), 0.0)
        self.assertTrue(all(position > 0 for row in raw["query_positions"] for position in row))

    def test_hidden_goal_pair_has_identical_initial_model_inputs(self) -> None:
        raw = pretraining_world_py.EvictionRolloutBatch(
            max_tokens=64,
            serialization_order="canonical",
        ).learner_batch()
        indices = [
            index
            for index, kind in enumerate(raw["case_kinds"])
            if kind == "hidden_goal_leakage_check"
        ]
        self.assertEqual(len(indices), 2)
        left, right = indices
        for field in ("role_ids", "key_ids", "position_ids", "payloads", "attention_mask"):
            self.assertEqual(raw[field][left], raw[field][right], field)

    def test_validation_oracle_solves_every_complete_episode(self) -> None:
        rollout = pretraining_world_py.EvictionRolloutBatch(
            max_tokens=64,
            serialization_order="canonical",
        )
        while not rollout.all_done():
            rollout.step(rollout.privileged_oracle_actions())
        summary = rollout.summary()
        self.assertTrue(all(summary["success"]))
        self.assertTrue(all(summary["first_action_optimal"]))

    def test_opening_contrast_can_pass_while_both_witnesses_fail(self) -> None:
        summary = {
            "case_ids": ["witness-goal-occupied", "witness-goal-free"],
            "case_kinds": ["witness", "witness"],
            "commands": [["evict_1", "evict_1"], ["evict_2", "evict_2"]],
            "first_action_optimal": [True, True],
            "success": [False, False],
        }
        opening = first_action_diagnostic(summary)
        self.assertTrue(opening["passes_opening_contrast"])
        self.assertEqual(sum(summary["success"]), 0)

    def test_hill_climb_rejects_first_action_only_progress(self) -> None:
        def score(successes: int, cases: int) -> dict[str, float | int]:
            return {"successes": successes, "cases": cases, "rate": successes / cases}

        previous = {
            "headline_metric": 0.0,
            "diagnostics": {
                "witness": score(0, 2),
                "fixed_goal_control": score(0, 1),
                "state_predicts_goal_control": score(0, 2),
                "hidden_goal_check": score(0, 2),
                "renamed_witness": score(0, 2),
                "counterfactual_goal_sensitivity": 0.0,
                "presentation_order_invariance": 1.0,
            },
            "transfer": None,
        }
        candidate = {
            "headline_metric": 0.9,
            "diagnostics": {
                "witness": score(0, 2),
                "fixed_goal_control": score(1, 1),
                "state_predicts_goal_control": score(2, 2),
                "hidden_goal_check": score(0, 2),
                "renamed_witness": score(0, 2),
                "counterfactual_goal_sensitivity": 1.0,
                "presentation_order_invariance": 1.0,
            },
            "transfer": None,
        }
        decision = classify_against(previous, candidate)
        self.assertEqual(decision["class"], "FalseProgress")
        self.assertFalse(decision["accept_local_step"])
        self.assertTrue(any("witness" in reason for reason in decision["failed_checks"]))

    def test_random_model_evaluation_keeps_evidence_levels_separate(self) -> None:
        torch.manual_seed(4)
        result = evaluate_model(
            PretrainingForTrajectoryPrediction(tiny_profiled_config()), max_tokens=64
        )
        self.assertIn("witness", result["complete_episode_success"])
        self.assertIn("passes_opening_contrast", result["first_action_diagnostic"])
        self.assertEqual(
            result["policy_evidence"]["witness"],
            result["complete_episode_success"]["witness"],
        )
        self.assertEqual(
            result["policy_evidence"]["counterfactual_goal_sensitivity"],
            float(result["first_action_diagnostic"]["passes_opening_contrast"]),
        )
        self.assertNotIn("transfer", result)
        self.assertEqual(result["hidden_goal_public_ceiling"], 0.5)
        self.assertGreater(
            result["serialization_runs"]["canonical"]["throughput"][
                "episodes_per_second"
            ],
            0.0,
        )
        self.assertIn("matched learning curves", result["interpretation"])

    def test_legacy_goal_interface_remains_default_and_profiled_is_opt_in(self) -> None:
        legacy = pretraining_world_py.GoalConditioningRolloutBatch(
            max_tokens=64,
            serialization_order="canonical",
        ).learner_batch()
        profiled = pretraining_world_py.GoalConditioningRolloutBatch(
            max_tokens=64,
            serialization_order="canonical",
            profiled=True,
        ).learner_batch()
        self.assertEqual(legacy["token_abi_version"], "physical-event-abi-0.2.0")
        self.assertEqual(profiled["token_abi_version"], "physical-event-abi-0.3.1")
        self.assertEqual(profiled["lengths"], [length + 1 for length in legacy["lengths"]])
        self.assertEqual(
            profiled["query_positions"],
            [position + 1 for position in legacy["query_positions"]],
        )


if __name__ == "__main__":
    unittest.main()
