from __future__ import annotations

import unittest

import torch

import pretraining_world_py

from pretraining_experiments.data import tensorize
from pretraining_experiments.goal_conditioning import classify_against, evaluate_model
from pretraining_experiments.model import PretrainingConfig, PretrainingForTrajectoryPrediction


def tiny_config() -> PretrainingConfig:
    return PretrainingConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        attention_heads=4,
        max_position_embeddings=64,
        num_roles=11,
        payload_dim=8,
        action_horizon=16,
        token_abi_version="physical-event-abi-0.2.0",
    )


def score(successes: int, cases: int) -> dict[str, float | int]:
    return {"successes": successes, "cases": cases, "rate": successes / cases}


class GoalConditioningIntegrationTests(unittest.TestCase):
    def test_diagnostic_uses_the_versioned_real_learner_abi(self) -> None:
        versions = pretraining_world_py.versions()
        self.assertEqual(
            versions["diagnostic_serialization"],
            "goal-conditioned-continuous-control-0.1.0",
        )
        batch = pretraining_world_py.GoalConditioningRolloutBatch(
            max_tokens=64,
            serialization_order="canonical",
        )
        raw = batch.learner_batch()
        tensors = tensorize(raw)
        self.assertEqual(tuple(tensors["role_ids"].shape), (9, 64))
        self.assertEqual(tuple(tensors["payloads"].shape), (9, 64, 8))
        self.assertTrue(all(position >= 0 for position in raw["query_positions"]))

    def test_hidden_goal_pairs_have_identical_initial_model_inputs(self) -> None:
        raw = pretraining_world_py.GoalConditioningRolloutBatch(
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

    def test_supervision_perturbation_cannot_change_predictions(self) -> None:
        torch.manual_seed(2)
        model = PretrainingForTrajectoryPrediction(tiny_config()).eval()
        raw = pretraining_world_py.GoalConditioningRolloutBatch(
            max_tokens=64,
            serialization_order="canonical",
        ).learner_batch()
        original = tensorize(raw)
        perturbed = {name: value.clone() for name, value in original.items()}
        perturbed["action_targets"].uniform_(-1.0, 1.0)
        perturbed["action_target_mask"].fill_(1.0)
        perturbed["future_targets"].uniform_(-1.0, 1.0)
        perturbed["future_target_mask"].fill_(1.0)
        with torch.no_grad():
            before = model(**original).action_predictions
            after = model(**perturbed).action_predictions
        self.assertTrue(torch.equal(before, after))

    def test_representation_probe_arms_have_matched_training_volume(self) -> None:
        fixed = pretraining_world_py.generate_goal_conditioning_training_batch(
            arm="fixed", max_tokens=64
        )
        orbit = pretraining_world_py.generate_goal_conditioning_training_batch(
            arm="orbit", max_tokens=64
        )
        self.assertEqual(fixed["semantic_cases"], orbit["semantic_cases"])
        self.assertEqual(fixed["records"], orbit["records"])
        self.assertEqual(fixed["lengths"], orbit["lengths"])
        self.assertEqual(
            sum(sum(sum(row) for row in record) for record in fixed["action_target_mask"]),
            sum(sum(sum(row) for row in record) for record in orbit["action_target_mask"]),
        )
        self.assertNotEqual(fixed["key_ids"], orbit["key_ids"])

    def test_untrained_learner_evaluation_is_local_evidence_not_transfer(self) -> None:
        torch.manual_seed(3)
        result = evaluate_model(PretrainingForTrajectoryPrediction(tiny_config()), max_tokens=64)
        evidence = result["checkpoint_evidence"]
        self.assertIsNone(evidence["transfer"])
        self.assertLessEqual(evidence["diagnostics"]["hidden_goal_check"]["rate"], 0.5)
        self.assertGreaterEqual(evidence["diagnostics"]["presentation_order_invariance"], 0.0)
        self.assertLessEqual(evidence["diagnostics"]["presentation_order_invariance"], 1.0)
        self.assertIn("not project-level progress", result["interpretation"])

    def test_rust_gate_rejects_an_order_dependent_apparent_gain(self) -> None:
        previous = {
            "headline_metric": 0.4,
            "diagnostics": {
                "witness": score(1, 2),
                "fixed_goal_control": score(1, 1),
                "state_predicts_goal_control": score(2, 2),
                "hidden_goal_check": score(1, 2),
                "renamed_witness": score(1, 2),
                "counterfactual_goal_sensitivity": 0.0,
                "presentation_order_invariance": 1.0,
            },
            "transfer": None,
        }
        candidate = {
            "headline_metric": 0.9,
            "diagnostics": {
                "witness": score(2, 2),
                "fixed_goal_control": score(1, 1),
                "state_predicts_goal_control": score(2, 2),
                "hidden_goal_check": score(1, 2),
                "renamed_witness": score(2, 2),
                "counterfactual_goal_sensitivity": 1.0,
                "presentation_order_invariance": 0.5,
            },
            "transfer": None,
        }
        decision = classify_against(previous, candidate)
        self.assertEqual(decision["class"], "FalseProgress")
        self.assertFalse(decision["accept_local_step"])
        self.assertTrue(any("reordered" in reason for reason in decision["failed_checks"]))


if __name__ == "__main__":
    unittest.main()
