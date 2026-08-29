from __future__ import annotations

import unittest

import pretraining_world_py

from pretraining_experiments.baselines import baseline_band
from pretraining_experiments.data import MODEL_FIELDS, assert_world_model_compatibility, tensorize
from pretraining_experiments.model import PretrainingConfig


WORLD = {
    "d_min": 1,
    "d_max": 4,
    "gain_min": 0.75,
    "gain_max": 1.25,
    "action_limit": 0.20,
    "calibration_pulse": 0.10,
    "max_control_steps": 4,
}


class WorldContractTests(unittest.TestCase):
    def test_versioned_role_contract_is_0_2(self) -> None:
        versions = pretraining_world_py.versions()
        self.assertEqual(versions["world"], "calibrated-monomial-0.2.0")
        self.assertEqual(versions["token_abi"], "physical-event-abi-0.2.0")
        self.assertEqual(versions["role_count"], 11)
        raw = pretraining_world_py.generate_training_batch(
            seed=11,
            start_index=0,
            batch_size=1,
            max_tokens=192,
            **WORLD,
        )
        public_roles = set(raw["role_ids"][0][: raw["lengths"][0]])
        self.assertIn(5, public_roles)  # Goal
        self.assertIn(9, public_roles)  # FutureQuery

    def test_model_config_and_world_abi_match(self) -> None:
        config = PretrainingConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            attention_heads=4,
            max_position_embeddings=192,
        )
        assert_world_model_compatibility(config)
        config.num_roles = 10
        with self.assertRaises(RuntimeError):
            assert_world_model_compatibility(config)

    def test_profiled_interface_is_explicit_and_shape_compatible(self) -> None:
        config = PretrainingConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            attention_heads=4,
            max_position_embeddings=193,
            token_abi_version="physical-event-abi-0.3.1",
        )
        assert_world_model_compatibility(config, profiled=True)
        with self.assertRaises(RuntimeError):
            assert_world_model_compatibility(config, profiled=False)

        legacy = pretraining_world_py.generate_training_batch(
            seed=17,
            start_index=0,
            batch_size=2,
            max_tokens=193,
            **WORLD,
        )
        profiled = pretraining_world_py.generate_training_batch(
            seed=17,
            start_index=0,
            batch_size=2,
            max_tokens=193,
            profiled=True,
            **WORLD,
        )
        self.assertEqual(profiled["token_abi_version"], "physical-event-abi-0.3.1")
        self.assertEqual(profiled["lengths"], [length + 1 for length in legacy["lengths"]])
        self.assertTrue(all(role == 4 for role in (row[0] for row in profiled["role_ids"])))
        self.assertTrue(all(key == 1 for key in (row[0] for row in profiled["key_ids"])))

    def test_public_oracle_and_support_validation(self) -> None:
        result = pretraining_world_py.validate_generated_worlds(
            seed=123,
            start_index=0,
            count=256,
            **WORLD,
        )
        self.assertEqual(result["dimension_counts"][1:5], [64, 64, 64, 64])
        self.assertLess(result["max_oracle_error"], 1.0e-5)
        self.assertGreater(result["action_targets"], 0)
        self.assertGreater(result["future_targets"], result["action_targets"])
        self.assertLessEqual(result["max_length"], 192)

    def test_batched_public_tensor_contract(self) -> None:
        raw = pretraining_world_py.generate_training_batch(
            seed=17,
            start_index=0,
            batch_size=8,
            max_tokens=192,
            **WORLD,
        )
        tensors = tensorize(raw)
        self.assertEqual(set(tensors), set(MODEL_FIELDS))
        self.assertEqual(tuple(tensors["role_ids"].shape), (8, 192))
        self.assertEqual(tuple(tensors["payloads"].shape), (8, 192, 8))
        self.assertEqual(tuple(tensors["action_targets"].shape), (8, 192, 16))
        self.assertTrue((tensors["action_target_mask"].sum(dim=-1) <= 1).all())
        self.assertNotIn("indices", tensors)
        self.assertNotIn("dimensions", tensors)

    def test_truncation_is_a_loud_error(self) -> None:
        with self.assertRaises(ValueError):
            pretraining_world_py.generate_training_batch(
                seed=17,
                start_index=3,
                batch_size=1,
                max_tokens=16,
                **WORLD,
            )

    def test_privileged_oracle_solves_but_zero_policy_does_not(self) -> None:
        oracle = pretraining_world_py.RolloutBatch(
            seed=41,
            start_index=0,
            batch_size=16,
            max_tokens=192,
            **WORLD,
        )
        while not oracle.all_done():
            oracle.step(oracle.privileged_oracle_actions())
        oracle_summary = oracle.summary()
        self.assertTrue(all(oracle_summary["success"]))

        zero = pretraining_world_py.RolloutBatch(
            seed=41,
            start_index=0,
            batch_size=16,
            max_tokens=192,
            **WORLD,
        )
        while not zero.all_done():
            batch = zero.learner_batch()
            actions = [([] if done else [0.0] * dimension) for done, dimension in zip(batch["done"], batch["dimensions"])]
            zero.step(actions)
        self.assertFalse(any(zero.summary()["success"]))

    def test_trivial_policy_band_is_decision_facing(self) -> None:
        config = {
            "model": {"sequence_length": 192},
            "world": {**WORLD, "rollout_seed": 41},
            "evaluation": {
                "error_thresholds": [0.01, 0.05, 0.10],
                "trivial_policy_scales": [0.0, 0.75, 1.0],
                "baseline_episodes": 16,
            },
        }
        result = baseline_band(config)
        zero = result["policies"]["scaled_oracle_0"]
        partial = result["policies"]["scaled_oracle_0.75"]
        oracle = result["policies"]["scaled_oracle_1"]
        self.assertEqual(len(zero["trial_curve"]), WORLD["max_control_steps"] + 1)
        self.assertEqual(zero["threshold_success"]["0.05"], 0.0)
        self.assertGreater(partial["mean_error_reduction"], zero["mean_error_reduction"])
        self.assertEqual(oracle["threshold_success"]["0.05"], 1.0)


if __name__ == "__main__":
    unittest.main()
