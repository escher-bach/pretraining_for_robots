from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from accelerate import Accelerator
from transformers import set_seed

from pretraining_experiments.data import generate_torch_batch, tensorize
from pretraining_experiments.model import (
    PretrainingConfig,
    PretrainingForTrajectoryPrediction,
    PretrainingOutput,
    assert_selected_parameter_report,
    assert_selected_profile,
    parameter_report,
)
from pretraining_experiments.evaluation import paired_learning_delta
from pretraining_experiments.train import (
    FixedCohortDataset,
    closed_loop_eval,
    held_out_learner_evaluation,
    run_overfit_gate,
    train_with_resume_smoke,
)


ROOT = Path(__file__).resolve().parents[2]
SELECTED_CONFIG = ROOT / "artifacts" / "icrt-derived-small" / "model_config.json"
WORLD = {
    "d_min": 1,
    "d_max": 4,
    "gain_min": 0.75,
    "gain_max": 1.25,
    "action_limit": 0.20,
    "calibration_pulse": 0.10,
    "max_control_steps": 4,
}


def tiny_config() -> PretrainingConfig:
    return PretrainingConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        attention_heads=4,
        max_position_embeddings=192,
        num_roles=11,
        payload_dim=8,
        action_horizon=16,
        token_abi_version="physical-event-abi-0.2.0",
    )


def adapter_batch() -> dict[str, torch.Tensor]:
    """Small public batch for CPU-only modality-boundary invariants."""
    batch = 1
    tokens = 8
    horizon = 16
    attention = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.long)
    return {
        "role_ids": torch.ones(batch, tokens, dtype=torch.long),
        "key_ids": torch.arange(tokens, dtype=torch.long).unsqueeze(0),
        "position_ids": torch.arange(tokens, dtype=torch.long).unsqueeze(0),
        "payloads": torch.zeros(batch, tokens, 8),
        "attention_mask": attention,
        "action_targets": torch.zeros(batch, tokens, horizon),
        "action_target_mask": torch.zeros(batch, tokens, horizon),
        "future_targets": torch.zeros(batch, tokens),
        "future_target_mask": torch.zeros(batch, tokens),
    }


class ModelIntegrationTests(unittest.TestCase):
    def test_output_exposes_raw_action_logits_without_changing_tanh_predictions(self) -> None:
        torch.manual_seed(13)
        model = PretrainingForTrajectoryPrediction(tiny_config())
        batch, _ = generate_torch_batch(
            seed=1313, start_index=0, batch_size=1, max_tokens=192, world=WORLD
        )
        output = model(**batch)
        self.assertIsNotNone(output.action_logits)
        self.assertTrue(torch.allclose(output.action_predictions, torch.tanh(output.action_logits)))

    def test_grouped_action_query_loss_uses_categorical_alternatives_and_gradients(self) -> None:
        # Two decision groups; the second has two equally correct alternatives.
        # Raw logits are intentionally used: tanh-L1 saturation must not be the
        # path that trains this categorical readout.
        logits = torch.tensor([[[0.0], [1.0], [2.0], [0.0], [0.0]]], requires_grad=True)
        targets = torch.tensor([[[1.0], [0.0], [1.0], [1.0], [0.0]]])
        mask = torch.ones_like(targets)
        groups = torch.tensor([[7, 7, 9, 9, 9]])
        loss = PretrainingForTrajectoryPrediction._grouped_action_query_cross_entropy(
            logits, targets, mask, groups
        )
        # group 7: -log softmax(0,1)[0]; group 9: uniform target over 2,0,0.
        expected = -torch.log_softmax(logits[0, :2, 0], dim=0)[0]
        expected = expected + -0.5 * torch.log_softmax(logits[0, 2:, 0], dim=0)[:2].sum()
        expected = expected / 2.0
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_group_metadata_is_loss_only_and_legacy_l1_is_unchanged_when_absent(self) -> None:
        torch.manual_seed(14)
        model = PretrainingForTrajectoryPrediction(tiny_config())
        batch, _ = generate_torch_batch(
            seed=1414, start_index=0, batch_size=1, max_tokens=192, world=WORLD
        )
        with patch.object(model, "embed_events", wraps=model.embed_events) as embedded:
            legacy = model(**batch)
            with self.assertRaisesRegex(ValueError, "needs a nonnegative group id"):
                model(
                    **batch,
                    action_decision_groups=torch.full((1, 192), -1, dtype=torch.long),
                )
        # The ordinary world has supervised rows, so malformed group metadata
        # must fail rather than silently become a learner-visible input.
        self.assertIsNotNone(legacy.action_loss)
        expected_legacy = model._masked_l1(
            legacy.action_predictions, batch["action_targets"], batch["action_target_mask"]
        )
        self.assertTrue(torch.allclose(legacy.action_loss, expected_legacy))
        self.assertEqual(len(embedded.call_args_list), 2)
        for call in embedded.call_args_list:
            self.assertNotIn("action_decision_groups", call.kwargs)

    def test_selected_exact_profile_runs_real_world_backward(self) -> None:
        torch.manual_seed(1)
        config = PretrainingConfig.from_project_json(SELECTED_CONFIG)
        assert_selected_profile(config)
        model = PretrainingForTrajectoryPrediction(config)
        report = parameter_report(model)
        assert_selected_parameter_report(report)
        batch, _ = generate_torch_batch(
            seed=101,
            start_index=3,
            batch_size=1,
            max_tokens=192,
            world=WORLD,
        )
        output = model(**batch)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(model.action_head.weight.grad)
        self.assertTrue(torch.isfinite(model.action_head.weight.grad).all())

    def test_fixed_real_batch_overfits_with_tiny_core(self) -> None:
        torch.manual_seed(2)
        model = PretrainingForTrajectoryPrediction(tiny_config())
        batch, _ = generate_torch_batch(
            seed=202,
            start_index=0,
            batch_size=2,
            max_tokens=192,
            world=WORLD,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3)
        model.eval()
        with torch.no_grad():
            initial = model(**batch).loss.item()
        model.train()
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            final = model(**batch).loss.item()
        self.assertLess(final, 0.75 * initial, (initial, final))

    def test_standard_pretrained_round_trip_preserves_outputs(self) -> None:
        torch.manual_seed(3)
        model = PretrainingForTrajectoryPrediction(tiny_config()).eval()
        batch, _ = generate_torch_batch(
            seed=303,
            start_index=1,
            batch_size=1,
            max_tokens=192,
            world=WORLD,
        )
        with torch.no_grad():
            before = model(**batch).action_predictions
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            restored = PretrainingForTrajectoryPrediction.from_pretrained(directory).eval()
            with torch.no_grad():
                after = restored(**batch).action_predictions
        self.assertTrue(torch.equal(before, after))

    def test_external_adapter_tensorization_is_optional_and_not_metadata(self) -> None:
        raw = {
            "role_ids": [[1, 2]],
            "key_ids": [[0, 1]],
            "position_ids": [[0, 1]],
            "payloads": [[[0.0] * 8, [0.0] * 8]],
            "attention_mask": [[1, 1]],
            "action_targets": [[[0.0] * 16, [0.0] * 16]],
            "action_target_mask": [[[0.0] * 16, [0.0] * 16]],
            "future_targets": [[0.0, 0.0]],
            "future_target_mask": [[0.0, 0.0]],
            "canonical_content_embeds": [[[0.0] * 64, [1.0] * 64]],
            "canonical_content_mask": [[0, 1]],
            "generator_only": "must not become a learner input",
        }
        tensors = tensorize(raw)
        self.assertEqual(tuple(tensors["canonical_content_embeds"].shape), (1, 2, 64))
        self.assertEqual(tensors["canonical_content_mask"].dtype, torch.bool)
        self.assertNotIn("generator_only", tensors)

    def test_external_adapter_requires_aligned_content_and_mask(self) -> None:
        torch.manual_seed(4)
        model = PretrainingForTrajectoryPrediction(tiny_config())
        batch = adapter_batch()
        canonical = torch.zeros(1, 8, 64)
        aligned = batch["attention_mask"].to(dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "supplied together"):
            model(**batch, canonical_content_embeds=canonical)
        with self.assertRaisesRegex(ValueError, "supplied together"):
            model(**batch, canonical_content_mask=aligned)
        with self.assertRaisesRegex(ValueError, "canonical_content_embeds must have shape"):
            model(**batch, canonical_content_embeds=canonical[..., :-1], canonical_content_mask=aligned)
        with self.assertRaisesRegex(ValueError, "canonical_content_mask must have shape"):
            model(**batch, canonical_content_embeds=canonical, canonical_content_mask=aligned[:, :-1])
        bad_alignment = aligned.clone()
        bad_alignment[0, batch["attention_mask"][0].sum()] = True
        with self.assertRaisesRegex(ValueError, "unmasked event positions"):
            model(**batch, canonical_content_embeds=canonical, canonical_content_mask=bad_alignment)
        nonfinite = canonical.clone()
        nonfinite[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "aligned canonical_content_embeds must be finite"):
            model(**batch, canonical_content_embeds=nonfinite, canonical_content_mask=aligned)
        self.assertFalse(any(name.startswith("visual_") for name, _ in model.named_parameters()))

    def test_masked_adapter_garbage_is_inert_and_gradients_are_aligned(self) -> None:
        torch.manual_seed(41)
        model = PretrainingForTrajectoryPrediction(tiny_config()).eval()
        batch = adapter_batch()
        content_mask = torch.zeros_like(batch["attention_mask"], dtype=torch.bool)
        content_mask[0, :3] = True
        canonical = torch.randn(1, 8, 64, requires_grad=True)
        garbage = canonical.detach().clone()
        garbage[:, ~content_mask[0]] = torch.randn_like(garbage[:, ~content_mask[0]]) * 1.0e6
        garbage[0, 4, 0] = float("nan")
        garbage[0, 5, 1] = float("inf")
        with torch.no_grad():
            clean_output = model(
                **batch,
                canonical_content_embeds=canonical.detach(),
                canonical_content_mask=content_mask,
            )
            garbage_output = model(
                **batch,
                canonical_content_embeds=garbage,
                canonical_content_mask=content_mask,
            )
        self.assertTrue(torch.equal(clean_output.action_logits, garbage_output.action_logits))

        model(
            **batch,
            canonical_content_embeds=canonical,
            canonical_content_mask=content_mask,
        ).action_logits.sum().backward()
        self.assertTrue(torch.equal(canonical.grad[:, ~content_mask[0]], torch.zeros_like(canonical.grad[:, ~content_mask[0]])))
        self.assertGreater(float(canonical.grad[:, content_mask[0]].abs().sum()), 0.0)

    def test_adapter_content_preserves_causal_suffix_invariance(self) -> None:
        torch.manual_seed(42)
        model = PretrainingForTrajectoryPrediction(tiny_config()).eval()
        batch = adapter_batch()
        query = 2
        content_mask = batch["attention_mask"].to(dtype=torch.bool)
        first = torch.randn(1, 8, 64)
        second = first.clone()
        second[:, query + 1 :] = torch.randn_like(second[:, query + 1 :])
        with torch.no_grad():
            left = model(
                **batch,
                canonical_content_embeds=first,
                canonical_content_mask=content_mask,
            )
            right = model(
                **batch,
                canonical_content_embeds=second,
                canonical_content_mask=content_mask,
            )
        self.assertTrue(torch.equal(left.action_logits[:, query], right.action_logits[:, query]))

    def test_key_encoding_has_initialization_scale(self) -> None:
        model = PretrainingForTrajectoryPrediction(tiny_config())
        keys = torch.tensor([[0, 1, 17, 65535]])
        norms = model.deterministic_key_embedding(keys).norm(dim=-1)
        expected = tiny_config().initializer_range * (tiny_config().hidden_size / 2) ** 0.5
        self.assertTrue(torch.allclose(norms, torch.full_like(norms, expected), atol=1.0e-6))

    def test_equal_rank_shards_match_full_batch_loss(self) -> None:
        torch.manual_seed(5)
        model = PretrainingForTrajectoryPrediction(tiny_config()).eval()
        batch, _ = generate_torch_batch(
            seed=505,
            start_index=0,
            batch_size=4,
            max_tokens=192,
            world=WORLD,
        )
        with torch.no_grad():
            full = model(**batch)
            left = model(**{name: value[:2] for name, value in batch.items()})
            right = model(**{name: value[2:] for name, value in batch.items()})
        for field in ("loss", "action_loss", "future_loss"):
            expected = (getattr(left, field) + getattr(right, field)) / 2
            self.assertTrue(torch.allclose(getattr(full, field), expected, atol=1.0e-6))

    def test_trainer_gate_runs_on_real_rust_batch(self) -> None:
        config = {
            "run": {
                "seed": 5,
                "overfit_per_device_batch_size": 1,
                "learning_rate": 2.0e-3,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "overfit_updates": 12,
                "overfit_warmup_updates": 2,
                "overfit_required_fraction": 0.90,
            },
            "model": {"sequence_length": 192},
            "world": {**WORLD, "overfit_seed": 505},
        }
        with tempfile.TemporaryDirectory() as directory:
            result, trainer = run_overfit_gate(
                tiny_config(), config, Path(directory), use_cpu=True
            )
            self.assertTrue(result["passed"])
            self.assertEqual(result["successful_optimizer_steps"], 12)
            self.assertFalse(trainer.model_accepts_loss_kwargs)
            del trainer

    def test_model_and_oracle_closed_loop_paths_share_rust_rollout(self) -> None:
        accelerator = Accelerator(cpu=True)
        model = PretrainingForTrajectoryPrediction(tiny_config()).to(accelerator.device)
        config = {
            "model": {"sequence_length": 192},
            "world": {**WORLD, "d_min": 1, "d_max": 4},
        }
        model_result = closed_loop_eval(
            accelerator,
            model,
            config,
            seed=606,
            start_index=0,
            episodes_per_rank=4,
        )
        oracle_result = closed_loop_eval(
            accelerator,
            model,
            config,
            seed=606,
            start_index=0,
            episodes_per_rank=4,
            use_oracle=True,
        )
        self.assertEqual(model_result["episodes"], 4)
        self.assertEqual(len(model_result["trial_curve"]), WORLD["max_control_steps"] + 1)
        self.assertIn("0.05", model_result["threshold_success"])
        self.assertIn("terminal_error_distribution", model_result)
        self.assertEqual(oracle_result["success_rate"], 1.0)
        accelerator.free_memory()

    def test_trainer_checkpoint_resumes_optimizer_and_scheduler(self) -> None:
        batch, _ = generate_torch_batch(
            seed=707,
            start_index=0,
            batch_size=2,
            max_tokens=192,
            world=WORLD,
        )
        dataset = FixedCohortDataset(batch, repeats=8)
        run = {
            "seed": 7,
            "mixed_precision": "no",
            "gradient_accumulation_steps": 1,
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "log_every": 1,
            "resume_smoke_update": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            set_seed(7)
            resumed, metrics, smoke = train_with_resume_smoke(
                model=PretrainingForTrajectoryPrediction(tiny_config()),
                dataset=dataset,
                trainer_state_dir=root,
                run=run,
                total_steps=3,
                per_device_batch_size=1,
                warmup_steps=1,
                use_cpu=True,
            )
            checkpoint = root / "checkpoint-2"
            self.assertTrue((checkpoint / "optimizer.pt").is_file())
            self.assertTrue((checkpoint / "scheduler.pt").is_file())
            self.assertTrue(smoke["passed"])
            self.assertIn("train_loss", metrics)
            self.assertEqual(resumed.state.global_step, 3)

    def test_step_zero_and_candidate_share_one_held_out_support(self) -> None:
        accelerator = Accelerator(cpu=True)
        config = {
            "model": {"sequence_length": 192},
            "world": {**WORLD, "validation_seed": 808, "rollout_seed": 909},
            "run": {"per_device_batch_size": 2, "eval_batches": 1},
            "preflight": {"rollout_episodes_per_rank": 4},
        }
        set_seed(11)
        step_zero = held_out_learner_evaluation(
            accelerator,
            PretrainingForTrajectoryPrediction(tiny_config()).to(accelerator.device),
            config,
            rollout_episodes_per_rank=4,
        )
        set_seed(12)
        candidate = held_out_learner_evaluation(
            accelerator,
            PretrainingForTrajectoryPrediction(tiny_config()).to(accelerator.device),
            config,
            rollout_episodes_per_rank=4,
        )
        for part in ("validation", "closed_loop"):
            self.assertIn(part, step_zero)
            self.assertIn(part, candidate)
        # Identical starting conditions prove both learners met the same episodes.
        self.assertEqual(
            step_zero["closed_loop"]["initial_error_distribution"],
            candidate["closed_loop"]["initial_error_distribution"],
        )
        self.assertEqual(
            step_zero["closed_loop"]["episodes"], candidate["closed_loop"]["episodes"]
        )
        accelerator.free_memory()

        delta = paired_learning_delta(step_zero, candidate)
        self.assertEqual(delta["episodes"], 4)
        self.assertAlmostEqual(
            delta["terminal_error"]["improvement"],
            step_zero["closed_loop"]["terminal_error"]
            - candidate["closed_loop"]["terminal_error"],
            places=6,
        )
        self.assertAlmostEqual(
            delta["teacher_forced_action_l1"]["improvement"],
            step_zero["validation"]["action_l1"] - candidate["validation"]["action_l1"],
            places=6,
        )
        self.assertIn("0.05", delta["threshold_success"])

    def test_no_trainable_parameter_is_left_without_gradient(self) -> None:
        """A dead trainable parameter deadlocks DDP on its second step.

        Single-process CPU training cannot expose this, so assert it directly
        rather than discovering it on a two-GPU run.
        """

        torch.manual_seed(6)
        model = PretrainingForTrajectoryPrediction(
            PretrainingConfig.from_project_json(SELECTED_CONFIG)
        )
        batch, _ = generate_torch_batch(
            seed=33001,
            start_index=0,
            batch_size=2,
            max_tokens=192,
            world=WORLD,
        )
        model(**batch).loss.backward()
        starved = sorted(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        )
        self.assertEqual(starved, [], f"these parameters would stall DDP: {starved}")
        # The modality-free core has no token vocabulary to train.
        self.assertFalse(model.backbone.embed_tokens.weight.requires_grad)
        self.assertTrue(torch.equal(
            model.backbone.embed_tokens.weight,
            torch.zeros_like(model.backbone.embed_tokens.weight),
        ))

    def test_paired_delta_rejects_mismatched_support(self) -> None:
        left = {
            "closed_loop": {"episodes": 4},
            "validation": {"action_l1": 0.0, "future_l1": 0.0},
        }
        right = {
            "closed_loop": {"episodes": 8},
            "validation": {"action_l1": 0.0, "future_l1": 0.0},
        }
        with self.assertRaises(ValueError):
            paired_learning_delta(left, right)


if __name__ == "__main__":
    unittest.main()
