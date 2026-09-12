import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import torch

from pretraining_experiments.first_training import (
    FirstTrainingConfig,
    _resolve_device,
    build_first_system,
    load_first_training_config,
    run_cpu_smoke,
)
from pretraining_experiments.trajectory_data import (
    CalibratedReachDataset,
    TrajectoryDatasetConfig,
    first_system_adapter_specs,
)


class FirstTrainingTests(unittest.TestCase):
    def test_execution_target_is_explicit_and_cpu_profile_does_not_follow_host_cuda(self) -> None:
        from pretraining_experiments import first_training

        config = FirstTrainingConfig(device="cpu", mixed_precision="fp32")
        with patch.object(first_training.torch.cuda, "is_available", return_value=True):
            self.assertEqual(_resolve_device(config), torch.device("cpu"))

    def test_gpu_profile_fails_closed_without_cuda(self) -> None:
        from pretraining_experiments import first_training

        config = FirstTrainingConfig(device="cuda", mixed_precision="fp16")
        with patch.object(first_training.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                _resolve_device(config)

    def test_review_config_loads_bounded_run_fields(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = load_first_training_config(root / "configs" / "first_training_system_cpu.toml")
        self.assertEqual(config.seed, 17)
        self.assertEqual(config.dataset_size, 8)
        self.assertEqual(config.smoke_steps, 2)
        self.assertEqual(config.resumed_steps, 3)
        self.assertEqual(config.world.horizon, 12)
        self.assertEqual(config.goal_switch_steps, (-1, 3))
        self.assertEqual(config.disturbance_arms, (False, True))

    def test_run_purpose_and_effective_config_are_serializable(self) -> None:
        import dataclasses

        root = Path(__file__).resolve().parents[2]
        config = load_first_training_config(root / "configs" / "first_training_system_scientific_smoke.toml")
        self.assertEqual(config.purpose, "apparatus_verification")
        json.dumps(dataclasses.asdict(config))

    def test_world_fields_are_consumed_by_dataset_generation(self) -> None:
        root = Path(__file__).resolve().parents[2]
        base = (root / "configs" / "first_training_system_cpu.toml").read_text(encoding="utf-8")
        mutated = base.replace("horizon = 12", "horizon = 4", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.toml"
            path.write_text(mutated, encoding="utf-8")
            config = load_first_training_config(path)
        self.assertEqual(config.world.horizon, 4)
        dataset = CalibratedReachDataset(
            TrajectoryDatasetConfig(
                size=1,
                seed=config.seed,
                world=config.world,
                sensor_dims=config.sensor_dims,
                action_dims=config.action_dims,
                goal_modes=config.goal_modes,
                goal_switch_steps=config.goal_switch_steps,
                disturbance_arms=config.disturbance_arms,
            ),
            first_system_adapter_specs(),
        )
        self.assertEqual(int(dataset[0]["action_mask"].sum()), 4)

    def test_cli_contract_selects_scientific_budget_without_launching(self) -> None:
        from pretraining_experiments import first_training

        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(first_training, "run_scientific", return_value={"mode": "scientific"}) as run:
                status = first_training.main(
                    [
                        "--config",
                        str(root / "configs" / "first_training_system_gpu_proposal.toml"),
                        "--output-root",
                        directory,
                    ]
                )
            self.assertEqual(status, 0)
            config = run.call_args.args[1]
            self.assertEqual(config.mode, "scientific")
            self.assertEqual(config.resumed_steps, 20)
            self.assertEqual(config.evaluation_checkpoints, (0, 5, 10, 20))
            self.assertEqual(config.device, "cuda")
            self.assertEqual(config.world.sensor_dims, (8, 10))
            self.assertEqual(config.world.action_dims, (2, 4))
            self.assertEqual(config.world.disturbance_scale, 0.02)
            self.assertEqual(config.transform_sensor_permutations, ())
            self.assertEqual(config.transform_body_permutations, ())

    def test_declared_sensor_width_six_reaches_forward_and_gradient(self) -> None:
        config = FirstTrainingConfig(
            sensor_dims=(6,), action_dims=(2,), goal_modes=("absolute",),
            goal_switch_steps=(-1,), disturbance_arms=(False,),
            hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            attention_heads=2, max_position_embeddings=16,
        )
        model = build_first_system(config)
        raw = torch.zeros(1, 2, 6)
        raw[0, 0, :] = torch.arange(6, dtype=torch.float32)
        raw[0, 1, 0] = 1.0
        output = model(
            raw_values=raw,
            adapter_ids=torch.tensor([[0, 1]]),
            role_ids=torch.tensor([[2, 8]]),
            key_ids=torch.tensor([[1, 2]]),
            time_values=torch.tensor([[0.0, 0.1]]),
            available_at=torch.tensor([[0.0, 0.1]]),
            position_ids=torch.tensor([[0, 1]]),
            attention_mask=torch.ones(1, 2, dtype=torch.bool),
            action_mask=torch.ones(1, 2, dtype=torch.bool),
            action_targets=torch.zeros(1, 2, 4),
            action_target_mask=torch.ones(1, 2, 4, dtype=torch.bool),
        )
        self.assertEqual(tuple(output.actions.shape), (1, 2, 4))
        output.loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_cpu_smoke_trains_saves_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = run_cpu_smoke(
                Path(directory),
                FirstTrainingConfig(dataset_size=4, smoke_steps=2, resumed_steps=3),
            )
        self.assertTrue(receipt["apparatus_only"])
        self.assertEqual(receipt["first_steps"], 2)
        self.assertEqual(receipt["resumed_steps"], 3)
        self.assertTrue(receipt["continuation_match"])
        self.assertTrue(receipt["optimizer_match"])
        self.assertTrue(receipt["scheduler_match"])
        self.assertTrue(receipt["data_order_match"])


if __name__ == "__main__":
    unittest.main()
