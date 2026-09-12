import unittest
from collections import Counter
from itertools import product

import torch

from pretraining_experiments.trajectory_data import (
    CalibratedReachDataset,
    TrajectoryDatasetConfig,
    collate_trajectory_batch,
    first_system_adapter_specs,
)


class TrajectoryDataTests(unittest.TestCase):
    def test_fixed_mixture_covers_independent_factorial_cells(self) -> None:
        config = TrajectoryDatasetConfig(
            size=64,
            seed=31,
            goal_switch_steps=(-1, 3),
            disturbance_arms=(False, True),
        )
        dataset = CalibratedReachDataset(config, first_system_adapter_specs())
        expected = tuple(
            product(config.sensor_dims, config.action_dims, config.goal_modes, config.goal_switch_steps, config.disturbance_arms)
        )
        self.assertEqual(len(expected), 32)
        self.assertEqual(len(dataset.cells), 32)
        self.assertEqual(set(dataset.cells), set(expected))
        self.assertEqual(Counter(dataset.cells[index % len(dataset.cells)] for index in range(len(dataset))), Counter({cell: 2 for cell in expected}))

    def test_public_tensor_dataset_is_deterministic_and_keeps_action_validity(self) -> None:
        config = TrajectoryDatasetConfig(size=4, seed=21)
        dataset = CalibratedReachDataset(config, first_system_adapter_specs())
        left = dataset[0]
        right = dataset[0]
        for name in left:
            self.assertTrue(torch.equal(left[name], right[name]), name)
        self.assertEqual(tuple(left["raw_values"].shape[-1:]), (10,))
        self.assertEqual(left["available_at"].shape, left["time_values"].shape)
        self.assertTrue(torch.all(left["available_at"] >= left["time_values"]))
        self.assertEqual(int(left["action_target_mask"].sum()), 12 * 2)
        # Action queries are supervision-only addresses; event routing remains
        # a modality code consumed by the model's adapter registry.
        self.assertGreater(int(left["action_mask"].sum()), 0)

    def test_collation_pads_variable_calibration_lengths_without_new_labels(self) -> None:
        dataset = CalibratedReachDataset(
            TrajectoryDatasetConfig(size=4, seed=22), first_system_adapter_specs()
        )
        batch = collate_trajectory_batch([dataset[0], dataset[1]])
        self.assertEqual(batch["raw_values"].ndim, 3)
        self.assertEqual(batch["raw_values"].shape[-1], 10)
        lengths = [int(dataset[index]["attention_mask"].sum()) for index in (0, 1)]
        self.assertTrue(lengths[0] <= batch["attention_mask"].shape[1])
        self.assertTrue(torch.equal(batch["action_target_mask"][0, lengths[0] :], torch.zeros_like(batch["action_target_mask"][0, lengths[0] :])))
        self.assertTrue(torch.equal(batch["attention_mask"][1, lengths[1] :], torch.zeros_like(batch["attention_mask"][1, lengths[1] :])))


if __name__ == "__main__":
    unittest.main()
