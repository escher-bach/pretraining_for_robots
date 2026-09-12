from __future__ import annotations

import unittest

import torch

from pretraining_experiments.common_content import CommonContentBundle, CommonContentConfig
from pretraining_experiments.trajectory_data import (
    CalibratedReachDataset,
    TrajectoryDatasetConfig,
    collate_trajectory_batch,
    first_system_action_spec,
    first_system_adapter_specs,
)


class GeneratedContentBoundaryTests(unittest.TestCase):
    """Exercise the shared boundary with real generated trajectory records."""

    def test_generated_width_cross_product_reaches_every_adapter_and_gradient(self) -> None:
        torch.set_num_threads(1)
        config = TrajectoryDatasetConfig(
            size=32,
            seed=20260908,
            sensor_dims=(8, 10),
            action_dims=(2, 4),
            goal_modes=("absolute", "relative"),
            goal_switch_steps=(-1, 3),
            disturbance_arms=(False, True),
        )
        dataset = CalibratedReachDataset(config, first_system_adapter_specs())
        selected_cells = (
            (8, 2, "absolute", -1, False),
            (8, 4, "absolute", -1, False),
            (10, 2, "relative", 3, True),
            (10, 4, "relative", 3, True),
        )
        selected = [dataset[dataset.cells.index(cell)] for cell in selected_cells]
        batch = collate_trajectory_batch(selected)
        model = CommonContentBundle.from_parts(
            CommonContentConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                attention_heads=2,
                max_position_embeddings=128,
            ),
            first_system_adapter_specs(),
            first_system_action_spec(),
        )
        model.train()
        output = model(**batch)
        self.assertIsNotNone(output.loss)
        self.assertTrue(torch.isfinite(output.loss))

        # Labels are loss-only.  They must not alter generated content or the
        # temporal hidden states supplied to the action decoder.
        feature_batch = {
            name: value
            for name, value in batch.items()
            if name not in {"action_targets", "action_target_mask"}
        }
        with torch.no_grad():
            without_labels_output = model(**feature_batch)
            without_labels = without_labels_output.last_hidden_state
            with_labels = output.last_hidden_state
        torch.testing.assert_close(without_labels, with_labels, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            without_labels_output.actions,
            output.actions,
            atol=1e-6,
            rtol=1e-6,
        )

        output.loss.backward()
        for name in (
            "sensor8",
            "sensor10",
            "goal8",
            "goal10",
            "action2",
            "action4",
            "control1",
        ):
            gradient = model.adapters.adapters[name].projector.weight.grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.abs().sum()), 0.0, name)
        core_gradient = model.core.backbone.layers[0].self_attn.q_proj.weight.grad
        decoder_gradient = model.decoder.head[-1].weight.grad
        self.assertIsNotNone(core_gradient)
        self.assertIsNotNone(decoder_gradient)
        self.assertGreater(float(core_gradient.abs().sum()), 0.0)
        self.assertGreater(float(decoder_gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
