from __future__ import annotations

from pathlib import Path
import unittest

import torch

from pretraining_experiments.data import MODEL_FIELDS, generate_g0_mixed_torch_batch
from pretraining_experiments.seed_gate import (
    CONTRACT_HASHES,
    classify_seed_gate_pilot,
    grouped_action_decision_argmax,
    load_seed_gate_config,
)


ROOT = Path(__file__).resolve().parents[2]


def synthetic_fields(batch: int = 1, tokens: int = 4) -> dict[str, object]:
    fields: dict[str, object] = {
        "role_ids": [[0] * tokens for _ in range(batch)],
        "key_ids": [[0] * tokens for _ in range(batch)],
        "position_ids": [[0] * tokens for _ in range(batch)],
        "payloads": [[[0.0] * 8 for _ in range(tokens)] for _ in range(batch)],
        "attention_mask": [[1] * tokens for _ in range(batch)],
        "action_targets": [[[0.0] * 16 for _ in range(tokens)] for _ in range(batch)],
        "action_target_mask": [[[0.0] * 16 for _ in range(tokens)] for _ in range(batch)],
        "future_targets": [[0.0] * tokens for _ in range(batch)],
        "future_target_mask": [[0.0] * tokens for _ in range(batch)],
    }
    return fields


class SeedGateTests(unittest.TestCase):
    def test_config_is_the_predeclared_cpu_contract(self) -> None:
        config = load_seed_gate_config(ROOT / "configs" / "r10" / "seed_gate_cpu.toml")
        self.assertEqual(config["seed_gate"]["contract_hashes"], CONTRACT_HASHES)
        self.assertEqual(config["run"]["device"], "cpu")
        self.assertEqual(config["run"]["max_updates"], 64)

    def test_t4_config_preserves_the_seed_gate_and_selects_cuda_execution(self) -> None:
        config = load_seed_gate_config(ROOT / "configs" / "r10" / "seed_gate_t4.toml")
        self.assertEqual(config["seed_gate"]["contract_hashes"], CONTRACT_HASHES)
        self.assertEqual(config["run"]["device"], "cuda")
        self.assertEqual(config["run"]["mixed_precision"], "fp16")
        self.assertEqual(config["seed_gate"]["per_family_timeout_seconds"], 120)

    def test_grouped_argmax_scores_choices_not_individual_query_rows(self) -> None:
        # In each decision only one ActionQuery is selected.  The first group is
        # right and the second deliberately picks a rejected alternative.
        predictions = torch.zeros(1, 4, 16)
        predictions[0, 0, 0], predictions[0, 1, 0] = 0.1, 0.9
        predictions[0, 2, 0], predictions[0, 3, 0] = 0.1, 0.9
        targets = torch.zeros_like(predictions)
        targets[0, 1, 0], targets[0, 2, 0] = 1.0, 1.0
        targets[0, 0, 0], targets[0, 3, 0] = -1.0, -1.0
        mask = torch.zeros_like(predictions)
        mask[0, :, 0] = 1.0
        result = grouped_action_decision_argmax(
            predictions, targets, mask,
            families=["card04"],
            case_kinds=[["witness_goal_conditioning", "alias_control"]],
            decision_groups=[[0, 0, 1, 1]], primary_case_kinds=["witness_goal_conditioning"],
        )
        self.assertEqual(result["decisions"], 2)
        self.assertEqual(result["macro_argmax"], 0.5)
        self.assertEqual(result["by_primary_case_kind"]["witness_goal_conditioning"]["argmax"], 0.5)
        self.assertEqual(result["by_case_kind"]["alias_control"]["argmax"], 0.5)

    def test_wrapper_keeps_manifest_metadata_out_of_model_inputs(self) -> None:
        from pretraining_experiments import data

        received: dict[str, object] = {}
        raw = synthetic_fields()
        raw.update({"families": ["card04"], "contract_hashes": {"card04": "x"}})

        def fake(**kwargs):
            received.update(kwargs)
            return raw

        previous = getattr(data.pretraining_world_py, "generate_g0_mixed_training_batch", None)
        data.pretraining_world_py.generate_g0_mixed_training_batch = fake
        try:
            tensors, metadata = generate_g0_mixed_torch_batch(
                families=["card04"], weights=[1.0], seed=1, start_index=2,
                batch_size=1, max_tokens=4,
            )
        finally:
            if previous is None:
                delattr(data.pretraining_world_py, "generate_g0_mixed_training_batch")
            else:
                data.pretraining_world_py.generate_g0_mixed_training_batch = previous
        self.assertEqual(set(tensors), set(MODEL_FIELDS))
        self.assertEqual(metadata["families"], ["card04"])
        self.assertEqual(received["families"], ["card04"])
        self.assertEqual(received["weights"], [1])

    def test_classification_has_no_transfer_path(self) -> None:
        gate = {
            "max_updates": 64,
            "required_macro_argmax": 0.80,
            "required_improvement": 0.25,
            "required_primary_case_kind_argmax": 0.60,
        }
        result = {
            "complete": True, "finite": True, "abi_and_bounds_ok": True, "timed_out": False,
            "evaluations": {
                0: {"macro_argmax": 0.50},
                64: {
                    "macro_argmax": 0.80,
                    "by_case_kind": {"witness": {"argmax": 0.60}},
                },
            },
        }
        self.assertEqual(classify_seed_gate_pilot(result=result, gate=gate), "admitted")
        result["timed_out"] = True
        self.assertEqual(classify_seed_gate_pilot(result=result, gate=gate), "unscored")


if __name__ == "__main__":
    unittest.main()
