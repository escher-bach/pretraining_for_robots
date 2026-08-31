from __future__ import annotations

from pathlib import Path
import unittest

import torch

from pretraining_experiments.data import MODEL_FIELDS, generate_g0_mixed_torch_batch, g0_corpus_manifest
from pretraining_experiments.seed_gate import (
    CONTRACT_HASHES,
    CARD06_SCALE_PROFILE,
    card06_scale_decision,
    classify_card06_scale_pilot,
    classify_seed_gate_pilot,
    consumed_training_cost,
    evaluate_g0_corpus,
    grouped_action_decision_argmax,
    load_seed_gate_config,
    validate_seed_gate_config,
)
from pretraining_experiments.model import PretrainingOutput


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
    def test_grouped_evaluator_ranks_raw_fp16_logits_not_saturated_tanh_predictions(self) -> None:
        raw_logits = torch.zeros(1, 4, 16, dtype=torch.float16)
        raw_logits[0, 0, 0], raw_logits[0, 1, 0] = 8.0, 10.0
        self.assertEqual(float(torch.tanh(raw_logits[0, 0, 0])), float(torch.tanh(raw_logits[0, 1, 0])))
        fields = synthetic_fields(tokens=4)
        fields["action_targets"][0][0][0] = 1.0
        fields["action_target_mask"][0][0][0] = 1.0
        fields["action_target_mask"][0][1][0] = 1.0
        fields.update({
            "families": ["card04"],
            "case_kinds": [["witness_goal_conditioning"]],
            "primary_case_kinds": ["witness_goal_conditioning"],
            "action_decision_groups": [[0, 0, -1, -1]],
        })

        class SaturatingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, **_kwargs):
                return PretrainingOutput(
                    action_logits=raw_logits.to(self.anchor.device),
                    action_predictions=torch.tanh(raw_logits).to(self.anchor.device),
                )

        result = evaluate_g0_corpus(SaturatingModel(), fields)
        # The larger rejected raw logit wins.  Ranking tanh values would tie at
        # one and incorrectly select the first (positive) row instead.
        self.assertEqual(result["macro_argmax"], 0.0)

    def test_real_g0_manifests_have_complete_group_addresses_and_multi_positive_targets(self) -> None:
        multi_positive_families: set[str] = set()
        for family in ("card04", "card03", "card02", "card05", "card06"):
            manifest = g0_corpus_manifest(families=[family], max_tokens=192)
            for targets, mask, groups in zip(
                manifest["action_targets"],
                manifest["action_target_mask"],
                manifest["action_decision_groups"],
            ):
                decisions: dict[int, int] = {}
                for target, target_mask, group in zip(targets, mask, groups):
                    if target_mask[0]:
                        self.assertGreaterEqual(group, 0)
                        decisions[group] = decisions.get(group, 0) + int(target[0] > 0.0)
                    else:
                        self.assertLess(group, 0)
                if any(count > 1 for count in decisions.values()):
                    multi_positive_families.add(family)
        self.assertTrue({"card03", "card02", "card05", "card06"}.issubset(multi_positive_families))

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

    def test_versioned_grouped_repair_preserves_all_scientific_settings(self) -> None:
        original = load_seed_gate_config(ROOT / "configs" / "r10" / "seed_gate_t4.toml")
        repaired = load_seed_gate_config(ROOT / "configs" / "r10" / "seed_gate_t4_grouped.toml")
        self.assertEqual(
            {key: original["seed_gate"][key] for key in (
                "root_seed", "family_order", "evaluation_steps", "per_family_timeout_seconds",
                "total_timeout_seconds", "timing_updates", "timing_updates_max_seconds",
                "timing_eval_max_seconds", "required_macro_argmax", "required_improvement",
                "required_primary_case_kind_argmax", "distinct_episode_counts", "contract_hashes",
                "family_seeds",
            )},
            {key: repaired["seed_gate"][key] for key in (
                "root_seed", "family_order", "evaluation_steps", "per_family_timeout_seconds",
                "total_timeout_seconds", "timing_updates", "timing_updates_max_seconds",
                "timing_eval_max_seconds", "required_macro_argmax", "required_improvement",
                "required_primary_case_kind_argmax", "distinct_episode_counts", "contract_hashes",
                "family_seeds",
            )},
        )
        self.assertEqual(
            {key: original["run"][key] for key in original["run"] if key not in {"name", "purpose", "checkpoint_label"}},
            {key: repaired["run"][key] for key in repaired["run"] if key not in {"name", "purpose", "checkpoint_label"}},
        )
        self.assertEqual(original["model"], repaired["model"])
        self.assertEqual(repaired["seed_gate"]["action_query_objective"], "grouped_action_query_cross_entropy")
        self.assertEqual(repaired["seed_gate"]["apparatus_repair"], "r10-grouped-action-query-objective-v1")

    def test_card06_scale_profile_is_a_separate_fixed_contract(self) -> None:
        config = load_seed_gate_config(
            ROOT / "configs" / "r10" / "card06_compatibility_scale_t4.toml"
        )
        gate = config["scale_diagnostic"]
        self.assertNotIn("seed_gate", config)
        self.assertEqual(gate["profile"], CARD06_SCALE_PROFILE)
        self.assertEqual(gate["family"], "card06")
        self.assertEqual(gate["contract_hash"], CONTRACT_HASHES["card06"])
        self.assertEqual(gate["evaluation_steps"], [0, 64, 128, 256])
        self.assertEqual(gate["schedule_horizon_updates"], 256)
        self.assertFalse(gate["r10_replication"])
        self.assertEqual(gate["decision_rungs"], [64, 128, 256])
        self.assertEqual(
            gate["stable_exact_action"],
            "support-compatible; eligible only for a separately declared Card06 generalization profile; no R10 admission or R11 claim",
        )
        self.assertEqual(config["run"]["max_updates"], 256)
        self.assertEqual(config["run"]["seed"], 20260829)
        self.assertEqual(config["run"]["entrypoint"], "seed_gate")
        self.assertEqual(config["run"]["max_grad_norm"], 1.0)
        self.assertEqual(config["run"]["log_every"], 16)
        self.assertEqual(config["run"]["seed_gate_phase_timeout_seconds"], 600)
        self.assertEqual(config["run"]["max_wall_clock_seconds"], 3600)
        self.assertEqual(config["model"]["config"], "artifacts/icrt-derived-small/model_config.json")

    def test_card06_scale_profile_accepts_runner_resolved_model_config_path(self) -> None:
        config = load_seed_gate_config(
            ROOT / "configs" / "r10" / "card06_compatibility_scale_t4.toml"
        )
        config["model"]["config"] = str(
            (ROOT / config["model"]["config"]).resolve()
        )
        validate_seed_gate_config(config)

    def test_card06_scale_profile_keeps_non_path_execution_drift_strict(self) -> None:
        config = load_seed_gate_config(
            ROOT / "configs" / "r10" / "card06_compatibility_scale_t4.toml"
        )
        config["model"]["config"] = str(
            (ROOT / config["model"]["config"]).resolve()
        )
        config["run"]["max_updates"] = 128
        with self.assertRaisesRegex(ValueError, "execution drift"):
            validate_seed_gate_config(config)

    def test_card06_scale_classification_requires_exact_full_support_fit(self) -> None:
        result = {
            "complete": True, "finite": True, "abi_and_bounds_ok": True, "timed_out": False,
            "evaluations": {
                step: {
                    "macro_argmax": 1.0,
                    "by_case_kind": {"agent_equivalence": {"argmax": 1.0}},
                }
                for step in (64, 128, 256)
            },
        }
        self.assertEqual(
            classify_card06_scale_pilot(result=result, max_updates=256), "exact_support_fit"
        )
        decision = card06_scale_decision(result=result, max_updates=256)
        self.assertEqual(decision["exact_fit_by_step"], {64: True, 128: True, 256: True})
        self.assertEqual(decision["earliest_exact_fit_step"], 64)
        result["evaluations"][128]["by_case_kind"]["agent_equivalence"]["argmax"] = 0.99
        self.assertEqual(
            classify_card06_scale_pilot(result=result, max_updates=256), "unstable_support_fit"
        )
        result["evaluations"][64]["by_case_kind"]["agent_equivalence"]["argmax"] = 0.99
        result["evaluations"][256]["by_case_kind"]["agent_equivalence"]["argmax"] = 0.99
        self.assertEqual(
            classify_card06_scale_pilot(result=result, max_updates=256), "support_fit_incomplete"
        )

    def test_completed_step_cost_excludes_iterable_prefetch(self) -> None:
        from pretraining_experiments import seed_gate

        observed: dict[str, object] = {}

        def fake(**kwargs):
            observed.update(kwargs)
            targets = torch.zeros(16, 4, 16)
            mask = torch.zeros_like(targets)
            mask[:, :3, 0] = 1.0
            fields = synthetic_fields(batch=16, tokens=4)
            fields["action_targets"] = targets.tolist()
            fields["action_target_mask"] = mask.tolist()
            return {
                name: torch.tensor(value) if name not in {"role_ids", "key_ids", "position_ids", "attention_mask"} else torch.tensor(value, dtype=torch.long)
                for name, value in fields.items()
            }, {}

        previous = seed_gate.generate_g0_mixed_torch_batch
        seed_gate.generate_g0_mixed_torch_batch = fake
        try:
            cost = consumed_training_cost(
                family="card04", seed=19, max_tokens=4, completed_updates=2,
                per_device_batch_size=4, gradient_accumulation_steps=2, world_size=1,
            )
        finally:
            seed_gate.generate_g0_mixed_torch_batch = previous
        self.assertEqual(cost, {"episode_presentations": 16, "scored_action_query_targets": 48})
        self.assertEqual(observed["start_index"], 0)
        self.assertEqual(observed["batch_size"], 16)

    def test_completed_step_cost_rejects_multi_process_reconstruction(self) -> None:
        with self.assertRaisesRegex(ValueError, r"world_size=1.*world_size=2"):
            consumed_training_cost(
                family="card04", seed=19, max_tokens=4, completed_updates=2,
                per_device_batch_size=4, gradient_accumulation_steps=2, world_size=2,
            )

    def test_group_addresses_are_added_only_to_the_loss_adapter_dataset(self) -> None:
        from pretraining_experiments import seed_gate

        raw = synthetic_fields(tokens=4)
        tensors = {
            name: torch.tensor(
                value,
                dtype=torch.long if name in {"role_ids", "key_ids", "position_ids", "attention_mask"} else torch.float32,
            )
            for name, value in raw.items()
        }

        def fake(**_kwargs):
            return tensors, {"action_decision_groups": [[-1, 4, 4, -1]]}

        previous = seed_gate.generate_g0_mixed_torch_batch
        seed_gate.generate_g0_mixed_torch_batch = fake
        try:
            legacy = next(iter(seed_gate.DistinctEpisodeDataset("card04", 1, 4)))
            grouped = next(iter(seed_gate.DistinctEpisodeDataset(
                "card04", 1, 4, include_action_decision_groups=True
            )))
        finally:
            seed_gate.generate_g0_mixed_torch_batch = previous
        self.assertEqual(set(legacy), set(MODEL_FIELDS))
        self.assertEqual(set(grouped), set(MODEL_FIELDS) | {"action_decision_groups"})
        self.assertNotIn("action_decision_groups", MODEL_FIELDS)
        self.assertTrue(torch.equal(grouped["action_decision_groups"], torch.tensor([-1, 4, 4, -1])))

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
