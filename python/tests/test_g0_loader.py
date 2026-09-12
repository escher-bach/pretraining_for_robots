from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from pretraining_experiments.g0_loader import CORPUS_MODEL_FIELDS

from pretraining_experiments.data import (
    collate_g0_episodes,
    g0_corpus_loader,
    load_g0_corpus,
)


def _token(event: int = 0) -> dict[str, object]:
    return {
        "role": 6,
        "key": 3,
        "event": event,
        "payload": [0.0] * 8,
        "action_target": [0.0] * 16,
        "action_mask": [False] * 16,
        "future_target": 0.0,
        "future_mask": False,
    }


def _artifact() -> dict[str, object]:
    return {
        "schema_version": 2,
        "train_spec": {"seed": 1},
        "held_out_spec": {"seed": 2},
        "examples": [
            {
                "split": "Train",
                "semantic_hash": "train-world",
                "generator_seed": 1,
                "generator_index": 0,
                "fingerprint": 10,
                "plan_length": 3,
                "episode": {},
                "tokens": [_token(4), _token(5)],
            },
            {
                "split": "HeldOut",
                "semantic_hash": "held-world",
                "generator_seed": 2,
                "generator_index": 0,
                "fingerprint": 20,
                "plan_length": 1,
                "episode": {},
                "tokens": [_token(8)],
            },
        ],
    }


class G0LoaderTests(unittest.TestCase):
    def test_split_selection_and_projection_hide_metadata(self) -> None:
        train = load_g0_corpus(_artifact(), split="train")
        held_out = load_g0_corpus(_artifact(), split="held_out")
        self.assertEqual(len(train), 1)
        self.assertEqual(len(held_out), 1)
        item = train[0]
        self.assertEqual(set(item), set(CORPUS_MODEL_FIELDS))
        self.assertEqual(tuple(item["payloads"].shape), (2, 8))
        self.assertEqual(item["position_ids"].tolist(), [4, 5])
        self.assertNotIn("semantic_hash", item)
        self.assertEqual(train.metadata(0)["semantic_hash"], "train-world")
        self.assertEqual(train.metadata(0)["plan_length"], 3)
        self.assertNotIn("plan_length", item)

    def test_batch_padding_preserves_episode_boundaries(self) -> None:
        train = load_g0_corpus(_artifact())
        batch = collate_g0_episodes([train[0], load_g0_corpus(_artifact(), split="held_out")[0]])
        self.assertEqual(tuple(batch["role_ids"].shape), (2, 2))
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1], [1, 0]])
        self.assertEqual(batch["action_target_mask"].dtype, torch.float32)
        self.assertNotIn("lengths", batch)
        self.assertNotIn("generator_seed", batch)

    def test_standard_dataloader_uses_the_same_model_batch(self) -> None:
        loader = g0_corpus_loader(_artifact(), batch_size=1, max_tokens=3)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["role_ids"].shape), (1, 3))
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 0]])

    def test_path_input_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            path.write_text(json.dumps(_artifact()), encoding="utf-8")
            self.assertEqual(len(load_g0_corpus(path, split="held-out")), 1)

    def test_invalid_artifact_values_are_rejected(self) -> None:
        cases = []
        invalid = _artifact()
        invalid["schema_version"] = 1
        cases.append(invalid)
        invalid = _artifact()
        del invalid["examples"][0]["plan_length"]
        cases.append(invalid)
        invalid = _artifact()
        invalid["examples"][0]["plan_length"] = -1
        cases.append(invalid)
        invalid = _artifact()
        invalid["examples"][0]["plan_length"] = True
        cases.append(invalid)
        invalid = _artifact()
        invalid["examples"][0]["plan_length"] = 1.5
        cases.append(invalid)
        invalid = _artifact()
        invalid["examples"][0]["tokens"][0]["payload"] = [0.0] * 7
        cases.append(invalid)
        invalid = _artifact()
        invalid["examples"][0]["tokens"][0]["future_target"] = float("nan")
        cases.append(invalid)
        invalid = _artifact()
        invalid["examples"][0]["tokens"][0]["role"] = 11
        cases.append(invalid)
        invalid = _artifact()
        invalid["examples"][0]["tokens"][0]["action_mask"][0] = 2
        cases.append(invalid)
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    load_g0_corpus(case)

    def test_plan_length_is_metadata_only(self) -> None:
        artifact = _artifact()
        dataset = load_g0_corpus(artifact)
        self.assertEqual(dataset.metadata(0)["plan_length"], 3)
        batch = next(iter(g0_corpus_loader(artifact, batch_size=1)))
        self.assertEqual(set(batch), set(CORPUS_MODEL_FIELDS))
        self.assertNotIn("plan_length", batch)

    def test_loading_does_not_mutate_artifact(self) -> None:
        artifact = _artifact()
        original = copy.deepcopy(artifact)
        load_g0_corpus(artifact)
        self.assertEqual(artifact, original)

    def test_decision_groups_reach_model_loss_with_tied_actions_and_padding(self) -> None:
        from pretraining_experiments.model import PretrainingConfig, PretrainingForTrajectoryPrediction

        artifact = _artifact()
        tokens = [_token(1)]
        for event, targets in ((2, [1, -1]), (4, [1, 1])):
            for key, target in enumerate(targets):
                token = _token(event)
                token.update(role=7, key=key)
                token["action_target"][0] = target
                token["action_mask"][0] = True
                tokens.append(token)
        artifact["examples"][0]["tokens"] = tokens
        batch = collate_g0_episodes(
            [load_g0_corpus(artifact)[0], load_g0_corpus(artifact, split="held_out")[0]],
            max_tokens=7,
        )
        self.assertEqual(batch["action_decision_groups"].tolist(),
                         [[-1, 2, 2, 4, 4, -1, -1], [-1] * 7])
        model = PretrainingForTrajectoryPrediction(PretrainingConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=1, attention_heads=2,
        )).eval()
        output = model(**batch)
        logits = output.action_logits[0, :, 0]
        expected = (F.cross_entropy(logits[1:3], torch.tensor([1.0, 0.0]))
                    + F.cross_entropy(logits[3:5], torch.tensor([0.5, 0.5]))) / 2
        torch.testing.assert_close(output.action_loss, expected)
        output.loss.backward()
        self.assertTrue(torch.isfinite(model.action_head.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
