from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def load_control_plane():
    path = ROOT / "tools" / "kaggle_run.py"
    spec = importlib.util.spec_from_file_location("pretraining_kaggle_control", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KaggleContractTests(unittest.TestCase):
    def test_push_receipt_parser_preserves_exact_version(self) -> None:
        control = load_control_plane()
        kernel = "aniruddhavarma/pretraining-single-world-deadbee"
        output = (
            "Kernel version 3 successfully pushed.  Please check progress at "
            f"https://www.kaggle.com/code/{kernel}"
        )
        self.assertEqual(
            control.parse_push_result(output, kernel),
            (3, f"https://www.kaggle.com/code/{kernel}"),
        )

    def test_downloaded_artifacts_are_checked_against_remote_manifest(self) -> None:
        control = load_control_plane()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "training-result.json"
            artifact.write_text('{"ok":true}', encoding="utf-8")
            manifest = {
                "artifacts": [
                    {
                        "path": artifact.name,
                        "size": artifact.stat().st_size,
                        "sha256": control.sha256_file(artifact),
                    }
                ]
            }
            self.assertEqual(
                control.verify_downloaded_against_manifest(root, manifest),
                [artifact.name],
            )

    def test_generated_notebook_is_a_thin_exact_sha_launcher(self) -> None:
        control = load_control_plane()
        sha = "1" * 40
        notebook = control.notebook(
            "https://github.com/example/repository.git",
            sha,
            "configs/kaggle/t4x2_single_world.toml",
            "/kaggle/working/pretraining-results",
        )
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertEqual(sum(cell["cell_type"] == "code" for cell in notebook["cells"]), 3)
        self.assertIn(sha, code)
        self.assertEqual(code.count("pretraining_experiments.runner"), 1)
        self.assertNotIn("class PretrainingForTrajectoryPrediction", code)
        self.assertNotIn("generate_training_batch", code)
        self.assertNotIn("torch.optim", code)

    def test_registry_paths_exist(self) -> None:
        control = load_control_plane()
        data, experiment = control.experiment("single-world-apparatus")
        self.assertEqual(data["accelerator"], "NvidiaTeslaT4")
        self.assertTrue((ROOT / experiment["config"]).is_file())
        self.assertTrue((ROOT / "requirements-kaggle.txt").is_file())
        toolchain = (ROOT / "rust-toolchain.toml").read_text(encoding="utf-8")
        self.assertIn('channel = "1.88.0"', toolchain)

    def test_r10_registry_uses_the_fixed_one_t4_seed_gate_contract(self) -> None:
        control = load_control_plane()
        data, experiment = control.experiment("r10-seed-gate")
        self.assertEqual(data["accelerator"], "NvidiaTeslaT4")
        config_path = ROOT / experiment["config"]
        self.assertTrue(config_path.is_file())
        config = __import__("tomllib").loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["run"]["entrypoint"], "seed_gate")
        self.assertEqual(config["run"]["device"], "cuda")
        self.assertEqual(config["seed_gate"]["family_order"], ["card04", "card03", "card02", "card05", "card06"])
        self.assertEqual(config["seed_gate"]["total_timeout_seconds"], 720)

    def test_r10_grouped_repair_uses_a_distinct_registry_entry_and_contract(self) -> None:
        control = load_control_plane()
        _, experiment = control.experiment("r10-seed-gate-grouped")
        self.assertEqual(experiment["config"], "configs/r10/seed_gate_t4_grouped.toml")
        config_path = ROOT / experiment["config"]
        config = __import__("tomllib").loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["run"]["entrypoint"], "seed_gate")
        self.assertEqual(config["seed_gate"]["action_query_objective"], "grouped_action_query_cross_entropy")
        self.assertEqual(config["seed_gate"]["apparatus_repair"], "r10-grouped-action-query-objective-v1")


if __name__ == "__main__":
    unittest.main()
