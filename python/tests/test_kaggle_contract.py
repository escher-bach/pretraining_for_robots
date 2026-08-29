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


if __name__ == "__main__":
    unittest.main()
