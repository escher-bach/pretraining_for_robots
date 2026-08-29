"""Thin project control plane around the official Kaggle CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import tomllib
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "kaggle" / "experiments.toml"


def command_output(command: list[str], *, cwd: Path = ROOT) -> str:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout.strip()
    if completed.returncode:
        raise SystemExit(
            f"command failed with exit code {completed.returncode}: {command}\n{output}"
        )
    return output


def command_run(command: list[str], *, cwd: Path = ROOT) -> None:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_push_result(output: str, expected_kernel: str) -> tuple[int, str]:
    match = re.search(r"Kernel version (\d+) successfully pushed\.\s+Please check progress at (\S+)", output)
    if match is None:
        raise SystemExit(f"could not parse Kaggle push receipt:\n{output}")
    version = int(match.group(1))
    url = match.group(2)
    if expected_kernel not in url:
        raise SystemExit(f"Kaggle push URL does not identify {expected_kernel}: {url}")
    return version, url


def exact_kernel_listing(kernel: str) -> dict[str, Any]:
    slug = kernel.split("/", 1)[1]
    output = command_output(
        [
            "kaggle",
            "kernels",
            "list",
            "--mine",
            "--search",
            slug,
            "--format",
            "json",
            "--page-size",
            "100",
        ]
    )
    rows = json.loads(output)
    matches = [row for row in rows if row.get("ref") == kernel]
    if len(matches) != 1:
        raise SystemExit(f"expected one exact Kaggle listing for {kernel}, found {matches}")
    return matches[0]


def verify_downloaded_against_manifest(run_root: Path, manifest: dict[str, Any]) -> list[str]:
    declared = {item["path"]: item for item in manifest.get("artifacts", [])}
    verified: list[str] = []
    excluded = {"audit-manifest.json", "summary.json", "receipt.json"}
    for path in sorted(run_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_root).as_posix()
        if relative in excluded:
            continue
        expected = declared.get(relative)
        if expected is None:
            raise SystemExit(f"downloaded artifact is absent from remote manifest: {relative}")
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        if actual_size != int(expected["size"]) or actual_sha != expected["sha256"]:
            raise SystemExit(
                f"downloaded artifact failed remote-manifest verification: {relative}; "
                f"expected size/hash {expected['size']}/{expected['sha256']}, "
                f"got {actual_size}/{actual_sha}"
            )
        verified.append(relative)
    return verified


def registry() -> dict[str, Any]:
    return tomllib.loads(REGISTRY.read_text(encoding="utf-8"))


def experiment(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    data = registry()
    try:
        selected = data["experiments"][name]
    except KeyError as exc:
        choices = ", ".join(sorted(data.get("experiments", {})))
        raise SystemExit(f"unknown experiment {name!r}; choices: {choices}") from exc
    return data, selected


def exact_head() -> str:
    sha = command_output(["git", "rev-parse", "HEAD"])
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise SystemExit(f"HEAD is not a full Git SHA: {sha}")
    return sha


def verify_remote_sha(remote: str, sha: str) -> str:
    remote_url = command_output(["git", "remote", "get-url", remote])
    refs = command_output(["git", "ls-remote", remote_url])
    if not any(line.startswith(sha + "\t") for line in refs.splitlines()):
        raise SystemExit(
            f"commit {sha} is not the tip of a reachable ref on {remote} ({remote_url}); push it before launch"
        )
    ssh_match = re.fullmatch(r"git@github\.com:([^/]+/.+?)(?:\.git)?", remote_url)
    if ssh_match:
        remote_url = f"https://github.com/{ssh_match.group(1)}.git"
    if not remote_url.startswith(("https://", "http://")):
        raise SystemExit(
            "the Git remote must have a public HTTP(S) clone URL that Kaggle can read"
        )
    return remote_url


def notebook(repo_url: str, sha: str, config: str, output_root: str) -> dict[str, Any]:
    code1 = f'''from pathlib import Path
import os
import subprocess
import sys

REPO_URL = {repo_url!r}
GIT_COMMIT = {sha!r}
CONFIG_REL = {config!r}
RUNTIME = Path("/tmp/pretraining-runtime")
SOURCE = RUNTIME / "source"
OUTPUT = Path({output_root!r})
assert len(GIT_COMMIT) == 40 and all(c in "0123456789abcdef" for c in GIT_COMMIT)
assert not SOURCE.exists(), f"fresh batch session required: {{SOURCE}}"
RUNTIME.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
'''
    code2 = '''env = os.environ.copy()
env.update({
    "GIT_TERMINAL_PROMPT": "0",
    "PYTHONUNBUFFERED": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "WANDB_MODE": "disabled",
})
subprocess.run(["git", "clone", REPO_URL, str(SOURCE)], check=True, env=env)
subprocess.run(["git", "-C", str(SOURCE), "checkout", "--detach", GIT_COMMIT], check=True, env=env)
resolved = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True, env=env).strip()
assert resolved == GIT_COMMIT, (resolved, GIT_COMMIT)
assert (SOURCE / CONFIG_REL).is_file(), SOURCE / CONFIG_REL
'''
    code3 = '''env["PYTHONPATH"] = str(SOURCE / "python")
cmd = [
    sys.executable,
    "-m",
    "pretraining_experiments.runner",
    "--config", str(SOURCE / CONFIG_REL),
    "--output-root", str(OUTPUT),
]
# Outermost guard. The runner arms its own wall-clock watchdog; this one
# covers the case where the runner process itself wedges and cannot enforce
# it. Kaggle would otherwise hold the session until the platform limit.
NOTEBOOK_TIMEOUT_SECONDS = 6000
try:
    completed = subprocess.run(cmd, cwd=str(SOURCE), env=env, check=False, timeout=NOTEBOOK_TIMEOUT_SECONDS)
except subprocess.TimeoutExpired:
    raise RuntimeError(f"pretraining runner exceeded the {{NOTEBOOK_TIMEOUT_SECONDS}}s notebook budget and was killed; partial evidence is retained under {{OUTPUT}}")
if completed.returncode != 0:
    raise RuntimeError(f"pretraining runner failed with exit code {{completed.returncode}}; evidence is retained under {{OUTPUT}}")
'''
    def cell(source: str) -> dict[str, Any]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.splitlines()],
        }

    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Robot pretraining two-T4 run\n", "Generated launcher; repository code owns the experiment.\n"],
            },
            cell(code1),
            cell(code2),
            cell(code3),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def launch(name: str) -> str:
    data, selected = experiment(name)
    sha = exact_head()
    config = str(selected["config"])
    command_run(["git", "cat-file", "-e", f"{sha}:{config}"])
    repo_url = verify_remote_sha(str(data["git_remote"]), sha)
    config_bytes = subprocess.check_output(["git", "show", f"{sha}:{config}"], cwd=ROOT)
    config_data = tomllib.loads(config_bytes.decode("utf-8"))
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    slug = f"{selected['slug_prefix']}-{sha[:7]}"
    kernel_ref = f"{data['owner']}/{slug}"
    dirty = command_output(["git", "status", "--porcelain"])
    if dirty:
        print("Working tree has uncommitted files; the run still uses only the verified HEAD commit:")
        print(dirty)
    with tempfile.TemporaryDirectory(prefix="pretraining-kaggle-") as temporary:
        staging = Path(temporary)
        notebook_name = "pretraining_t4x2_launcher.ipynb"
        (staging / notebook_name).write_text(
            json.dumps(
                notebook(repo_url, sha, config, str(data["output_root"])),
                indent=1,
            ),
            encoding="utf-8",
        )
        metadata = {
            "id": kernel_ref,
            # Kaggle rejects a new kernel when title slugification does not
            # reproduce the explicit id. Keep the scientific title in the
            # notebook and use the immutable run slug as the API title.
            "title": slug,
            "code_file": notebook_name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_internet": True,
            "machine_shape": str(data["accelerator"]),
        }
        (staging / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        push_output = command_output(
            [
                "kaggle",
                "kernels",
                "push",
                "--path",
                str(staging),
                "--accelerator",
                str(data["accelerator"]),
            ]
        )
        print(push_output)
        version, url = parse_push_result(push_output, kernel_ref)
    launch_dir = ROOT / "audit" / "launches"
    launch_dir.mkdir(parents=True, exist_ok=True)
    (launch_dir / f"{slug}.json").write_text(
        json.dumps(
            {
                "kernel": kernel_ref,
                "version": version,
                "url": url,
                "experiment": name,
                "git_sha": sha,
                "git_remote": str(data["git_remote"]),
                "git_remote_url": repo_url,
                "config": config,
                "config_sha256": config_sha256,
                "root_seed": int(config_data["run"]["seed"]),
                "requested_accelerator": str(data["accelerator"]),
                "purpose": selected["purpose"],
                "launched_at_unix": time.time(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(kernel_ref)
    return kernel_ref


def status(kernel: str) -> str:
    output = command_output(["kaggle", "kernels", "status", kernel])
    print(output)
    return output


def collect(kernel: str) -> Path:
    slug = kernel.split("/")[-1]
    launch_path = ROOT / "audit" / "launches" / f"{slug}.json"
    if not launch_path.is_file():
        raise SystemExit(f"missing local launch record: {launch_path}")
    launch_record = json.loads(launch_path.read_text(encoding="utf-8"))
    if launch_record.get("kernel") != kernel:
        raise SystemExit(f"launch record does not match requested kernel: {launch_record}")
    version = int(launch_record["version"])
    versioned_kernel = f"{kernel}/{version}"
    terminal_status = command_output(["kaggle", "kernels", "status", versioned_kernel])
    print(terminal_status)
    if not any(state in terminal_status.upper() for state in ("COMPLETE", "ERROR", "CANCEL")):
        raise SystemExit(f"Kaggle run is not terminal: {terminal_status}")
    listing = exact_kernel_listing(kernel)
    destination = ROOT / "audit" / "runs" / slug
    destination.mkdir(parents=True, exist_ok=True)
    pattern = (
        r"(^|/)(summary|training-result|architecture-gate-progress|cpu-benchmark|world-validation|"
        r"trivial-policy-baselines|phase_status|environment|audit-manifest)\.json$"
        r"|(^|/)logs/(pip-install|rustup-download|rustup-install|rust-toolchain|maturin-build|"
        r"world-wheel-install|rust-tests|python-tests|world-validation|cpu-benchmark|"
        r"trivial-policy-baselines|gpu-training)\.log$"
    )
    command_run(
        [
            "kaggle",
            "kernels",
            "output",
            versioned_kernel,
            "--path",
            str(destination),
            "--force",
            "--file-pattern",
            pattern,
        ]
    )
    summary_files = list(destination.rglob("summary.json"))
    manifest_files = list(destination.rglob("audit-manifest.json"))
    if len(summary_files) != 1 or len(manifest_files) != 1:
        raise SystemExit(
            f"collection did not retrieve exactly one summary and manifest: {summary_files}, {manifest_files}"
        )
    summary = json.loads(summary_files[0].read_text(encoding="utf-8"))
    manifest = json.loads(manifest_files[0].read_text(encoding="utf-8"))
    if summary_files[0].parent != manifest_files[0].parent:
        raise SystemExit("summary and manifest were not collected from one result root")
    run_root = summary_files[0].parent
    if summary.get("git_sha") != launch_record["git_sha"] or manifest.get("git_sha") != launch_record["git_sha"]:
        raise SystemExit("remote Git SHA does not match the launch record")
    if (
        summary.get("config_sha256") != launch_record["config_sha256"]
        or manifest.get("config_sha256") != launch_record["config_sha256"]
    ):
        raise SystemExit("remote configuration hash does not match the launch record")
    local_config_bytes = subprocess.check_output(
        ["git", "show", f"{launch_record['git_sha']}:{launch_record['config']}"], cwd=ROOT
    )
    local_config_hash = hashlib.sha256(local_config_bytes).hexdigest()
    if local_config_hash != launch_record["config_sha256"]:
        raise SystemExit("local configuration no longer matches the launched commit")
    verified_entries = verify_downloaded_against_manifest(run_root, manifest)
    training_files = list(run_root.glob("training-result.json"))
    training_result = (
        json.loads(training_files[0].read_text(encoding="utf-8")) if len(training_files) == 1 else {}
    )
    receipt = {
        "schema_version": 1,
        "run_id": versioned_kernel,
        "experiment": launch_record["experiment"],
        "kernel": kernel,
        "kaggle": {
            "owner_slug_version": versioned_kernel,
            "url": f"https://www.kaggle.com/code/{kernel}/versions/{version}",
            "push_url": launch_record["url"],
            "terminal_status": terminal_status,
            "completion_time": listing.get("lastRunTime"),
        },
        "collected_at_unix": time.time(),
        "status": summary.get("status"),
        "source": {
            "git_remote": launch_record["git_remote"],
            "git_remote_url": launch_record["git_remote_url"],
            "git_sha": summary.get("git_sha"),
        },
        "configuration": {
            "path": launch_record["config"],
            "sha256": summary.get("config_sha256"),
        },
        "root_seed": launch_record["root_seed"],
        "requested_accelerator": launch_record["requested_accelerator"],
        "observed_accelerator_inventory": training_result.get("device_names"),
        "world_size": training_result.get("world_size"),
        "upstream_kaggle_versions": [],
        "model_sha256": summary.get("model_sha256"),
        "recovery_artifact": summary.get("recovery_artifact"),
        "architecture_gate_passed": summary.get("architecture_gate_passed"),
        "architecture_gate_progress": summary.get("architecture_gate_progress"),
        "scientific_report": None,
        "audit_verified": True,
        "verified_manifest_entries": verified_entries,
        "remote_manifest_sha256": hashlib.sha256(
            manifest_files[0].read_bytes()
        ).hexdigest(),
        "downloaded_files": [
            {
                "path": path.relative_to(destination).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(destination.rglob("*"))
            if path.is_file() and path.name != "receipt.json"
        ],
    }
    (destination / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--experiment", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--experiment", required=True)
    run_parser.add_argument("--poll-seconds", type=int, default=30)

    for command in ("status", "logs", "collect"):
        child = subparsers.add_parser(command)
        child.add_argument("--kernel", required=True)

    args = parser.parse_args()
    if args.command == "launch":
        launch(args.experiment)
    elif args.command == "status":
        status(args.kernel)
    elif args.command == "logs":
        command_run(
            [
                "kaggle",
                "kernels",
                "output",
                args.kernel,
                "--path",
                str(ROOT / "audit" / "logs" / args.kernel.split("/")[-1]),
                "--force",
                "--file-pattern",
                r"(^|/)logs/.*\.log$",
            ]
        )
    elif args.command == "collect":
        collect(args.kernel)
    elif args.command == "run":
        kernel = launch(args.experiment)
        while True:
            output = status(kernel).lower()
            if "complete" in output:
                collect(kernel)
                break
            if "error" in output or "failed" in output or "cancel" in output:
                collect(kernel)
                raise SystemExit(1)
            time.sleep(max(5, min(args.poll_seconds, 60)))


if __name__ == "__main__":
    main()
