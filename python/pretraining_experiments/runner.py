"""Single non-interactive Kaggle runner for robot pretraining."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
import traceback
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Write JSON via a temporary file and one rename.

    A run killed partway through a plain write leaves a truncated audit file,
    losing exactly the evidence that explains why it was killed.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def kill_process_tree(process: subprocess.Popen) -> None:
    """Kill a child and every process it spawned."""

    if os.name != "nt":
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    process.kill()
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        pass


ACTIVE_CHILD: subprocess.Popen | None = None
RUN_DEADLINE: float | None = None


def effective_timeout(phase_timeout: int) -> int:
    """Clamp a phase timeout to whatever wall-clock budget remains."""

    if RUN_DEADLINE is None:
        return phase_timeout
    remaining = RUN_DEADLINE - time.time()
    if remaining <= 0:
        raise RuntimeError("run wall-clock budget exhausted before this phase began")
    return max(1, int(min(phase_timeout, remaining)))


def start_wall_clock_watchdog(budget: int, output_root: Path, context: dict[str, Any]) -> None:
    """Terminate the run at a fixed wall-clock deadline, whatever it is doing.

    Phase timeouts only fire for a phase that is being waited on. A wedge
    anywhere else, including in this runner itself, would otherwise hold a
    Kaggle session open until the platform limit. This deadline is
    unconditional: it kills the child process tree, writes what evidence it
    can, and exits without waiting for anything.
    """

    global RUN_DEADLINE
    RUN_DEADLINE = time.time() + budget

    def enforce() -> None:
        while True:
            remaining = (RUN_DEADLINE or 0) - time.time()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 15.0))
        message = (
            f"hard wall-clock stop: the {budget}s run budget expired; "
            "killing the child process tree and exiting"
        )
        print(message, flush=True)
        if ACTIVE_CHILD is not None:
            kill_process_tree(ACTIVE_CHILD)
        try:
            atomic_json(
                output_root / "summary.json",
                {"status": "failed", "error": message, "wall_clock_budget_seconds": budget, **context},
            )
            atomic_json(
                output_root / "audit-manifest.json",
                {"status": "failed", "error": message, "artifacts": []},
            )
        except Exception:  # the exit must happen even if evidence cannot be written
            pass
        os._exit(75)

    threading.Thread(target=enforce, daemon=True).start()


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout: int,
) -> None:
    """Stream a child's output to its log and to this runner's stdout.

    Buffering the child in a pipe and writing the log only after it exits
    loses the log entirely when the child is killed or times out, which is
    exactly when the log matters most. Streaming also lets the Kaggle console
    show progress inside the longest phase instead of going silent for it.
    """

    global ACTIVE_CHILD
    timeout = effective_timeout(timeout)
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        bufsize=1,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # Own session/group: `accelerate launch` spawns one worker per GPU, and
        # killing only the launcher would orphan workers that still hold the
        # GPUs. The whole tree must be killable as a unit.
        **({"start_new_session": True} if os.name != "nt" else {}),
    )
    ACTIVE_CHILD = process
    tail: deque[str] = deque(maxlen=120)

    def pump() -> None:
        assert process.stdout is not None
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(command) + "\n")
            handle.flush()
            for line in process.stdout:
                tail.append(line.rstrip("\n"))
                handle.write(line)
                handle.flush()
                print(line, end="", flush=True)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_process_tree(process)
        reader.join(timeout=30)
        raise RuntimeError(
            f"command timed out after {timeout}s: {command}\n"
            f"log: {log_path}\n--- child log tail ---\n" + "\n".join(tail)
        ) from None
    reader.join(timeout=30)
    if returncode:
        raise RuntimeError(
            f"command failed with exit code {returncode} after "
            f"{time.time() - started:.1f}s: {command}\n"
            f"log: {log_path}\n--- child log tail ---\n" + "\n".join(tail)
        )


def phase_update(path: Path, phases: dict[str, Any], name: str, status: str, **extra: Any) -> None:
    entry = phases.setdefault(name, {})
    now = time.time()
    if status == "running":
        entry["started_at"] = now
    elif "started_at" in entry:
        entry["elapsed_seconds"] = now - float(entry["started_at"])
    entry.update({"status": status, "timestamp": now, **extra})
    atomic_json(path, phases)


def safe_environment() -> dict[str, str]:
    allowed = {
        "KAGGLE_KERNEL_RUN_TYPE",
        "KAGGLE_URL_BASE",
        "KAGGLE_DATA_PROXY_TOKEN",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "PYTHONHASHSEED",
    }
    result = {}
    for name in allowed:
        if name not in os.environ:
            continue
        result[name] = "<redacted>" if "TOKEN" in name else os.environ[name]
    return result


def capture_environment(repo: Path, config_text: str, command_line: list[str]) -> dict[str, Any]:
    def checked(command: list[str]) -> str:
        return subprocess.check_output(command, cwd=repo, text=True, stderr=subprocess.STDOUT).strip()

    return {
        "git_sha": checked(["git", "rev-parse", "HEAD"]),
        "git_status": checked(["git", "status", "--porcelain"]),
        "config_sha256": sha256_text(config_text),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "command_line": command_line,
        "safe_environment": safe_environment(),
        "nvidia_smi": subprocess.run(
            ["nvidia-smi"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
        ).stdout,
        "disk_free_bytes": shutil.disk_usage("/kaggle/working").free
        if Path("/kaggle/working").exists()
        else shutil.disk_usage(repo).free,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config_text = config_path.read_text(encoding="utf-8")
    config = tomllib.loads(config_text)
    repo = config_path.parents[2]
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / "logs"
    phase_path = output_root / "phase_status.json"
    phases: dict[str, Any] = {}
    status = "failed"
    error: str | None = None
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_MODE": "disabled",
            "PYTHONPATH": str(repo / "python")
            + os.pathsep
            + env.get("PYTHONPATH", ""),
        }
    )

    start_wall_clock_watchdog(
        int(config["run"].get("max_wall_clock_seconds", 4800)),
        output_root,
        {
            "purpose": config["run"]["purpose"],
            "checkpoint_label": config["run"]["checkpoint_label"],
            "config_sha256": sha256_text(config_text),
        },
    )

    try:
        phase_update(phase_path, phases, "capture_environment", "running")
        environment = capture_environment(repo, config_text, sys.argv)
        atomic_json(output_root / "environment.json", environment)
        if environment["git_status"]:
            raise RuntimeError(f"Kaggle source checkout is not clean: {environment['git_status']}")
        phase_update(phase_path, phases, "capture_environment", "complete")

        phase_update(phase_path, phases, "install_and_build", "running")
        run_logged(
            [sys.executable, "-m", "pip", "install", "-r", str(repo / "requirements-kaggle.txt")],
            cwd=repo,
            env=env,
            log_path=logs / "pip-install.log",
            timeout=900,
        )
        toolchain = tomllib.loads((repo / "rust-toolchain.toml").read_text(encoding="utf-8"))[
            "toolchain"
        ]["channel"]
        cargo_home = Path("/tmp/pretraining-cargo")
        rustup_home = Path("/tmp/pretraining-rustup")
        env.update(
            {
                "CARGO_HOME": str(cargo_home),
                "RUSTUP_HOME": str(rustup_home),
                "RUSTUP_TOOLCHAIN": str(toolchain),
                "PATH": str(cargo_home / "bin") + os.pathsep + env["PATH"],
            }
        )
        rustup_installer = Path("/tmp/rustup-init.sh")
        if not (cargo_home / "bin" / "cargo").is_file():
            run_logged(
                [
                    "curl",
                    "--proto",
                    "=https",
                    "--tlsv1.2",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "https://sh.rustup.rs",
                    "--output",
                    str(rustup_installer),
                ],
                cwd=repo,
                env=env,
                log_path=logs / "rustup-download.log",
                timeout=300,
            )
            run_logged(
                [
                    "sh",
                    str(rustup_installer),
                    "-y",
                    "--no-modify-path",
                    "--profile",
                    "minimal",
                    "--default-toolchain",
                    str(toolchain),
                ],
                cwd=repo,
                env=env,
                log_path=logs / "rustup-install.log",
                timeout=900,
            )
        run_logged(
            [str(cargo_home / "bin" / "rustc"), "--version", "--verbose"],
            cwd=repo,
            env=env,
            log_path=logs / "rust-toolchain.log",
            timeout=60,
        )
        wheel_dir = Path("/tmp/pretraining-wheels") if Path("/tmp").exists() else repo / "dist"
        wheel_dir.mkdir(parents=True, exist_ok=True)
        run_logged(
            [
                sys.executable,
                "-m",
                "maturin",
                "build",
                "--release",
                "--locked",
                "--manifest-path",
                str(repo / "crates" / "world-py" / "Cargo.toml"),
                "--out",
                str(wheel_dir),
            ],
            cwd=repo,
            env=env,
            log_path=logs / "maturin-build.log",
            timeout=1200,
        )
        wheels = sorted(wheel_dir.glob("pretraining_world_py-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one pretraining world wheel, found {wheels}")
        run_logged(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheels[0])],
            cwd=repo,
            env=env,
            log_path=logs / "world-wheel-install.log",
            timeout=300,
        )
        phase_update(
            phase_path,
            phases,
            "install_and_build",
            "complete",
            wheel_sha256=sha256_file(wheels[0]),
        )

        phase_update(phase_path, phases, "correctness_tests", "running")
        run_logged(
            ["cargo", "test", "--manifest-path", str(repo / "Cargo.toml"), "--workspace", "--locked"],
            cwd=repo,
            env=env,
            log_path=logs / "rust-tests.log",
            timeout=900,
        )
        run_logged(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(repo / "python" / "tests"),
                "-v",
            ],
            cwd=repo,
            env=env,
            log_path=logs / "python-tests.log",
            timeout=1200,
        )
        phase_update(phase_path, phases, "correctness_tests", "complete")

        phase_update(phase_path, phases, "world_validation", "running")
        validation_code = (
            "import json,tomllib,pretraining_world_py; from pathlib import Path; "
            f"c=tomllib.loads(Path(r'{config_path}').read_text()); w=c['world']; "
            "k={x:w[x] for x in ('d_min','d_max','gain_min','gain_max','action_limit','calibration_pulse','max_control_steps')}; "
            "r=pretraining_world_py.validate_generated_worlds(seed=int(w['validation_seed']),start_index=0,"
            "count=int(c['preflight']['validation_instances']),**k); "
            f"Path(r'{output_root / 'world-validation.json'}').write_text(json.dumps(r,indent=2,sort_keys=True))"
        )
        run_logged(
            [sys.executable, "-c", validation_code],
            cwd=repo,
            env=env,
            log_path=logs / "world-validation.log",
            timeout=300,
        )
        phase_update(phase_path, phases, "world_validation", "complete")

        phase_update(phase_path, phases, "cpu_benchmark", "running")
        run_logged(
            [
                sys.executable,
                "-m",
                "pretraining_experiments.benchmarks",
                "--config",
                str(config_path),
                "--output",
                str(output_root / "cpu-benchmark.json"),
            ],
            cwd=repo,
            env=env,
            log_path=logs / "cpu-benchmark.log",
            timeout=900,
        )
        phase_update(phase_path, phases, "cpu_benchmark", "complete")

        phase_update(phase_path, phases, "trivial_policy_baselines", "running")
        run_logged(
            [
                sys.executable,
                "-m",
                "pretraining_experiments.baselines",
                "--config",
                str(config_path),
                "--output",
                str(output_root / "trivial-policy-baselines.json"),
            ],
            cwd=repo,
            env=env,
            log_path=logs / "trivial-policy-baselines.log",
            timeout=300,
        )
        phase_update(phase_path, phases, "trivial_policy_baselines", "complete")

        phase_update(phase_path, phases, "gpu_training", "running")
        gpu_command = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--multi_gpu",
            "--num_processes",
            "2",
            "--num_machines",
            "1",
            "--mixed_precision",
            str(config["run"]["mixed_precision"]),
            "--dynamo_backend",
            "no",
            "-m",
            "pretraining_experiments.train",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
        ]
        run_logged(
            gpu_command,
            cwd=repo,
            env=env,
            log_path=logs / "gpu-training.log",
            timeout=int(config["run"].get("gpu_phase_timeout_seconds", 1800)),
        )
        training_result = json.loads(
            (output_root / "training-result.json").read_text(encoding="utf-8")
        )
        if not training_result.get("architecture_gate_passed"):
            raise RuntimeError("GPU architecture integration gate did not pass")
        phase_update(phase_path, phases, "gpu_training", "complete")
        status = "complete"
    except Exception as exc:  # package evidence before propagating the failure
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        print(error, file=sys.stderr, flush=True)
    finally:
        artifacts = []
        for path in sorted(output_root.rglob("*")):
            if path.is_file() and "checkpoints" not in path.parts:
                artifacts.append(
                    {
                        "path": path.relative_to(output_root).as_posix(),
                        "size": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        manifest = {
            "status": status,
            "error": error,
            "config_sha256": sha256_text(config_text),
            "git_sha": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                text=True,
                stdout=subprocess.PIPE,
                check=False,
            ).stdout.strip(),
            "artifacts": artifacts,
        }
        atomic_json(output_root / "audit-manifest.json", manifest)
        summary = {
            "status": status,
            "error": error,
            "purpose": config["run"]["purpose"],
            "checkpoint_label": config["run"]["checkpoint_label"],
            "phase_status": phases,
            "git_sha": manifest["git_sha"],
            "config_sha256": manifest["config_sha256"],
        }
        if (output_root / "architecture-gate-progress.json").exists():
            summary["architecture_gate_progress"] = json.loads(
                (output_root / "architecture-gate-progress.json").read_text(encoding="utf-8")
            )
        if (output_root / "training-result.json").exists():
            result = json.loads((output_root / "training-result.json").read_text(encoding="utf-8"))
            summary.update(
                {
                    "architecture_gate_passed": result.get("architecture_gate_passed"),
                    "resume_smoke": result.get("resume_smoke"),
                    "validation": result.get("validation"),
                    "closed_loop": result.get("closed_loop"),
                    "oracle_closed_loop": result.get("oracle_closed_loop"),
                    "untrained_baseline": result.get("untrained_baseline"),
                    "paired_learning_delta": result.get("paired_learning_delta"),
                    "model_sha256": result.get("model_sha256"),
                    "recovery_artifact": result.get("recovery_artifact"),
                    "root_seed": result.get("root_seed"),
                    "accelerator_inventory": {
                        "world_size": result.get("world_size"),
                        "device_names": result.get("device_names"),
                        "torch_version": result.get("torch_version"),
                    },
                }
            )
        if (output_root / "trivial-policy-baselines.json").exists():
            summary["trivial_policy_baselines"] = json.loads(
                (output_root / "trivial-policy-baselines.json").read_text(encoding="utf-8")
            )
        atomic_json(output_root / "summary.json", summary)

    if status != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
