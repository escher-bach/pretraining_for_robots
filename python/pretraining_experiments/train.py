"""Maintained Trainer path for procedural trajectory pretraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import tomllib
from typing import Any, Iterator

from accelerate import Accelerator
import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

import pretraining_world_py

from .data import (
    MODEL_FIELDS,
    assert_world_model_compatibility,
    generate_torch_batch,
    tensorize_rollout,
    world_kwargs,
)
from .evaluation import (
    error_thresholds,
    paired_learning_delta,
    summarize_episode_rows,
    summarize_trial_rows,
)
from .model import (
    PretrainingConfig,
    PretrainingForTrajectoryPrediction,
    assert_selected_parameter_report,
    assert_selected_profile,
    parameter_report,
)


def mark_stage(name: str) -> None:
    """Emit a per-rank stage marker.

    A collective that the ranks enter unequal numbers of times deadlocks with
    no traceback. Printing every stage from every rank turns that silence into
    a readable record of which rank stopped where.
    """

    rank = os.environ.get("RANK", "0")
    print(f"[pretraining-stage rank{rank}] {time.strftime('%H:%M:%S')} {name}", flush=True)


def assert_every_rank_arrived(trainer: Trainer, stage_name: str) -> None:
    """Port of the STEP 1 preflight rank-completion check."""

    processes = trainer.accelerator.num_processes
    arrived = trainer.accelerator.gather_for_metrics(
        torch.tensor([trainer.args.process_index], device=trainer.args.device)
    ).detach().cpu().tolist()
    if sorted(arrived) != list(range(processes)):
        raise AssertionError(f"not every rank reached {stage_name}: {arrived}")


def load_config(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(config_path: Path, relative: str) -> Path:
    return config_path.resolve().parents[2] / relative


class ProceduralTrajectoryDataset(IterableDataset):
    """Infinite deterministic stream; Trainer/Accelerate owns rank sharding."""

    def __init__(self, *, seed: int, max_tokens: int, world: dict[str, Any]) -> None:
        super().__init__()
        self.seed = seed
        self.max_tokens = max_tokens
        self.world = dict(world)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = torch.utils.data.get_worker_info()
        index = 0 if worker is None else worker.id
        stride = 1 if worker is None else worker.num_workers
        while True:
            batch, _ = generate_torch_batch(
                seed=self.seed,
                start_index=index,
                batch_size=1,
                max_tokens=self.max_tokens,
                world=self.world,
            )
            yield {name: batch[name][0] for name in MODEL_FIELDS}
            index += stride


class FixedCohortDataset(Dataset):
    """Repeat one real generated cohort for a disposable wiring diagnostic."""

    def __init__(self, batch: dict[str, torch.Tensor], repeats: int) -> None:
        size = len(next(iter(batch.values())))
        self.rows = [
            {name: value[row] for name, value in batch.items()} for row in range(size)
        ]
        self.repeats = repeats

    def __len__(self) -> int:
        return len(self.rows) * self.repeats

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.rows[index % len(self.rows)]


def training_arguments(
    *,
    output_dir: Path,
    run: dict[str, Any],
    max_steps: int,
    per_device_batch_size: int,
    warmup_steps: int,
    save: bool,
    use_cpu: bool = False,
) -> TrainingArguments:
    mixed_precision = str(run.get("mixed_precision", "no"))
    return TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=int(run.get("gradient_accumulation_steps", 1)),
        learning_rate=float(run["learning_rate"]),
        weight_decay=float(run["weight_decay"]),
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1.0e-8,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        max_grad_norm=float(run["max_grad_norm"]),
        fp16=mixed_precision == "fp16" and not use_cpu,
        bf16=mixed_precision == "bf16" and not use_cpu,
        use_cpu=use_cpu,
        logging_strategy="steps",
        logging_steps=max(1, int(run.get("log_every", 1))),
        logging_first_step=True,
        save_strategy="steps" if save else "no",
        save_steps=max_steps,
        save_total_limit=2,
        eval_strategy="no",
        prediction_loss_only=True,
        remove_unused_columns=False,
        label_names=[
            "action_targets",
            "action_target_mask",
            "future_targets",
            "future_target_mask",
        ],
        dataloader_num_workers=0,
        dataloader_drop_last=False,
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
        average_tokens_across_devices=False,
        include_num_input_tokens_seen=True,
        report_to=[],
        disable_tqdm=True,
        seed=int(run["seed"]),
        data_seed=int(run["seed"]),
    )


class StopAfterStepCallback(TrainerCallback):
    """Stop after a declared step while asking Trainer to write its state."""

    def __init__(self, stop_step: int) -> None:
        self.stop_step = stop_step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_step:
            control.should_save = True
            control.should_training_stop = True
        return control


def train_with_resume_smoke(
    *,
    model: PretrainingForTrajectoryPrediction,
    dataset: IterableDataset | Dataset,
    trainer_state_dir: Path,
    run: dict[str, Any],
    total_steps: int,
    per_device_batch_size: int,
    warmup_steps: int,
    resume_checkpoint: str | None = None,
    use_cpu: bool = False,
) -> tuple[Trainer, dict[str, Any], dict[str, Any]]:
    """Exercise standard Trainer save/resume without increasing the step budget."""

    if resume_checkpoint is not None:
        arguments = training_arguments(
            output_dir=trainer_state_dir,
            run=run,
            max_steps=total_steps,
            per_device_batch_size=per_device_batch_size,
            warmup_steps=warmup_steps,
            save=True,
            use_cpu=use_cpu,
        )
        trainer = Trainer(model=model, args=arguments, train_dataset=dataset)
        output = trainer.train(resume_from_checkpoint=resume_checkpoint)
        return trainer, output.metrics, {
            "passed": True,
            "mode": "external",
            "checkpoint": str(Path(resume_checkpoint).resolve()),
            "resumed_to_step": int(trainer.state.global_step),
        }

    smoke_step = int(run["resume_smoke_update"])
    if not 0 < smoke_step < total_steps:
        raise ValueError("resume_smoke_update must be between zero and max_updates")
    first_arguments = training_arguments(
        output_dir=trainer_state_dir,
        run=run,
        max_steps=total_steps,
        per_device_batch_size=per_device_batch_size,
        warmup_steps=warmup_steps,
        save=True,
        use_cpu=use_cpu,
    )
    first_trainer = Trainer(
        model=model,
        args=first_arguments,
        train_dataset=dataset,
        callbacks=[StopAfterStepCallback(smoke_step)],
    )
    first_output = first_trainer.train()
    checkpoint = trainer_state_dir / f"checkpoint-{smoke_step}"
    required = ("model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing or int(first_trainer.state.global_step) != smoke_step:
        raise RuntimeError(
            "Trainer resume-smoke checkpoint incomplete: "
            f"step={first_trainer.state.global_step}, missing={missing}"
        )

    resumed_arguments = training_arguments(
        output_dir=trainer_state_dir,
        run=run,
        max_steps=total_steps,
        per_device_batch_size=per_device_batch_size,
        warmup_steps=warmup_steps,
        save=True,
        use_cpu=use_cpu,
    )
    resumed_trainer = Trainer(
        model=PretrainingForTrajectoryPrediction(model.config),
        args=resumed_arguments,
        train_dataset=dataset,
    )
    resumed_output = resumed_trainer.train(resume_from_checkpoint=str(checkpoint))
    passed = int(resumed_trainer.state.global_step) == total_steps
    resume_smoke = {
        "passed": passed,
        "mode": "automatic",
        "checkpoint": checkpoint.name,
        "checkpoint_step": smoke_step,
        "resumed_to_step": int(resumed_trainer.state.global_step),
        "required_state_files": list(required),
        "first_segment_metrics": first_output.metrics,
    }
    if not passed:
        raise RuntimeError(f"Trainer resume smoke failed: {resume_smoke}")
    return resumed_trainer, resumed_output.metrics, resume_smoke


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=False) for name, tensor in batch.items()}


@torch.no_grad()
def teacher_forced_eval(
    accelerator: Accelerator,
    model: torch.nn.Module,
    config: dict[str, Any],
    start_index: int,
) -> dict[str, Any]:
    run = config["run"]
    world = config["world"]
    sequence_length = int(config["model"]["sequence_length"])
    batch_size = int(run["per_device_batch_size"])
    model.eval()
    records: list[torch.Tensor] = []
    for batch_index in range(int(run["eval_batches"])):
        local_start = (
            start_index
            + (batch_index * accelerator.num_processes + accelerator.process_index) * batch_size
        )
        batch, metadata = generate_torch_batch(
            seed=int(world["validation_seed"]),
            start_index=local_start,
            batch_size=batch_size,
            max_tokens=sequence_length,
            world=world,
        )
        batch = move_batch(batch, accelerator.device)
        output = model(**batch)
        action_abs = (output.action_predictions - batch["action_targets"]).abs()
        future_abs = (output.future_predictions - batch["future_targets"]).abs()
        for row, dimension in enumerate(metadata["dimensions"]):
            action_mask = batch["action_target_mask"][row]
            future_mask = batch["future_target_mask"][row]
            records.append(
                torch.tensor(
                    [
                        float(dimension),
                        float((action_abs[row] * action_mask).sum().item()),
                        float(action_mask.sum().item()),
                        float((future_abs[row] * future_mask).sum().item()),
                        float(future_mask.sum().item()),
                    ],
                    device=accelerator.device,
                )
            )
    gathered = accelerator.gather_for_metrics(torch.stack(records)).cpu()
    result: dict[str, Any] = {"by_dimension": {}}
    total_action_sum = total_action_count = total_future_sum = total_future_count = 0.0
    for dimension in range(int(world["d_min"]), int(world["d_max"]) + 1):
        rows = gathered[gathered[:, 0] == float(dimension)]
        action_sum = rows[:, 1].sum().item()
        action_count = rows[:, 2].sum().item()
        future_sum = rows[:, 3].sum().item()
        future_count = rows[:, 4].sum().item()
        result["by_dimension"][str(dimension)] = {
            "episodes": int(rows.shape[0]),
            "action_l1": action_sum / max(action_count, 1.0),
            "future_l1": future_sum / max(future_count, 1.0),
        }
        total_action_sum += action_sum
        total_action_count += action_count
        total_future_sum += future_sum
        total_future_count += future_count
    result["action_l1"] = total_action_sum / max(total_action_count, 1.0)
    result["future_l1"] = total_future_sum / max(total_future_count, 1.0)
    return result


@torch.no_grad()
def closed_loop_eval(
    accelerator: Accelerator,
    model: torch.nn.Module,
    config: dict[str, Any],
    *,
    seed: int,
    start_index: int,
    episodes_per_rank: int,
    use_oracle: bool = False,
) -> dict[str, Any]:
    world = config["world"]
    sequence_length = int(config["model"]["sequence_length"])
    local_start = start_index + accelerator.process_index * episodes_per_rank
    rollouts = pretraining_world_py.RolloutBatch(
        seed=seed,
        start_index=local_start,
        batch_size=episodes_per_rank,
        max_tokens=sequence_length,
        **world_kwargs(world),
    )
    initial_summary = rollouts.summary()
    initial_errors = [float(value) for value in initial_summary["terminal_error"]]
    normalized_cost = [0.0] * episodes_per_rank
    physical_cost = [0.0] * episodes_per_rank
    local_trial_rows = [
        [0.0, initial_errors[index], 0.0, 0.0] for index in range(episodes_per_rank)
    ]
    model.eval()
    for trial in range(1, int(world["max_control_steps"]) + 1):
        raw = rollouts.learner_batch()
        if use_oracle:
            actions = rollouts.privileged_oracle_actions()
        else:
            tensors = move_batch(tensorize_rollout(raw, "cpu"), accelerator.device)
            predictions = model(**tensors).action_predictions[..., 0].detach().float().cpu()
            actions = []
            for row, dimension in enumerate(raw["dimensions"]):
                if raw["done"][row]:
                    actions.append([])
                    continue
                positions = raw["query_positions"][row][:dimension]
                actions.append([float(predictions[row, position]) for position in positions])
        for index, action in enumerate(actions):
            normalized_cost[index] += sum(abs(value) for value in action)
            physical_cost[index] += float(world["action_limit"]) * sum(
                abs(value) for value in action
            )
        rollouts.step(actions)
        current = rollouts.summary()
        local_trial_rows.extend(
            [
                float(trial),
                float(current["terminal_error"][index]),
                normalized_cost[index],
                physical_cost[index],
            ]
            for index in range(episodes_per_rank)
        )
    summary = rollouts.summary()
    local = torch.tensor(
        [
            [
                float(summary["dimensions"][i]),
                initial_errors[i],
                float(summary["terminal_error"][i]),
                float(summary["steps"][i]),
                normalized_cost[i],
                physical_cost[i],
            ]
            for i in range(episodes_per_rank)
        ],
        device=accelerator.device,
    )
    gathered = accelerator.gather_for_metrics(local).cpu()
    trial_tensor = torch.tensor(local_trial_rows, device=accelerator.device)
    gathered_trials = accelerator.gather_for_metrics(trial_tensor).cpu()
    thresholds = error_thresholds(config)
    result = summarize_episode_rows(gathered, thresholds)
    result["trial_curve"] = summarize_trial_rows(gathered_trials, thresholds)
    result["policy"] = "privileged_public_oracle" if use_oracle else "learner"
    return result


@torch.no_grad()
def held_out_learner_evaluation(
    accelerator: Accelerator,
    model: torch.nn.Module,
    config: dict[str, Any],
    *,
    rollout_episodes_per_rank: int,
) -> dict[str, Any]:
    """Evaluate one learner on the fixed held-out support.

    The step-zero baseline and the trained candidate both go through this one
    function so their evaluation support is identical by construction rather
    than by two call sites happening to agree.
    """

    return {
        "validation": teacher_forced_eval(accelerator, model, config, start_index=0),
        "closed_loop": closed_loop_eval(
            accelerator,
            model,
            config,
            seed=int(config["world"]["rollout_seed"]),
            start_index=0,
            episodes_per_rank=rollout_episodes_per_rank,
        ),
    }


def run_overfit_gate(
    model_config: PretrainingConfig,
    config: dict[str, Any],
    output_root: Path,
    *,
    use_cpu: bool = False,
) -> tuple[dict[str, Any], Trainer]:
    run = config["run"]
    world = config["world"]
    per_device = int(run["overfit_per_device_batch_size"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    cohort, _ = generate_torch_batch(
        seed=int(world["overfit_seed"]),
        start_index=0,
        batch_size=per_device * world_size,
        max_tokens=int(config["model"]["sequence_length"]),
        world=world,
    )
    updates = int(run["overfit_updates"])
    train_dataset = FixedCohortDataset(cohort, repeats=max(2, updates))
    eval_dataset = FixedCohortDataset(cohort, repeats=1)
    set_seed(int(run["seed"]))
    model = PretrainingForTrajectoryPrediction(model_config)
    arguments = training_arguments(
        output_dir=output_root / "diagnostic-trainer",
        run=run,
        max_steps=updates,
        per_device_batch_size=per_device,
        warmup_steps=int(run.get("overfit_warmup_updates", 0)),
        save=False,
        use_cpu=use_cpu,
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=train_dataset)
    initial = float(trainer.evaluate(eval_dataset=eval_dataset)["eval_loss"])
    train_output = trainer.train()
    final = float(trainer.evaluate(eval_dataset=eval_dataset)["eval_loss"])
    required = float(run["overfit_required_fraction"])
    passed = math.isfinite(initial) and math.isfinite(final) and final <= required * initial
    result = {
        "initial_loss": initial,
        "final_loss": final,
        "required_final_fraction": required,
        "observed_final_fraction": final / initial,
        "updates": updates,
        "successful_optimizer_steps": int(trainer.state.global_step),
        "trainer_metrics": train_output.metrics,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"real-batch Trainer overfit gate failed: {result}")
    return result, trainer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_inventory(root: Path) -> dict[str, Any]:
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "remote_path": (Path(root.parents[1].name) / "checkpoints" / root.name).as_posix(),
        "size": sum(int(item["size"]) for item in files),
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


def load_lineage_model(
    model_config: PretrainingConfig,
    parent_model: str | None,
) -> tuple[PretrainingForTrajectoryPrediction, str | None]:
    if parent_model is None:
        return PretrainingForTrajectoryPrediction(model_config), None
    parent = Path(parent_model).resolve()
    model = PretrainingForTrajectoryPrediction.from_pretrained(parent)
    assert_selected_profile(model.config)
    assert_world_model_compatibility(model.config)
    return model, str(parent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--parent-model")
    parser.add_argument("--resume-checkpoint")
    args = parser.parse_args()
    if args.parent_model and args.resume_checkpoint:
        parser.error("--parent-model and --resume-checkpoint are mutually exclusive")
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    run = config["run"]
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if bool(run["require_two_t4"]):
        if torch.cuda.device_count() != 2:
            raise RuntimeError(f"expected exactly two CUDA devices, found {torch.cuda.device_count()}")
        names = [torch.cuda.get_device_name(index) for index in range(2)]
        if any("T4" not in name.upper() for name in names):
            raise RuntimeError(f"expected two T4 GPUs, found {names}")

    model_config = PretrainingConfig.from_project_json(
        resolve_project_path(config_path, str(config["model"]["config"]))
    )
    assert_selected_profile(model_config)
    assert_world_model_compatibility(
        model_config, profiled=bool(config["world"].get("profiled", False))
    )
    started = time.time()

    mark_stage("architecture-gate:start")
    overfit, diagnostic_trainer = run_overfit_gate(model_config, config, output_root)
    mark_stage("architecture-gate:done")
    mark_stage("diagnostic-closed-loop:start")
    diagnostic_closed_loop = closed_loop_eval(
        diagnostic_trainer.accelerator,
        diagnostic_trainer.model,
        config,
        seed=int(config["world"]["overfit_seed"]),
        start_index=0,
        episodes_per_rank=int(run["overfit_per_device_batch_size"]),
    )
    gate_progress_path = output_root / "architecture-gate-progress.json"
    gate_progress = {
        "overfit_gate": overfit,
        "diagnostic_closed_loop": diagnostic_closed_loop,
        "diagnostic_weights_discarded": True,
        "training_lineage_initialized": False,
        "untrained_baseline": None,
        "resume_smoke": {"attempted": False, "passed": False},
    }
    if diagnostic_trainer.is_world_process_zero():
        gate_progress_path.write_text(
            json.dumps(gate_progress, indent=2, sort_keys=True), encoding="utf-8"
        )

    # Evaluate the exact weights training is about to start from, on the same
    # held-out support used after training. Without this paired step-zero
    # baseline the run can only demonstrate apparatus execution.
    rollout_count = int(config["preflight"]["rollout_episodes_per_rank"])
    set_seed(int(run["seed"]))
    model, parent_model = load_lineage_model(model_config, args.parent_model)
    params = parameter_report(model)
    assert_selected_parameter_report(params)
    mark_stage("untrained-baseline:start")
    untrained_baseline = held_out_learner_evaluation(
        diagnostic_trainer.accelerator,
        model.to(diagnostic_trainer.accelerator.device),
        config,
        rollout_episodes_per_rank=rollout_count,
    )
    untrained_baseline.update(
        {
            "purpose": "fixed step-zero learner on the held-out support; paired baseline for the trained candidate",
            "parent_model": parent_model,
            "updates": 0,
        }
    )
    mark_stage("untrained-baseline:done")
    gate_progress["untrained_baseline"] = untrained_baseline
    if diagnostic_trainer.is_world_process_zero():
        gate_progress_path.write_text(
            json.dumps(gate_progress, indent=2, sort_keys=True), encoding="utf-8"
        )
    del diagnostic_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    checkpoint_root = output_root / "checkpoints" / str(run["checkpoint_label"])
    trainer_state_dir = checkpoint_root / "trainer-state"
    dataset = ProceduralTrajectoryDataset(
        seed=int(config["world"]["train_seed"]),
        max_tokens=int(config["model"]["sequence_length"]),
        world=config["world"],
    )
    gate_progress["training_lineage_initialized"] = True
    gate_progress["resume_smoke"] = {"attempted": True, "passed": False}
    if int(os.environ.get("RANK", "0")) == 0:
        gate_progress_path.write_text(
            json.dumps(gate_progress, indent=2, sort_keys=True), encoding="utf-8"
        )
    mark_stage("lineage-training:start")
    trainer, train_metrics, resume_smoke = train_with_resume_smoke(
        model=model,
        dataset=dataset,
        trainer_state_dir=trainer_state_dir,
        run=run,
        total_steps=int(run["max_updates"]),
        per_device_batch_size=int(run["per_device_batch_size"]),
        warmup_steps=int(run["warmup_updates"]),
        resume_checkpoint=args.resume_checkpoint,
    )
    gate_progress["resume_smoke"] = {**resume_smoke, "attempted": True}
    if trainer.is_world_process_zero():
        gate_progress_path.write_text(
            json.dumps(gate_progress, indent=2, sort_keys=True), encoding="utf-8"
        )

    mark_stage("lineage-training:done")
    assert_every_rank_arrived(trainer, "lineage training")
    mark_stage("final-evaluation:start")
    trained = held_out_learner_evaluation(
        trainer.accelerator,
        trainer.model,
        config,
        rollout_episodes_per_rank=rollout_count,
    )
    validation = trained["validation"]
    closed_loop = trained["closed_loop"]
    oracle_closed_loop = closed_loop_eval(
        trainer.accelerator,
        trainer.model,
        config,
        seed=int(config["world"]["rollout_seed"]),
        start_index=0,
        episodes_per_rank=rollout_count,
        use_oracle=True,
    )

    mark_stage("final-evaluation:done")
    model_dir = checkpoint_root / "model"
    trainer.save_model(model_dir)
    trainer.save_state()
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        weights = model_dir / "model.safetensors"
        recovery = checkpoint_inventory(checkpoint_root)
        result = {
            "status": "complete",
            "checkpoint_label": str(run["checkpoint_label"]),
            "checkpoint_classification": "candidate; source competence and transfer unestablished",
            "parent_model": parent_model,
            "diagnostic_weights_discarded_before_lineage": True,
            "overfit_gate": overfit,
            "diagnostic_closed_loop": diagnostic_closed_loop,
            "resume_smoke": resume_smoke,
            "parameter_report": params,
            "trainer_log_history": trainer.state.log_history,
            "trainer_metrics": train_metrics,
            "validation": validation,
            "closed_loop": closed_loop,
            "oracle_closed_loop": oracle_closed_loop,
            "untrained_baseline": untrained_baseline,
            "paired_learning_delta": paired_learning_delta(untrained_baseline, trained),
            "world_versions": pretraining_world_py.versions(),
            "updates": int(trainer.state.global_step),
            "global_episodes": int(run["max_updates"])
            * int(run["per_device_batch_size"])
            * int(run.get("gradient_accumulation_steps", 1))
            * int(trainer.args.world_size),
            "elapsed_seconds": time.time() - started,
            "world_size": int(trainer.args.world_size),
            "device_names": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
            "torch_version": torch.__version__,
            "transformers_trainer_owned": True,
            "model_sha256": sha256_file(weights),
            "recovery_artifact": recovery,
            "root_seed": int(run["seed"]),
            "completed_at_unix": time.time(),
            "architecture_gate_passed": bool(overfit["passed"] and resume_smoke["passed"]),
        }
        (output_root / "training-result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))
    trainer.accelerator.wait_for_everyone()
    mark_stage("run:complete")


def shutdown_distributed() -> None:
    """Tear the process group down explicitly before interpreter exit.

    Leaving it alive is what stalled run f3dbf51: training finished and wrote
    its result in 53 seconds, then the launcher sat for eighty minutes because
    the ranks never completed their own shutdown. Releasing it here makes the
    process exit a step the run performs rather than one it hopes for.
    """

    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
    except Exception as exc:  # shutdown must never mask the real outcome
        print(f"[pretraining-stage] non-fatal distributed shutdown error: {exc}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutdown_distributed()
        mark_stage("process:exiting")
