"""Bounded first-system Trainer smoke for the versioned content boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import hashlib
import argparse
import json
import tomllib
from contextlib import nullcontext
from dataclasses import replace

import numpy as np
from dataclasses import field

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

from .common_content import CommonContentBundle, CommonContentConfig
from .trajectory_data import (
    CalibratedReachDataset,
    TrajectoryDatasetConfig,
    collate_trajectory_batch,
    first_system_action_spec,
    first_system_adapter_specs,
    first_system_raw_width,
)
from .trajectory_world import CalibratedReachConfig


@dataclass(frozen=True)
class FirstTrainingConfig:
    mode: str = "apparatus"
    purpose: str = "apparatus_verification"
    seed: int = 17
    dataset_size: int = 8
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 2
    attention_heads: int = 4
    max_position_embeddings: int = 128
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.0
    per_device_train_batch_size: int = 1
    # Explicit execution target.  The CPU scientific profile must remain CPU
    # even on a CUDA host; the GPU profile opts in with ``cuda``.
    device: str = "cpu"
    mixed_precision: str = "fp32"
    cpu_threads: int = 1
    max_wall_clock_seconds: int = 4800
    max_grad_norm: float = 1.0
    smoke_steps: int = 2
    resumed_steps: int = 3
    world: CalibratedReachConfig = field(default_factory=CalibratedReachConfig)
    sensor_dims: tuple[int, ...] = (8, 10)
    action_dims: tuple[int, ...] = (2, 4)
    goal_modes: tuple[str, ...] = ("absolute", "relative")
    goal_switch_steps: tuple[int, ...] = (-1,)
    disturbance_arms: tuple[bool, ...] = (False,)
    held_out_episodes: int = 0
    evaluation_checkpoints: tuple[int, ...] = (0,)
    closed_loop_checkpoints: tuple[int, ...] = (0,)
    success_tolerance_sensor_l2: float = 0.05
    success_tolerance_physical: float = 0.05
    teacher_effort_objective: str = "bounded_residual_plus_minimum_l2"
    teacher_effort_regularization: float = 1.0e-8
    held_out_seed_start: int = 0
    held_out_seed_count: int = 0
    transform_sensor_permutations: tuple[tuple[int, ...], ...] = ()
    transform_body_permutations: tuple[tuple[int, ...], ...] = ()
    evaluation_seed_start: int = 100000
    evaluation_seed_count: int = 1
    ablations: tuple[str, ...] = ("no_goal", "no_calibration", "no_action_history")


def build_first_system(config: FirstTrainingConfig) -> CommonContentBundle:
    core_config = CommonContentConfig(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        attention_heads=config.attention_heads,
        max_position_embeddings=config.max_position_embeddings,
        role_vocab_size=16,
        key_vocab_size=256,
    )
    return CommonContentBundle.from_parts(
        core_config,
        first_system_adapter_specs(config.sensor_dims, config.action_dims),
        first_system_action_spec(),
    )


def _training_arguments(
    output_dir: Path, config: FirstTrainingConfig, *, max_steps: int, save: bool
) -> TrainingArguments:
    device = _resolve_device(config)
    return TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        max_steps=max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        warmup_steps=0,
        lr_scheduler_type="cosine",
        use_cpu=device.type == "cpu",
        fp16=config.mixed_precision == "fp16" and device.type == "cuda",
        bf16=False,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps" if save else "no",
        # Keep this stable across the resumed Trainer construction so loading
        # a checkpoint does not silently change the serialized cadence.
        save_steps=config.smoke_steps,
        save_total_limit=2,
        eval_strategy="no",
        prediction_loss_only=True,
        remove_unused_columns=False,
        label_names=["action_targets", "action_target_mask"],
        dataloader_num_workers=0,
        dataloader_drop_last=False,
        optim="adamw_torch",
        report_to=[],
        disable_tqdm=True,
        seed=config.seed,
        data_seed=config.seed,
    )


def _resolve_device(config: FirstTrainingConfig) -> torch.device:
    """Resolve and validate the run's explicit execution target."""
    requested = str(config.device).lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu, cuda, or auto")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU scientific profile requests cuda but CUDA is unavailable")
    if config.mixed_precision not in {"fp32", "fp16"}:
        raise ValueError("mixed_precision must be fp32 or fp16")
    if config.mixed_precision == "fp16" and requested != "cuda":
        raise ValueError("fp16 requires device=cuda")
    return torch.device(requested)


def _evaluation_autocast(config: FirstTrainingConfig, model: torch.nn.Module):
    """Use the same CUDA fp16 autocast contract for manual evaluations."""
    device = next(model.parameters()).device
    if config.mixed_precision == "fp16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


class _StopAfterStep(TrainerCallback):
    """Stop the first phase while retaining the final total-step schedule."""

    def __init__(self, stop_step: int) -> None:
        self.stop_step = stop_step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_step:
            control.should_training_stop = True
        return control


class _FixedEvaluationCallback(TrainerCallback):
    """Run the declared cheap held-out action evaluation at fixed updates."""

    def __init__(self, dataset: CalibratedReachDataset, checkpoints: tuple[int, ...], *, config: FirstTrainingConfig):
        self.dataset = dataset
        self.checkpoints = set(checkpoints)
        self.config = config
        self.records: list[dict[str, Any]] = []
        self.closed_loop_records: list[dict[str, Any]] = []

    def _evaluate(self, step: int, model: Any) -> None:
        model.eval()
        losses: list[float] = []
        with torch.inference_mode(), _evaluation_autocast(self.config, model):
            for index in range(len(self.dataset)):
                output = model(**_move_batch_to_model(
                    collate_trajectory_batch([self.dataset[index]]), model
                ))
                losses.append(float(output.loss.detach().cpu()))
        model.train()
        self.records.append({
            "update": step,
            "heldout_examples": len(self.dataset),
            "heldout_action_loss": float(np.mean(losses)),
        })

        if step in set(self.config.closed_loop_checkpoints):
            # evaluation_seed_count is the total held-out episode count;
            # evaluate_closed_loop derives seed+7919*cell_index below.
            seeds = (self.config.evaluation_seed_start,)
            self.closed_loop_records.append({
                "update": step,
                "rows": evaluate_closed_loop(
                    model,
                    config=self.config,
                    seeds=seeds,
                    include_ablations=bool(self.config.ablations),
                    ablations=self.config.ablations,
                ),
            })

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if 0 in self.checkpoints and model is not None:
            self._evaluate(0, model)
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if step in self.checkpoints and model is not None:
            self._evaluate(step, model)
        return control


def _move_batch_to_model(
    batch: dict[str, torch.Tensor], model: torch.nn.Module
) -> dict[str, torch.Tensor]:
    """Move collated public tensors to the device hosting ``model``.

    Dataset and rollout construction intentionally stays CPU-side.  Trainer
    normally moves its training batch, but callback and closed-loop inference
    call the model directly and therefore need the same standard PyTorch
    placement explicitly for a future CUDA run.
    """

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    return {
        name: value.to(device=device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for name, value in batch.items()
    }

def run_scientific(output_dir: str | Path, config: FirstTrainingConfig) -> dict[str, Any]:
    """Run the fixed-mixture scientific path selected explicitly by the CLI."""
    if config.mode != "scientific":
        raise ValueError("run_scientific requires config mode=scientific")
    if config.purpose not in {"apparatus_verification", "source_acquisition"}:
        raise ValueError("unsupported run purpose")
    device = _resolve_device(config)
    if config.world.action_limit != 0.5 or config.world.dt != 0.1:
        raise ValueError("first-system action decoder supports only action_limit=0.5 and dt=0.1")
    if config.cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    if device.type == "cpu":
        torch.set_num_threads(config.cpu_threads)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    model = build_first_system(config).to(device)
    adapter_specs = first_system_adapter_specs(config.sensor_dims, config.action_dims)
    train_data = CalibratedReachDataset(
        TrajectoryDatasetConfig(
            size=config.dataset_size,
            seed=config.seed,
            world=config.world,
            sensor_dims=config.sensor_dims,
            action_dims=config.action_dims,
            goal_modes=config.goal_modes,
            goal_switch_steps=config.goal_switch_steps,
            disturbance_arms=config.disturbance_arms,
        ),
        adapter_specs,
    )
    heldout_data = CalibratedReachDataset(
        TrajectoryDatasetConfig(
            size=max(config.held_out_episodes, config.evaluation_seed_count, 1),
            seed=config.evaluation_seed_start,
            world=config.world,
            sensor_dims=config.sensor_dims,
            action_dims=config.action_dims,
            goal_modes=config.goal_modes,
            goal_switch_steps=config.goal_switch_steps,
            disturbance_arms=config.disturbance_arms,
        ),
        adapter_specs,
    )
    checkpoints = tuple(sorted(set(config.evaluation_checkpoints) & set(range(0, config.resumed_steps + 1))))
    callback = _FixedEvaluationCallback(heldout_data, checkpoints, config=config)
    args = _training_arguments(output_dir, config, max_steps=config.resumed_steps, save=True)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        data_collator=collate_trajectory_batch,
        callbacks=[callback],
    )
    trainer.train()
    receipt = {
        "apparatus_only": config.purpose == "apparatus_verification",
        "purpose": config.purpose,
        "effective_config": asdict(config),
        "mode": "scientific",
        "updates": int(trainer.state.global_step),
        "evaluation": callback.records,
        "closed_loop": callback.closed_loop_records,
        "seed_holdout": {"start": config.evaluation_seed_start, "count": config.evaluation_seed_count},
        "transform_holdouts_declared": {
            "sensor": [list(item) for item in config.transform_sensor_permutations],
            "body": [list(item) for item in config.transform_body_permutations],
        },
        "checkpoint": str(output_dir / f"checkpoint-{trainer.state.global_step}"),
    }
    (output_dir / "scientific_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def _digest_batch(items: list[dict[str, torch.Tensor]]) -> str:
    """Return a stable digest of one collated public batch.

    The digest is diagnostic only.  It lets the bounded resume check compare
    the actual tensor batches consumed by each Trainer without adding an
    example id or any generator metadata to the learner input.
    """
    digest = hashlib.sha256()
    for item in items:
        for name in sorted(item):
            value = item[name].detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _recording_collator(digests: list[str]):
    """Build a standard collator that records batch order outside the bundle."""

    def collator(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        digests.append(_digest_batch(items))
        return collate_trajectory_batch(items)

    return collator


def _nested_state_equal(left: Any, right: Any) -> bool:
    """Compare Trainer optimizer/scheduler state without lossy serialization."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _nested_state_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def run_cpu_smoke(output_dir: str | Path, config: FirstTrainingConfig | None = None) -> dict[str, Any]:
    """Run a tiny train/save/resume check and return apparatus evidence."""
    config = config or FirstTrainingConfig()
    if config.mixed_precision != "fp32" or config.per_device_train_batch_size != 1:
        raise ValueError("run_cpu_smoke requires per_device_train_batch_size=1 and mixed_precision=fp32")
    if config.cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(config.cpu_threads)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    model = build_first_system(config)
    dataset = CalibratedReachDataset(
        TrajectoryDatasetConfig(
            size=config.dataset_size,
            seed=config.seed,
            world=config.world,
            sensor_dims=config.sensor_dims,
            action_dims=config.action_dims,
            goal_modes=config.goal_modes,
            goal_switch_steps=config.goal_switch_steps,
            disturbance_arms=config.disturbance_arms,
        ),
        first_system_adapter_specs(config.sensor_dims, config.action_dims),
    )
    args = _training_arguments(output_dir, config, max_steps=config.resumed_steps, save=True)
    first_batch_digests: list[str] = []
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=_recording_collator(first_batch_digests),
        callbacks=[_StopAfterStep(config.smoke_steps)],
    )
    first = trainer.train()
    checkpoint = output_dir / f"checkpoint-{config.smoke_steps}"
    if not checkpoint.is_dir():
        raise AssertionError(f"Trainer did not write expected checkpoint {checkpoint}")
    del trainer

    # Seed before construction as well as before the uninterrupted branch.
    # The checkpoint replaces these parameters, but deterministic construction
    # keeps the three Trainer setups comparable and avoids hidden RNG drift.
    set_seed(config.seed)
    resumed_model = build_first_system(config)
    resumed_args = _training_arguments(
        output_dir, config, max_steps=config.resumed_steps, save=False
    )
    resumed_batch_digests: list[str] = []
    resumed = Trainer(
        model=resumed_model,
        args=resumed_args,
        train_dataset=dataset,
        data_collator=_recording_collator(resumed_batch_digests),
    )
    resumed.train(resume_from_checkpoint=str(checkpoint))
    if resumed.state.global_step != config.resumed_steps:
        raise AssertionError(
            f"resume stopped at {resumed.state.global_step}, expected {config.resumed_steps}"
        )
    # Compare the resumed weights with one uninterrupted run under the same
    # seed, optimizer, data order, and step budget.  This is a checkpoint/data
    # continuation check, not a learning result.
    set_seed(config.seed)
    uninterrupted_model = build_first_system(config)
    uninterrupted_batch_digests: list[str] = []
    uninterrupted = Trainer(
        model=uninterrupted_model,
        args=_training_arguments(
            output_dir / "uninterrupted", config, max_steps=config.resumed_steps, save=False
        ),
        train_dataset=dataset,
        data_collator=_recording_collator(uninterrupted_batch_digests),
    )
    uninterrupted.train()
    continuation_match = all(
        torch.equal(left, right)
        for (left, right) in zip(
            resumed_model.state_dict().values(), uninterrupted_model.state_dict().values()
        )
    )
    if not continuation_match:
        raise AssertionError("checkpoint resume weights differ from uninterrupted continuation")
    optimizer_match = _nested_state_equal(
        resumed.optimizer.state_dict(), uninterrupted.optimizer.state_dict()
    )
    scheduler_match = _nested_state_equal(
        resumed.lr_scheduler.state_dict(), uninterrupted.lr_scheduler.state_dict()
    )
    # Depending on the installed Transformers/Accelerate versions, the resume
    # loader either exposes only the unconsumed suffix or replays the skipped
    # prefix while restoring its sampler RNG.  Accept exactly those two forms;
    # a different sequence is a real data-continuation failure.
    data_order_match = (
        resumed_batch_digests == uninterrupted_batch_digests[config.smoke_steps:]
        or first_batch_digests + resumed_batch_digests == uninterrupted_batch_digests
    )
    if not optimizer_match or not scheduler_match:
        raise AssertionError("checkpoint optimizer/scheduler state differs from uninterrupted continuation")
    if not data_order_match:
        raise AssertionError("checkpoint resume consumed a different public batch sequence")
    return {
        "apparatus_only": True,
        "purpose": config.purpose,
        "effective_config": asdict(config),
        "first_steps": int(first.global_step),
        "resumed_steps": int(resumed.state.global_step),
        "continuation_match": continuation_match,
        "optimizer_match": optimizer_match,
        "scheduler_match": scheduler_match,
        "data_order_match": data_order_match,
        "batch_digests": {
            "first": first_batch_digests,
            "resumed": resumed_batch_digests,
            "uninterrupted": uninterrupted_batch_digests,
        },
        "checkpoint": str(checkpoint),
        "trainable_parameters": sum(
            parameter.numel() for parameter in resumed_model.parameters() if parameter.requires_grad
        ),
    }


def load_first_training_config(path: str | Path) -> FirstTrainingConfig:
    """Load the review configuration's bounded Trainer fields.

    The CPU apparatus profile uses a ``[run]`` table.  The fixed scientific
    profile uses ``[train]`` and ``[world]``; accepting both keeps this entry
    point useful for review without inventing a second configuration format.
    World-family controls remain in the dataset/world config and are not
    silently converted into learner features here.
    """
    payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    allowed_sections = {"run", "system", "world", "model", "train", "evaluation", "heldout"}
    unknown_sections = set(payload) - allowed_sections
    if unknown_sections:
        raise ValueError(f"unsupported first-system config sections: {sorted(unknown_sections)}")

    def section(name: str, allowed: set[str]) -> dict[str, Any]:
        values = payload.get(name, {})
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unsupported keys in [{name}]: {sorted(unknown)}")
        return values

    run = section(
        "run",
        {
            "seed", "dataset_size", "learning_rate", "weight_decay", "max_grad_norm",
            "smoke_steps", "resumed_steps", "max_steps", "per_device_train_batch_size",
            "mixed_precision", "mode", "purpose", "cpu_threads", "entrypoint",
            "max_wall_clock_seconds", "device",
        },
    )
    if run.get("entrypoint", "first_training") != "first_training":
        raise ValueError("first-system configs require run.entrypoint=first_training")
    system = section("system", {"version", "seed", "teacher", "online_dagger", "scheduler"})
    world_values = section(
        "world",
        {
            "train_episodes", "held_out_episodes", "horizon", "dt", "sensor_dims",
            "action_dims", "goal_modes", "goal_switch_steps", "action_limit",
            "calibration_pulse", "observation_noise", "disturbance_scale", "disturbance_arms",
        },
    )
    model = section(
        "model",
        {
            "hidden_size", "intermediate_size", "num_hidden_layers", "attention_heads",
            "max_position_embeddings", "max_tokens",
        },
    )
    train = section(
        "train", {"batch_size", "updates", "learning_rate", "weight_decay", "action_loss", "future_loss_weight"}
    )
    evaluation = section(
        "evaluation",
        {
            "checkpoints", "success_tolerance_sensor_l2", "success_tolerance_sensor_rmse", "teacher_effort_objective",
            "success_tolerance_physical", "teacher_effort_regularization", "held_out_seed_start", "held_out_seed_count",
            "metrics", "ablations", "closed_loop_checkpoints",
        },
    )
    heldout = section(
        "heldout",
        {"seed_start", "seed_count", "transform_sensor_permutations", "transform_body_permutations"},
    )
    if system.get("version", "calibrated-reach-v1") not in {"calibrated-reach-v1", "calibrated-reach-v2"}:
        raise ValueError("unsupported first-system version")
    if system.get("teacher", "public-history-calibration-teacher-v1") != "public-history-calibration-teacher-v1":
        raise ValueError("unsupported teacher")
    if system.get("online_dagger", False) is not False or system.get("scheduler", "none") != "none":
        raise ValueError("the first fixed system requires online_dagger=false and scheduler=none")
    if train.get("action_loss", "masked_smooth_l1") != "masked_smooth_l1":
        raise ValueError("the first fixed system requires action_loss=masked_smooth_l1")
    if float(train.get("future_loss_weight", 0.0)) != 0.0:
        raise ValueError("future_loss_weight must remain zero for the first system")
    if float(evaluation.get("teacher_effort_regularization", 1.0e-8)) != 1.0e-8:
        raise ValueError("teacher_effort_regularization must remain 1e-8 for the first system")
    def ints(value: Any, name: str) -> tuple[int, ...]:
        result = tuple(int(item) for item in value)
        if not result:
            raise ValueError(f"{name} must be nonempty")
        return result

    def bools(value: Any, name: str) -> tuple[bool, ...]:
        result = tuple(bool(item) for item in value)
        if not result:
            raise ValueError(f"{name} must be nonempty")
        return result

    world = CalibratedReachConfig(
        horizon=int(world_values.get("horizon", 12)),
        dt=float(world_values.get("dt", 0.1)),
        action_limit=float(world_values.get("action_limit", 0.5)),
        calibration_pulse=float(world_values.get("calibration_pulse", 0.06)),
        sensor_dims=ints(world_values.get("sensor_dims", [8, 10]), "sensor_dims"),
        action_dims=ints(world_values.get("action_dims", [2, 4]), "action_dims"),
        goal_modes=tuple(str(item) for item in world_values.get("goal_modes", ["absolute", "relative"])),
        observation_noise=float(world_values.get("observation_noise", 0.0)),
        disturbance_scale=float(world_values.get("disturbance_scale", 0.0)),
    )
    seed = int(run.get("seed", system.get("seed", 17)))
    updates = int(run.get("max_steps", train.get("updates", 2)))
    mode = str(run.get("mode", "scientific" if "train" in payload else "apparatus"))
    smoke_steps = int(run.get("smoke_steps", min(updates, 2)))
    resumed_steps = int(run.get("resumed_steps", max(smoke_steps + 1, updates)))
    if mode == "scientific":
        smoke_steps = updates
        resumed_steps = updates
    if smoke_steps <= 0 or resumed_steps < smoke_steps:
        raise ValueError("config must declare positive smoke steps and resumed_steps >= smoke_steps")
    purpose = str(run.get("purpose", "source_acquisition" if mode == "scientific" else "apparatus_verification"))
    if purpose not in {"apparatus_verification", "source_acquisition"}:
        raise ValueError("purpose must be apparatus_verification or source_acquisition")
    max_wall_clock_seconds = int(run.get("max_wall_clock_seconds", 4800))
    if max_wall_clock_seconds <= 0:
        raise ValueError("max_wall_clock_seconds must be positive")
    return FirstTrainingConfig(
        seed=seed,
        purpose=purpose,
        dataset_size=int(run.get("dataset_size", world_values.get("train_episodes", 8))),
        hidden_size=int(model.get("hidden_size", 64)),
        intermediate_size=int(model.get("intermediate_size", 128)),
        num_hidden_layers=int(model.get("num_hidden_layers", 2)),
        attention_heads=int(model.get("attention_heads", 4)),
        max_position_embeddings=int(model.get("max_position_embeddings", 128)),
        learning_rate=float(run.get("learning_rate", train.get("learning_rate", 2.0e-4))),
        weight_decay=float(run.get("weight_decay", train.get("weight_decay", 0.0))),
        per_device_train_batch_size=int(run.get("per_device_train_batch_size", train.get("batch_size", 1))),
        device=str(run.get("device", "cpu")),
        mixed_precision=str(run.get("mixed_precision", "fp32")),
        cpu_threads=int(run.get("cpu_threads", 1)),
        max_wall_clock_seconds=max_wall_clock_seconds,
        max_grad_norm=float(run.get("max_grad_norm", 1.0)),
        smoke_steps=smoke_steps,
        resumed_steps=resumed_steps,
        world=world,
        sensor_dims=tuple(world.sensor_dims),
        action_dims=tuple(world.action_dims),
        goal_modes=tuple(world.goal_modes),
        goal_switch_steps=ints(world_values.get("goal_switch_steps", [-1]), "goal_switch_steps"),
        disturbance_arms=bools(world_values.get("disturbance_arms", [False]), "disturbance_arms"),
        held_out_episodes=int(world_values.get("held_out_episodes", evaluation.get("held_out_seed_count", 0))),
        evaluation_checkpoints=ints(evaluation.get("checkpoints", [0]), "checkpoints"),
        closed_loop_checkpoints=ints(evaluation.get("closed_loop_checkpoints", [0]), "closed_loop_checkpoints"),
        success_tolerance_sensor_l2=float(evaluation.get("success_tolerance_sensor_l2", evaluation.get("success_tolerance_sensor_rmse", 0.05))),
        success_tolerance_physical=float(evaluation.get("success_tolerance_physical", 0.05)),
        teacher_effort_objective=str(
            evaluation.get("teacher_effort_objective", "bounded_residual_plus_minimum_l2")
        ),
        teacher_effort_regularization=float(evaluation.get("teacher_effort_regularization", 1.0e-8)),
        held_out_seed_start=int(heldout.get("seed_start", evaluation.get("held_out_seed_start", 0))),
        held_out_seed_count=int(heldout.get("seed_count", evaluation.get("held_out_seed_count", 0))),
        transform_sensor_permutations=tuple(
            tuple(int(item) for item in values)
            for values in heldout.get("transform_sensor_permutations", [])
        ),
        transform_body_permutations=tuple(
            tuple(int(item) for item in values)
            for values in heldout.get("transform_body_permutations", [])
        ),
        mode=mode,
        evaluation_seed_start=int(heldout.get("seed_start", evaluation.get("held_out_seed_start", 100000))),
        evaluation_seed_count=int(heldout.get("seed_count", evaluation.get("held_out_seed_count", 1))),
        ablations=tuple(str(item) for item in evaluation.get("ablations", ["no_goal", "no_calibration", "no_action_history"])),
    )


def evaluate_closed_loop(
    model: CommonContentBundle,
    *,
    config: FirstTrainingConfig,
    seeds: tuple[int, ...] = (20260909,),
    include_ablations: bool = True,
    ablations: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate policies through the public rollout API.

    The learner sees only each growing public prefix.  Physical terminal error
    and success use evaluator-only environment state, while public sensor RMS
    remains a separately reported diagnostic.  This avoids treating an
    eight-channel and ten-channel raw norm as the same task scale.
    """
    from .trajectory_data import ROLE_IDS, _event_route, _key_code
    from .trajectory_world import (
        CalibratedReachRollout,
        PublicHistoryTeacher,
        fixed_reactive_action,
    )

    adapter_codes = {
        spec.name: index
        for index, spec in enumerate(first_system_adapter_specs(config.sensor_dims, config.action_dims))
    }
    model.eval()

    def batch(events: list[Any]) -> dict[str, torch.Tensor]:
        count = len(events)
        raw = torch.zeros(
            1,
            count,
            first_system_raw_width(config.sensor_dims, config.action_dims),
            dtype=torch.float32,
        )
        adapters = torch.zeros(1, count, dtype=torch.long)
        roles = torch.zeros(1, count, dtype=torch.long)
        keys = torch.zeros(1, count, dtype=torch.long)
        times = torch.zeros(1, count, dtype=torch.float32)
        available = torch.zeros(1, count, dtype=torch.float32)
        positions = torch.arange(count, dtype=torch.long).unsqueeze(0)
        attention = torch.ones(1, count, dtype=torch.bool)
        action_mask = torch.ones(1, count, dtype=torch.bool)
        for index, event in enumerate(events):
            route, values = _event_route(event)
            raw[0, index, : len(values)] = torch.tensor(values, dtype=torch.float32)
            adapters[0, index] = adapter_codes[route]
            roles[0, index] = ROLE_IDS[event.kind]
            keys[0, index] = _key_code(event.key)
            times[0, index] = float(event.time)
            available[0, index] = float(event.available_at if event.available_at is not None else event.time)
        return {
            "raw_values": raw,
            "adapter_ids": adapters,
            "role_ids": roles,
            "key_ids": keys,
            "time_values": times,
            "available_at": available,
            "position_ids": positions,
            "attention_mask": attention,
            "action_mask": action_mask,
        }

    rows: list[dict[str, Any]] = []
    for seed_base in seeds:
        cell_index = 0
        for sensor_dim in config.sensor_dims:
            for action_dim in config.action_dims:
                for goal_mode in config.goal_modes:
                    for switch_step in config.goal_switch_steps:
                        switch = switch_step >= 0
                        for disturbance in config.disturbance_arms:
                            world = replace(
                                config.world,
                                goal_switch_step=(None if switch_step < 0 else switch_step),
                            )
                            for policy_name in ("learner", "teacher", "reactive", "inaction"):
                                cell_seed = seed_base + 7919 * cell_index
                                rollout = CalibratedReachRollout.from_seed(
                                    cell_seed,
                                    config=world,
                                    sensor_dim=sensor_dim,
                                    action_dim=action_dim,
                                    goal_mode=goal_mode,
                                    disturbance=disturbance,
                                )
                                rollout.reset()
                                actions: list[np.ndarray] = []
                                while not rollout.done:
                                    history = rollout.query()
                                    if policy_name == "learner":
                                        with torch.inference_mode(), _evaluation_autocast(config, model):
                                            output = model(**_move_batch_to_model(batch(rollout.events), model))
                                        action = output.actions[0, -1, :action_dim].detach().cpu().numpy()
                                    elif policy_name == "teacher":
                                        action = PublicHistoryTeacher(control_gain=world.control_gain).action_for(
                                            history, rollout.action_bounds
                                        )
                                    elif policy_name == "reactive":
                                        action = fixed_reactive_action(history, rollout.action_bounds)
                                    else:
                                        action = np.zeros(action_dim, dtype=np.float64)
                                    action = np.asarray(action, dtype=np.float64)
                                    if action.shape != (action_dim,) or not np.isfinite(action).all():
                                        raise ValueError("learner emitted a non-finite or wrong-width action")
                                    if np.any(action < rollout.action_bounds[:, 0]) or np.any(action > rollout.action_bounds[:, 1]):
                                        raise ValueError("learner emitted an out-of-bounds action")
                                    actions.append(action)
                                    rollout.step(action)
                                physical = rollout.privileged_metrics(
                                    success_tolerance_state_l2=config.success_tolerance_physical,
                                    success_tolerance_sensor_rmse=config.success_tolerance_sensor_l2,
                                )
                                effort = float(physical["action_effort_l2_sum"]) / max(len(actions), 1)
                                rows.append(
                                    {
                                        "seed": cell_seed,
                                        "seed_base": seed_base,
                                        "cell_index": cell_index,
                                        "body_dim": action_dim,
                                        "sensor_dim": sensor_dim,
                                        "goal_mode": goal_mode,
                                        "goal_switch": switch,
                                        "disturbance": bool(disturbance),
                                        "policy": policy_name,
                                        "public_sensor_rmse": float(physical["sensor_rmse"]),
                                        "physical_error": float(physical["state_error_l2"]),
                                        "success": bool(physical["success"]),
                                        "effort": effort,
                                    }
                                )
                            if include_ablations:
                                # Ablations are recorded only for the learner;
                                # teacher failure without its public calibration
                                # would conflate the control with the ablation.
                                for ablation in (ablations or ("no_goal", "no_calibration", "no_action_history")):
                                    rollout = CalibratedReachRollout.from_seed(
                                        cell_seed,
                                        config=world,
                                        sensor_dim=sensor_dim,
                                        action_dim=action_dim,
                                        goal_mode=goal_mode,
                                        disturbance=disturbance,
                                    )
                                    rollout.reset()
                                    actions = []
                                    while not rollout.done:
                                        rollout.query()
                                        events = list(rollout.events)
                                        if ablation == "no_goal":
                                            events = [event for event in events if event.kind != "goal"]
                                        elif ablation == "no_calibration":
                                            events = [
                                                event
                                                for event in events
                                                if event.kind not in {"calibration_action", "calibration_observation"}
                                            ]
                                        elif ablation == "no_action_history":
                                            events = [event for event in events if event.kind != "action_executed"]
                                        with torch.inference_mode(), _evaluation_autocast(config, model):
                                            output = model(**_move_batch_to_model(batch(events), model))
                                        action = np.asarray(output.actions[0, -1, :action_dim].cpu(), dtype=np.float64)
                                        if np.any(action < rollout.action_bounds[:, 0]) or np.any(action > rollout.action_bounds[:, 1]):
                                            raise ValueError("learner ablation emitted an out-of-bounds action")
                                        actions.append(action)
                                        rollout.step(action)
                                    physical = rollout.privileged_metrics(
                                        success_tolerance_state_l2=config.success_tolerance_physical,
                                        success_tolerance_sensor_rmse=config.success_tolerance_sensor_l2,
                                    )
                                    rows.append(
                                        {
                                            "seed": cell_seed,
                                            "seed_base": seed_base,
                                            "cell_index": cell_index,
                                            "body_dim": action_dim,
                                            "sensor_dim": sensor_dim,
                                            "goal_mode": goal_mode,
                                            "goal_switch": switch,
                                            "disturbance": bool(disturbance),
                                            "policy": f"learner:{ablation}",
                                            "public_sensor_rmse": float(physical["sensor_rmse"]),
                                            "physical_error": float(physical["state_error_l2"]),
                                            "success": bool(physical["success"]),
                                            "effort": float(physical["action_effort_l2_sum"]) / max(len(actions), 1),
                                        }
                                    )
                            cell_index += 1
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bounded first-system CPU apparatus check")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("apparatus", "scientific"), default=None)
    args = parser.parse_args(argv)
    config = load_first_training_config(args.config)
    if args.mode is not None and args.mode != config.mode:
        raise ValueError(f"CLI mode {args.mode!r} disagrees with config mode {config.mode!r}")
    receipt = (
        run_scientific(args.output_root, config)
        if config.mode == "scientific"
        else run_cpu_smoke(args.output_root, config)
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
