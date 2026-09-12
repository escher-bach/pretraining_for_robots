"""Tensor-only public trajectory data for the first CPU training system.

The generated world's private audit never enters this module's tensors.  The
adapter routing code is a modality dispatch field consumed inside
``CommonContentBundle``; it is removed before the temporal core sees content.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from itertools import product
from typing import Sequence

import torch
from torch.utils.data import Dataset

from .common_content import AdapterSpec, EmbodimentActionSpec
from .trajectory_world import (
    CalibratedReachConfig,
    PublicEvent,
    TrajectoryEpisode,
    generate_episode,
)


# 8/10 are the current fixture widths.  The tensor envelope is derived from
# the declared sensor/body family so another supported sensor width does not
# require changing the learner schema.
MAX_RAW_WIDTH = 10
MAX_ACTION_DIM = 4

ROLE_IDS = {
    "reset": 1,
    "observation": 2,
    "goal": 3,
    "calibration_action": 4,
    "calibration_observation": 5,
    "action_query": 6,
    "action_executed": 7,
    "episode_end": 8,
}


def first_system_raw_width(
    sensor_dims: Sequence[int] = (8, 10), action_dims: Sequence[int] = (2, 4)
) -> int:
    """Return the packed envelope width for one declared family mixture."""
    widths = [*map(int, sensor_dims), *map(int, action_dims), 1]
    if not widths or any(width <= 0 for width in widths):
        raise ValueError("declared raw widths must be positive")
    return max(widths)


def first_system_adapter_specs(
    sensor_dims: Sequence[int] = (8, 10), action_dims: Sequence[int] = (2, 4)
) -> list[AdapterSpec]:
    """Build adapters from the declared sensor/body family cross-product."""
    sensor_widths = tuple(dict.fromkeys(int(width) for width in sensor_dims))
    action_widths = tuple(dict.fromkeys(int(width) for width in action_dims))
    if not sensor_widths or not action_widths:
        raise ValueError("sensor_dims and action_dims must be nonempty")
    return [
        *(AdapterSpec(f"sensor{width}", "sensor", width) for width in sensor_widths),
        *(AdapterSpec(f"goal{width}", "goal", width) for width in sensor_widths),
        *(AdapterSpec(f"action{width}", "past_action", width) for width in action_widths),
        AdapterSpec("control1", "control", 1),
    ]


def first_system_action_spec() -> EmbodimentActionSpec:
    """Four-output decoder with validity masks for the two-actuator body."""
    return EmbodimentActionSpec(
        action_dim=MAX_ACTION_DIM,
        names=("actuator_0", "actuator_1", "actuator_2", "actuator_3"),
        units=("normalized",) * MAX_ACTION_DIM,
        coordinate_frame="body_local",
        semantics=("delta",) * MAX_ACTION_DIM,
        lower_bounds=(-0.5,) * MAX_ACTION_DIM,
        upper_bounds=(0.5,) * MAX_ACTION_DIM,
        control_dt=0.1,
    )


def _key_code(key: str, *, size: int = 255) -> int:
    """Stable public-key address; no family or generator identifier."""
    if not key:
        return 0
    return 1 + sum((index + 1) * ord(char) for index, char in enumerate(key)) % size


def _event_route(event: PublicEvent) -> tuple[str, list[float]]:
    if event.kind in {"observation", "calibration_observation"}:
        return f"sensor{len(event.values)}", list(event.values)
    if event.kind == "goal":
        return f"goal{len(event.values)}", list(event.values)
    if event.kind in {"calibration_action", "action_executed"}:
        return f"action{len(event.action)}", list(event.action)
    # A boundary marker is content-bearing only as a declared control token;
    # it is not a fabricated eight-float semantic observation.
    return "control1", [1.0]


def _encode_episode(
    episode: TrajectoryEpisode,
    adapter_codes: dict[str, int],
    raw_width: int,
) -> dict[str, torch.Tensor]:
    target_by_event = {target.query_event_id: target for target in episode.supervision}
    token_count = len(episode.events)
    raw_values = torch.zeros(token_count, raw_width, dtype=torch.float32)
    adapter_ids = torch.zeros(token_count, dtype=torch.long)
    role_ids = torch.zeros(token_count, dtype=torch.long)
    key_ids = torch.zeros(token_count, dtype=torch.long)
    time_values = torch.zeros(token_count, dtype=torch.float32)
    available_at = torch.zeros(token_count, dtype=torch.float32)
    position_ids = torch.arange(token_count, dtype=torch.long)
    attention_mask = torch.ones(token_count, dtype=torch.bool)
    action_targets = torch.zeros(token_count, MAX_ACTION_DIM, dtype=torch.float32)
    action_target_mask = torch.zeros(token_count, MAX_ACTION_DIM, dtype=torch.bool)
    for index, event in enumerate(episode.events):
        route, values = _event_route(event)
        if route not in adapter_codes:
            raise ValueError(f"no registered adapter for public route {route!r}")
        if len(values) > raw_width:
            raise ValueError("public event exceeds the first system raw width")
        raw_values[index, : len(values)] = torch.tensor(values, dtype=torch.float32)
        adapter_ids[index] = adapter_codes[route]
        try:
            role_ids[index] = ROLE_IDS[event.kind]
        except KeyError as exc:
            raise ValueError(f"unrecognized public event kind {event.kind!r}") from exc
        key_ids[index] = _key_code(event.key)
        time_values[index] = float(event.time)
        # Availability is a public structural timestamp.  It remains
        # separate from physical event time so delayed/asynchronous sensor
        # records cannot be mistaken for observations available at capture.
        available_at[index] = float(
            event.available_at if event.available_at is not None else event.time
        )
        target = target_by_event.get(event.event_id)
        if target is not None:
            action_dim = len(target.action)
            if action_dim > MAX_ACTION_DIM:
                raise ValueError("action supervision exceeds max body width")
            action_targets[index, :action_dim] = torch.tensor(target.action, dtype=torch.float32)
            action_target_mask[index, :action_dim] = torch.tensor(
                target.action_valid, dtype=torch.bool
            )
    return {
        "raw_values": raw_values,
        "adapter_ids": adapter_ids,
        "role_ids": role_ids,
        "key_ids": key_ids,
        "time_values": time_values,
        "available_at": available_at,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "action_mask": action_target_mask.any(dim=-1),
        "action_targets": action_targets,
        "action_target_mask": action_target_mask,
    }


def collate_trajectory_batch(items: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad tensor-only episodes for the standard PyTorch/Trainer collator."""
    if not items:
        raise ValueError("cannot collate an empty trajectory batch")
    max_tokens = max(int(item["raw_values"].shape[0]) for item in items)
    batch_size = len(items)
    result = {
        "raw_values": torch.zeros(batch_size, max_tokens, max(
            int(item["raw_values"].shape[-1]) for item in items
        )),
        "adapter_ids": torch.zeros(batch_size, max_tokens, dtype=torch.long),
        "role_ids": torch.zeros(batch_size, max_tokens, dtype=torch.long),
        "key_ids": torch.zeros(batch_size, max_tokens, dtype=torch.long),
        "time_values": torch.zeros(batch_size, max_tokens),
        "available_at": torch.zeros(batch_size, max_tokens),
        "position_ids": torch.zeros(batch_size, max_tokens, dtype=torch.long),
        "attention_mask": torch.zeros(batch_size, max_tokens, dtype=torch.bool),
        "action_mask": torch.zeros(batch_size, max_tokens, dtype=torch.bool),
        "action_targets": torch.zeros(batch_size, max_tokens, MAX_ACTION_DIM),
        "action_target_mask": torch.zeros(
            batch_size, max_tokens, MAX_ACTION_DIM, dtype=torch.bool
        ),
    }
    for row, item in enumerate(items):
        length = int(item["raw_values"].shape[0])
        for name, value in item.items():
            if name == "raw_values":
                result[name][row, :length, : value.shape[-1]] = value
            else:
                result[name][row, :length] = value
    return result


@dataclass(frozen=True)
class TrajectoryDatasetConfig:
    # ``size`` is a support size, not a small replay cohort.  Scientific
    # profiles use a large value so every Trainer index maps to a fresh,
    # deterministic generated instance.
    size: int = 16
    seed: int = 0
    world: CalibratedReachConfig = CalibratedReachConfig()
    sensor_dims: tuple[int, ...] = (8, 10)
    action_dims: tuple[int, ...] = (2, 4)
    goal_modes: tuple[str, ...] = ("absolute", "relative")
    goal_switch_steps: tuple[int, ...] = (-1,)
    disturbance_arms: tuple[bool, ...] = (False,)
    # Names are process-family contracts.  The default preserves the admitted
    # calibrated reach family; the compiler-backed world can add composed
    # families without changing the tensor envelope or Trainer.
    families: tuple[str, ...] = ("calibrated_reach_v2",)
    episode_seed_stride: int = 7919
    compiled_processes_path: str | None = None


class CalibratedReachDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic on-demand episodes with a fixed declared mixture."""

    def __init__(self, config: TrajectoryDatasetConfig, adapter_specs: Sequence[AdapterSpec]):
        if config.size <= 0:
            raise ValueError("dataset size must be positive")
        if not config.goal_switch_steps or any(
            step < -1 or step >= config.world.horizon for step in config.goal_switch_steps
        ):
            raise ValueError("goal_switch_steps must contain -1 or in-horizon steps")
        if not config.disturbance_arms:
            raise ValueError("disturbance_arms must be nonempty")
        if not config.families or any(not str(family) for family in config.families):
            raise ValueError("families must be nonempty and named")
        if config.episode_seed_stride <= 0:
            raise ValueError("episode_seed_stride must be positive")
        self.config = config
        self.adapter_codes = {spec.name: index for index, spec in enumerate(adapter_specs)}
        self.raw_width = first_system_raw_width(config.sensor_dims, config.action_dims)
        self.compiled_processes: tuple[dict[str, object], ...] = ()
        if config.compiled_processes_path is not None:
            from .process_runtime import load_compiled_processes

            self.compiled_processes = load_compiled_processes(config.compiled_processes_path)
            if len(self.compiled_processes) < 8:
                raise ValueError("compiled process export must contain the eight legal process compositions")
        compiled_mixture = all(str(family).startswith("compiled_") for family in config.families)
        cell_switch_steps = (-1,) if compiled_mixture else config.goal_switch_steps
        cell_disturbance_arms = (False,) if compiled_mixture else config.disturbance_arms
        base_cells = tuple(
            product(
                config.sensor_dims,
                config.action_dims,
                config.goal_modes,
                cell_switch_steps,
                cell_disturbance_arms,
            )
        )
        # Keep the historical single-family ``cells`` ABI (several audit
        # receipts inspect it directly), while exposing family in every
        # multi-family cell used by the compiler-backed profile.
        self.cells = (
            base_cells
            if config.families == ("calibrated_reach_v2",)
            else tuple(
                (family, *cell)
                for family in config.families
                for cell in base_cells
            )
        )
        if not self.cells:
            raise ValueError("the fixed mixture must contain at least one factorial cell")
        required = {
            *(f"sensor{width}" for width in config.sensor_dims),
            *(f"goal{width}" for width in config.sensor_dims),
            *(f"action{width}" for width in config.action_dims),
            "control1",
        }
        for required in required:
            if required not in self.adapter_codes:
                raise ValueError(f"adapter specs omit required first-system route {required!r}")

    def __len__(self) -> int:
        return self.config.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        # The index is the only data-order state.  Repeated access regenerates
        # identical public tensors and remains independent of worker count.
        seed = self.config.seed + self.config.episode_seed_stride * index
        selected = self.cells[index % len(self.cells)]
        if len(selected) == 5:
            family = "calibrated_reach_v2"
            sensor_dim, action_dim, goal_mode, switch_step, disturbance = selected
        else:
            family, sensor_dim, action_dim, goal_mode, switch_step, disturbance = selected
        world = replace(
            self.config.world,
            goal_switch_step=None if switch_step < 0 else switch_step,
        )
        # The compiler world owns family semantics.  Keep this adapter narrow:
        # it asks for a public episode and never receives generator metadata.
        # The fallback keeps historical configs runnable while the new
        # compiler registers additional family names in trajectory_world.
        if family == "calibrated_reach_v2":
            episode = generate_episode(
                seed,
                config=world,
                sensor_dim=sensor_dim,
                action_dim=action_dim,
                goal_mode=goal_mode,
                disturbance=disturbance,
            )
        elif family in {
            "compiled_reaching", "compiled_actuator_lag", "compiled_goal_switch",
            "compiled_disturbance", "compiled_actuator_lag_goal_switch",
            "compiled_actuator_lag_disturbance", "compiled_goal_switch_disturbance",
            "compiled_composed", "compiled_lag_goal_switch_disturbance", "composed_reach_v1",
        }:
            from .process_runtime import ProcessIR, generate_composed_episode

            # Family programs are Rust-generated once per launch.  Their
            # graph, wiring, and lowered runtime values are authoritative;
            # Python only supplies the seed-specific realization.
            if self.compiled_processes:
                wanted = {
                    "compiled_reaching": "Reaching",
                    "compiled_actuator_lag": "ActuatorLag",
                    "compiled_goal_switch": "GoalSwitch",
                    "compiled_disturbance": "Disturbance",
                    "compiled_actuator_lag_goal_switch": "ActuatorLagGoalSwitch",
                    "compiled_actuator_lag_disturbance": "ActuatorLagDisturbance",
                    "compiled_goal_switch_disturbance": "GoalSwitchDisturbance",
                    "compiled_lag_goal_switch_disturbance": "LagGoalSwitchDisturbance",
                }.get(family)
                candidates = []
                for item in self.compiled_processes:
                    process = item["process"]
                    enum_family = str(process.get("family", "")).split("::")[-1]
                    if wanted is not None and enum_family != wanted:
                        continue
                    if wanted is None and enum_family in {"Reaching", "ActuatorLag"}:
                        continue
                    if int(process.get("sensor_dim", -1)) != sensor_dim or int(process.get("action_dim", -1)) != action_dim:
                        continue
                    candidates.append(process)
                if not candidates:
                    raise ValueError(
                        f"compiled export has no {family} program for sensor={sensor_dim}, action={action_dim}"
                    )
                compiled = candidates[index % len(candidates)]
                ir = ProcessIR.from_compiled_process(compiled)
            else:
                if family.startswith("compiled_"):
                    raise ValueError("compiled process families require a Rust process export")
                # Historical compatibility path for old local profiles.
                components = ["reach", "actuator_lag"]
                if switch_step >= 0:
                    components.append("goal_switch")
                if disturbance:
                    components.append("disturbance")
                ir = ProcessIR(
                    name="compiled-reach-composition",
                    components=tuple(components),
                    horizon=world.horizon,
                    dt=world.dt,
                    goal_switch_step=(None if switch_step < 0 else switch_step),
                    disturbance_scale=(world.disturbance_scale if disturbance else 0.0),
                )
            episode = generate_composed_episode(
                seed,
                ir=ir,
                sensor_dim=sensor_dim,
                action_dim=action_dim,
                goal_mode=goal_mode,
            )
        else:
            raise ValueError(f"trajectory family {family!r} is unavailable in this world build")
        return _encode_episode(episode, self.adapter_codes, self.raw_width)
