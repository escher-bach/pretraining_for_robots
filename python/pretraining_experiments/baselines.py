"""Pre-scheduling trivial-policy band for a world family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib
from typing import Any

import torch

import pretraining_world_py

from .data import world_kwargs
from .evaluation import error_thresholds, summarize_episode_rows, summarize_trial_rows


def scaled_oracle_baseline(
    config: dict[str, Any],
    *,
    scale: float,
    episodes: int,
) -> dict[str, Any]:
    world = config["world"]
    rollouts = pretraining_world_py.RolloutBatch(
        seed=int(world["rollout_seed"]),
        start_index=0,
        batch_size=episodes,
        max_tokens=int(config["model"]["sequence_length"]),
        **world_kwargs(world),
    )
    initial = rollouts.summary()
    initial_errors = [float(value) for value in initial["terminal_error"]]
    normalized_cost = [0.0] * episodes
    physical_cost = [0.0] * episodes
    trial_rows = [
        [0.0, initial_errors[index], 0.0, 0.0] for index in range(episodes)
    ]
    for trial in range(1, int(world["max_control_steps"]) + 1):
        oracle = rollouts.privileged_oracle_actions()
        actions = []
        for index, values in enumerate(oracle):
            action = [scale * float(value) for value in values]
            actions.append(action)
            normalized_cost[index] += sum(abs(value) for value in action)
            physical_cost[index] += float(world["action_limit"]) * sum(
                abs(value) for value in action
            )
        rollouts.step(actions)
        current = rollouts.summary()
        trial_rows.extend(
            [trial, float(current["terminal_error"][index]), normalized_cost[index], physical_cost[index]]
            for index in range(episodes)
        )
    final = rollouts.summary()
    episode_rows = torch.tensor(
        [
            [
                float(final["dimensions"][index]),
                initial_errors[index],
                float(final["terminal_error"][index]),
                float(final["steps"][index]),
                normalized_cost[index],
                physical_cost[index],
            ]
            for index in range(episodes)
        ]
    )
    thresholds = error_thresholds(config)
    result = summarize_episode_rows(episode_rows, thresholds)
    result["trial_curve"] = summarize_trial_rows(torch.tensor(trial_rows), thresholds)
    result["oracle_scale"] = scale
    return result


def baseline_band(config: dict[str, Any]) -> dict[str, Any]:
    evaluation = config.get("evaluation", {})
    scales = tuple(float(value) for value in evaluation.get("trivial_policy_scales", (0.0, 0.75, 1.0)))
    episodes = int(evaluation.get("baseline_episodes", 4096))
    if episodes <= 0 or not scales:
        raise ValueError("baseline_episodes and trivial_policy_scales must be nonempty")
    return {
        "purpose": "pre-scheduling trivial-policy band; not learner evidence",
        "episodes_per_policy": episodes,
        "policies": {
            f"scaled_oracle_{format(scale, '.6g')}": scaled_oracle_baseline(
                config, scale=scale, episodes=episodes
            )
            for scale in scales
        },
        "world_versions": pretraining_world_py.versions(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(baseline_band(config), indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
