"""CPU world/oracle and Python-boundary throughput measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import tomllib
from typing import Any

import torch

import pretraining_world_py

from .data import tensorize, world_kwargs


def _rate(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
    }


def run_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    preflight = config["preflight"]
    world = config["world"]
    model = config["model"]
    total = int(preflight["cpu_benchmark_episodes"])
    batch_size = int(preflight["cpu_benchmark_batch_size"])
    repeats = int(preflight["cpu_benchmark_repeats"])
    sequence_length = int(model["sequence_length"])
    batches = (total + batch_size - 1) // batch_size

    generation_rates: list[float] = []
    token_rates: list[float] = []
    conversion_rates: list[float] = []
    last_raw: dict[str, Any] | None = None
    for repeat in range(repeats + 1):
        started = time.perf_counter()
        episodes = 0
        tokens = 0
        for batch_index in range(batches):
            current = min(batch_size, total - episodes)
            raw = pretraining_world_py.generate_training_batch(
                seed=int(world["train_seed"]),
                start_index=repeat * 10_000_000 + batch_index * batch_size,
                batch_size=current,
                max_tokens=sequence_length,
                **world_kwargs(world),
            )
            episodes += current
            tokens += sum(raw["lengths"])
            last_raw = raw
        elapsed = time.perf_counter() - started
        if repeat:
            generation_rates.append(episodes / elapsed)
            token_rates.append(tokens / elapsed)

    assert last_raw is not None
    for repeat in range(repeats + 1):
        started = time.perf_counter()
        converted = None
        for _ in range(100):
            converted = tensorize(last_raw)
        elapsed = time.perf_counter() - started
        if repeat:
            conversion_rates.append(100 * len(last_raw["lengths"]) / elapsed)
        del converted

    return {
        "episodes": total,
        "batch_size": batch_size,
        "repeats": repeats,
        "generated_episodes_per_second": _rate(generation_rates),
        "generated_unpadded_tokens_per_second": _rate(token_rates),
        "existing_batch_tensor_conversions_per_second": _rate(conversion_rates),
        "torch_version": torch.__version__,
        "world_versions": pretraining_world_py.versions(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_benchmark(config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
