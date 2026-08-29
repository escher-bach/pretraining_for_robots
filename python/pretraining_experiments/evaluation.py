"""Decision-facing metrics for online world interaction."""

from __future__ import annotations

from typing import Any, Iterable

import torch


DEFAULT_ERROR_THRESHOLDS = (0.01, 0.025, 0.05, 0.1, 0.2, 0.4)


def error_thresholds(config: dict[str, Any]) -> tuple[float, ...]:
    values = config.get("evaluation", {}).get("error_thresholds", DEFAULT_ERROR_THRESHOLDS)
    thresholds = tuple(sorted({float(value) for value in values}))
    if not thresholds or thresholds[0] <= 0:
        raise ValueError("evaluation error thresholds must be positive")
    return thresholds


def threshold_key(value: float) -> str:
    return format(value, ".6g")


def distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().cpu()
    if values.numel() == 0:
        return {key: float("nan") for key in ("mean", "p10", "p25", "p50", "p75", "p90", "p95", "max")}
    quantiles = torch.quantile(
        values,
        torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90, 0.95]),
    )
    return {
        "mean": float(values.mean().item()),
        "p10": float(quantiles[0].item()),
        "p25": float(quantiles[1].item()),
        "p50": float(quantiles[2].item()),
        "p75": float(quantiles[3].item()),
        "p90": float(quantiles[4].item()),
        "p95": float(quantiles[5].item()),
        "max": float(values.max().item()),
    }


def summarize_episode_rows(
    rows: torch.Tensor,
    thresholds: Iterable[float],
    *,
    include_by_dimension: bool = True,
) -> dict[str, Any]:
    """Summarize rows: dimension, initial/final error, steps, two action costs."""

    rows = rows.detach().float().cpu()
    if rows.ndim != 2 or rows.shape[1] != 6:
        raise ValueError("episode rows must have shape [episodes, 6]")
    initial = rows[:, 1]
    final = rows[:, 2]
    scale = initial.clamp_min(torch.finfo(initial.dtype).eps)
    threshold_values = tuple(float(value) for value in thresholds)
    threshold_success = {
        threshold_key(value): float((final <= value).float().mean().item())
        for value in threshold_values
    }
    result: dict[str, Any] = {
        "episodes": int(rows.shape[0]),
        "terminal_error": float(final.mean().item()),
        "terminal_error_distribution": distribution(final),
        "initial_error_distribution": distribution(initial),
        "threshold_success": threshold_success,
        "success_rate": threshold_success.get("0.05", float("nan")),
        "mean_steps": float(rows[:, 3].mean().item()),
        "mean_error_reduction": float((initial - final).mean().item()),
        "mean_fractional_error_reduction": float(((initial - final) / scale).mean().item()),
        "mean_normalized_action_l1": float(rows[:, 4].mean().item()),
        "mean_physical_action_l1": float(rows[:, 5].mean().item()),
    }
    if include_by_dimension:
        result["by_dimension"] = {}
        for dimension in sorted({int(value) for value in rows[:, 0].tolist()}):
            selected = rows[rows[:, 0] == float(dimension)]
            result["by_dimension"][str(dimension)] = summarize_episode_rows(
                selected,
                threshold_values,
                include_by_dimension=False,
            )
    return result


def summarize_trial_rows(rows: torch.Tensor, thresholds: Iterable[float]) -> list[dict[str, Any]]:
    """Summarize rows: trial index, current error, normalized/physical cost."""

    rows = rows.detach().float().cpu()
    if rows.ndim != 2 or rows.shape[1] != 4:
        raise ValueError("trial rows must have shape [records, 4]")
    curve = []
    for trial in sorted({int(value) for value in rows[:, 0].tolist()}):
        selected = rows[rows[:, 0] == float(trial)]
        errors = selected[:, 1]
        curve.append(
            {
                "actions_allowed": trial,
                "error": distribution(errors),
                "threshold_success": {
                    threshold_key(float(value)): float((errors <= float(value)).float().mean().item())
                    for value in thresholds
                },
                "mean_cumulative_normalized_action_l1": float(selected[:, 2].mean().item()),
                "mean_cumulative_physical_action_l1": float(selected[:, 3].mean().item()),
            }
        )
    return curve


def paired_learning_delta(
    untrained: dict[str, Any],
    trained: dict[str, Any],
) -> dict[str, Any]:
    """Contrast a trained learner with its own step-zero weights.

    Both arguments must come from the same held-out support. A positive
    ``improvement`` means the trained learner is better on that metric; the
    sign convention is fixed here so the reported numbers cannot be read
    backwards.
    """

    before_loop = untrained["closed_loop"]
    after_loop = trained["closed_loop"]
    before_regression = untrained["validation"]
    after_regression = trained["validation"]
    if before_loop["episodes"] != after_loop["episodes"]:
        raise ValueError("paired comparison requires the same held-out episode count")

    return {
        "episodes": int(after_loop["episodes"]),
        "terminal_error": {
            "untrained": before_loop["terminal_error"],
            "trained": after_loop["terminal_error"],
            "improvement": before_loop["terminal_error"] - after_loop["terminal_error"],
        },
        "mean_fractional_error_reduction": {
            "untrained": before_loop["mean_fractional_error_reduction"],
            "trained": after_loop["mean_fractional_error_reduction"],
            "improvement": after_loop["mean_fractional_error_reduction"]
            - before_loop["mean_fractional_error_reduction"],
        },
        "threshold_success": {
            key: {
                "untrained": before_loop["threshold_success"][key],
                "trained": after_loop["threshold_success"][key],
                "improvement": after_loop["threshold_success"][key]
                - before_loop["threshold_success"][key],
            }
            for key in sorted(after_loop["threshold_success"])
            if key in before_loop["threshold_success"]
        },
        "mean_normalized_action_l1": {
            "untrained": before_loop["mean_normalized_action_l1"],
            "trained": after_loop["mean_normalized_action_l1"],
        },
        "teacher_forced_action_l1": {
            "untrained": before_regression["action_l1"],
            "trained": after_regression["action_l1"],
            "improvement": before_regression["action_l1"] - after_regression["action_l1"],
        },
        "teacher_forced_future_l1": {
            "untrained": before_regression["future_l1"],
            "trained": after_regression["future_l1"],
            "improvement": before_regression["future_l1"] - after_regression["future_l1"],
        },
    }
