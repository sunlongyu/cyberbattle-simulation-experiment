from __future__ import annotations

from typing import Iterable, List, Optional
import math


def _to_float_list(values: Iterable[float]) -> List[float]:
    result: List[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric):
            continue
        result.append(numeric)
    return result


def compute_reward_volatility(values: Iterable[float]) -> float:
    series = _to_float_list(values)
    if len(series) < 2:
        return 0.0
    mean_value = sum(series) / len(series)
    variance = sum((value - mean_value) ** 2 for value in series) / (len(series) - 1)
    return math.sqrt(max(variance, 0.0))


def compute_convergence_step(
    rewards: Iterable[float],
    threshold_ratio: float = 0.95,
    stability_window: int = 20,
    final_window: int = 30,
) -> Optional[int]:
    """Return the first 1-based episode index that satisfies the convergence rule."""

    series = _to_float_list(rewards)
    if not series:
        return None
    final_slice = series[-min(final_window, len(series)) :]
    final_mean = sum(final_slice) / len(final_slice)
    # Use a symmetric "within 5% of the final mean" threshold that still works
    # when rewards are negative. Direct multiplication would make the criterion
    # stricter than the final mean for negative returns.
    target = final_mean - (1.0 - threshold_ratio) * abs(final_mean)
    if stability_window <= 0:
        stability_window = 1
    for start_idx in range(len(series)):
        end_idx = start_idx + stability_window
        if end_idx > len(series):
            break
        window = series[start_idx:end_idx]
        if all(value >= target for value in window):
            return start_idx + 1
    return None
