from __future__ import annotations

import math
import random
from typing import Callable, Iterable, Sequence


def paired_bootstrap_ci(
    deltas: Sequence[float],
    num_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 13,
) -> dict[str, float]:
    if not deltas:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}
    rng = random.Random(seed)
    means = []
    for _ in range(num_samples):
        sample = [deltas[rng.randrange(len(deltas))] for _ in range(len(deltas))]
        means.append(sum(sample) / len(sample))
    means.sort()
    low_idx = max(0, int(math.floor((alpha / 2) * len(means))) - 1)
    high_idx = min(len(means) - 1, int(math.ceil((1 - alpha / 2) * len(means))) - 1)
    return {
        "mean": sum(deltas) / len(deltas),
        "low": means[low_idx],
        "high": means[high_idx],
    }


def paired_permutation_test(
    deltas: Sequence[float],
    num_samples: int = 5000,
    seed: int = 13,
) -> dict[str, float]:
    if not deltas:
        return {"observed_mean": 0.0, "p_value": 1.0}
    rng = random.Random(seed)
    observed = abs(sum(deltas) / len(deltas))
    exceed = 0
    for _ in range(num_samples):
        permuted = [delta if rng.random() < 0.5 else -delta for delta in deltas]
        statistic = abs(sum(permuted) / len(permuted))
        if statistic >= observed:
            exceed += 1
    return {"observed_mean": sum(deltas) / len(deltas), "p_value": (exceed + 1) / (num_samples + 1)}


def win_tie_loss(deltas: Sequence[float], tolerance: float = 1e-12) -> dict[str, int]:
    wins = sum(1 for delta in deltas if delta > tolerance)
    losses = sum(1 for delta in deltas if delta < -tolerance)
    ties = len(deltas) - wins - losses
    return {"wins": wins, "ties": ties, "losses": losses}
