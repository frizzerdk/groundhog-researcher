"""Paired-comparison statistics — stdlib only, no runner imports.

Kept dependency-free on purpose so in-run consumers (e.g. an A/B
meta-strategy) can import these helpers without the offline harness.
"""

from dataclasses import dataclass
from statistics import fmean, stdev
from typing import List, Optional, Sequence


def mean_std(values: Sequence) -> tuple:
    """Mean and sample stdev of the non-None values; (None, None) when empty."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    return fmean(vals), (stdev(vals) if len(vals) > 1 else 0.0)


@dataclass
class PairedStats:
    """Per-metric summary of two value sequences paired by position."""
    metric: str
    higher_is_better: bool
    n: int
    wins_a: int
    wins_b: int
    ties: int
    mean_a: Optional[float]
    mean_b: Optional[float]
    spread_a: Optional[float]
    spread_b: Optional[float]
    mean_delta: Optional[float]
    deltas: List[Optional[float]]


def paired_stats(metric: str, a_values: Sequence, b_values: Sequence,
                 higher_is_better: bool = True) -> PairedStats:
    """Compare two same-length sequences paired by position (= by seed).

    ``None`` means "no value" (e.g. never improved) and loses to any value;
    two Nones tie. Deltas (a - b) are only computed where both sides are
    present; ``mean_delta`` averages those.
    """
    wins_a = wins_b = ties = 0
    deltas: List[Optional[float]] = []
    for a, b in zip(a_values, b_values):
        if a is None and b is None:
            ties += 1
            deltas.append(None)
        elif a is None:
            wins_b += 1
            deltas.append(None)
        elif b is None:
            wins_a += 1
            deltas.append(None)
        else:
            deltas.append(a - b)
            if a == b:
                ties += 1
            elif (a > b) == higher_is_better:
                wins_a += 1
            else:
                wins_b += 1
    mean_a, spread_a = mean_std(a_values)
    mean_b, spread_b = mean_std(b_values)
    mean_delta, _ = mean_std(deltas)
    return PairedStats(
        metric=metric, higher_is_better=higher_is_better, n=len(deltas),
        wins_a=wins_a, wins_b=wins_b, ties=ties,
        mean_a=mean_a, mean_b=mean_b, spread_a=spread_a, spread_b=spread_b,
        mean_delta=mean_delta, deltas=deltas,
    )
