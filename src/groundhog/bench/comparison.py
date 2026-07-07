"""Paired A/B comparison of two bench configs over the same seeds."""

from dataclasses import dataclass
from typing import Dict, Sequence, Union

from groundhog.bench.runner import (
    METRIC_DIRECTIONS, BenchConfig, BenchResult, _normalize_seeds, run_config,
)
from groundhog.bench.stats import PairedStats, paired_stats


@dataclass
class Comparison:
    a: BenchResult
    b: BenchResult
    metrics: Dict[str, PairedStats]
    verdict: str

    def format(self) -> str:
        n = len(self.a.runs)
        if self.a.n_iterations == self.b.n_iterations:
            iters = str(self.a.n_iterations)
        else:
            iters = f"{self.a.n_iterations} vs {self.b.n_iterations}"
        lines = [f"A: {self.a.config_name} | B: {self.b.config_name} | "
                 f"{n} seeds x {iters} iterations"]
        lines.append(f"  {'metric':<18} {'A mean+/-sd':<20} {'B mean+/-sd':<20} "
                     f"{'delta(A-B)':<11} {'A/B/tie':<8}")
        for name, ps in self.metrics.items():
            delta = "-" if ps.mean_delta is None else f"{ps.mean_delta:+.4f}"
            lines.append(f"  {name:<18} {_pm(ps.mean_a, ps.spread_a):<20} "
                         f"{_pm(ps.mean_b, ps.spread_b):<20} {delta:<11} "
                         f"{ps.wins_a}/{ps.wins_b}/{ps.ties}")
        lines.append(f"verdict: {self.verdict}")
        return "\n".join(lines)


def compare(config_a: BenchConfig, config_b: BenchConfig,
            seeds: Union[int, Sequence[int]],
            quiet: bool = True, progress: bool = False) -> Comparison:
    """Run both configs on the SAME seeds and pair the results per seed."""
    seed_list = _normalize_seeds(seeds)
    result_a = run_config(config_a, seed_list, quiet=quiet, progress=progress)
    result_b = run_config(config_b, seed_list, quiet=quiet, progress=progress)
    return compare_results(result_a, result_b)


def compare_results(a: BenchResult, b: BenchResult) -> Comparison:
    """Pair two precomputed BenchResults (runs must share seed order)."""
    metrics = {
        name: paired_stats(name, a.metric(name), b.metric(name),
                           higher_is_better=higher)
        for name, higher in METRIC_DIRECTIONS.items()
    }
    return Comparison(a=a, b=b, metrics=metrics,
                      verdict=_verdict(a, b, metrics["final_best"]))


def _verdict(a: BenchResult, b: BenchResult, fb: PairedStats) -> str:
    delta = fb.mean_delta if fb.mean_delta is not None else 0.0
    if fb.wins_a > fb.wins_b:
        return (f"A ({a.config_name}) better on {fb.wins_a}/{fb.n} seeds, "
                f"mean {delta:+.4f} best score")
    if fb.wins_b > fb.wins_a:
        return (f"B ({b.config_name}) better on {fb.wins_b}/{fb.n} seeds, "
                f"mean {-delta:+.4f} best score")
    return (f"A ({a.config_name}) and B ({b.config_name}) tied: "
            f"{fb.wins_a} wins each, {fb.ties} ties, "
            f"mean {delta:+.4f} best score")


def _pm(mean, spread) -> str:
    if mean is None:
        return "-"
    return f"{mean:.4f} +/- {spread:.4f}"
