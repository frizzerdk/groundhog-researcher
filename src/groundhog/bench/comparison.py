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
                     f"{'delta(A-B)+/-sd':<20} {'A/B/tie':<8}")
        for name, ps in self.metrics.items():
            if ps.mean_delta is None:
                delta = "-"
            else:
                delta = f"{ps.mean_delta:+.4f} +/- {ps.sd_delta:.4f}"
            row = (f"  {name:<18} {_pm(ps.mean_a, ps.spread_a):<20} "
                   f"{_pm(ps.mean_b, ps.spread_b):<20} {delta:<20} "
                   f"{ps.wins_a}/{ps.wins_b}/{ps.ties}")
            if ps.only_a or ps.only_b:
                # One-sided pairs are excluded from the delta — say so.
                row += f" ({ps.only_a} A-only, {ps.only_b} B-only)"
            lines.append(row)
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
    """Pair two precomputed BenchResults; their seed lists must be identical
    (zip would silently truncate a mismatch into a wrong pairing)."""
    if a.seeds != b.seeds:
        raise ValueError(
            f"cannot pair results run on different seeds: "
            f"A ({a.config_name}) seeds={a.seeds}, "
            f"B ({b.config_name}) seeds={b.seeds}")
    metrics = {
        name: paired_stats(name, a.metric(name), b.metric(name),
                           higher_is_better=higher)
        for name, higher in METRIC_DIRECTIONS.items()
    }
    return Comparison(a=a, b=b, metrics=metrics,
                      verdict=_verdict(a, b, metrics["final_best"]))


# A single-seed win lead is noise at bench scale; the verdict only names a
# winner when the lead is at least this many seeds.
_WIN_MARGIN = 2


def _verdict(a: BenchResult, b: BenchResult, fb: PairedStats) -> str:
    delta = fb.mean_delta if fb.mean_delta is not None else 0.0
    sd = f" +/- {fb.sd_delta:.4f}" if fb.sd_delta is not None else ""
    notes = []
    if fb.only_a:
        notes.append(f"A improved on {fb.only_a}/{fb.n} seeds B did not")
    if fb.only_b:
        notes.append(f"B improved on {fb.only_b}/{fb.n} seeds A did not")
    note = f" ({'; '.join(notes)})" if notes else ""
    margin = fb.wins_a - fb.wins_b
    if margin >= _WIN_MARGIN:
        return (f"A ({a.config_name}) better on {fb.wins_a}/{fb.n} seeds, "
                f"mean {delta:+.4f}{sd} best score{note}")
    if -margin >= _WIN_MARGIN:
        return (f"B ({b.config_name}) better on {fb.wins_b}/{fb.n} seeds, "
                f"mean {-delta:+.4f}{sd} best score{note}")
    if fb.wins_a == fb.wins_b:
        return (f"A ({a.config_name}) and B ({b.config_name}) tied: "
                f"{fb.wins_a} wins each, {fb.ties} ties, "
                f"mean {delta:+.4f}{sd} best score{note}")
    return (f"no clear winner (win margin below {_WIN_MARGIN}): "
            f"A ({a.config_name}) {fb.wins_a} vs B ({b.config_name}) "
            f"{fb.wins_b} wins, {fb.ties} ties on {fb.n} seeds, "
            f"mean {delta:+.4f}{sd} best score{note}")


def _pm(mean, spread) -> str:
    if mean is None:
        return "-"
    return f"{mean:.4f} +/- {spread:.4f}"
