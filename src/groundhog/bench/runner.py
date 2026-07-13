"""Offline benchmark runner — run one BenchConfig across seeds, collect metrics.

Each seed run assembles a fresh toolkit in a throwaway temp dir, runs
``SimpleOptimizer`` for ``n_iterations``, and reduces the attempt history to
per-seed metrics (final best score, score trajectory, attempts to first
improvement, total attempts, wall time). Scores stay read-side, computed via
the task's final stage scorer — nothing new is persisted.
"""

import io
import random
import tempfile
import time
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

from groundhog.assemble import assemble_toolkit
from groundhog.optimizers.simple import SimpleOptimizer
from groundhog.tools.attempt_log import AttemptLog
from groundhog.tools.attempt_logger import MarkdownAttemptLogger
from groundhog.utils.selection import scorer_for

METRIC_DIRECTIONS = {
    "final_best": True,
    "first_improvement": False,
    "total_attempts": False,
    "wall_time_s": False,
}


@dataclass
class BenchConfig:
    """One benchmark setup.

    ``task_factory(seed)`` must return a fresh Task; the same seed also seeds
    the toolkit rng, so runs are deterministic and comparisons pair by seed.
    ``strategies`` is the SimpleOptimizer rotation schedule
    ``[(strategy, repeats), ...]`` (a bare strategy is accepted).
    ``seed_strategy`` defaults to None, which DISABLES seeding — the first
    rotation strategy runs against an empty history. (The optimizer's own
    default is the string ``"default"``, which seeds with FreshApproach and
    needs an LLM; pass that string or a strategy instance to opt in.)
    ``configure_toolkit(toolkit)`` runs after assembly and is the injection
    point for LLM-backed configs (e.g. ``toolkit.llm = auto_registry()``) —
    those work but cost money.
    """
    name: str
    task_factory: Callable[[int], object]
    strategies: Sequence
    n_iterations: int
    seed_strategy: object = None
    through: Optional[str] = None
    configure_toolkit: Optional[Callable] = None


@dataclass
class SeedRun:
    """Metrics from one seeded optimizer run."""
    seed: int
    scores: List[float]
    trajectory: List[float]
    final_best: float
    first_improvement: Optional[int]
    total_attempts: int
    wall_time_s: float

    def metric(self, name: str):
        return getattr(self, name)


@dataclass
class BenchResult:
    config_name: str
    n_iterations: int
    runs: List[SeedRun]

    @property
    def seeds(self) -> List[int]:
        return [r.seed for r in self.runs]

    def metric(self, name: str) -> list:
        return [r.metric(name) for r in self.runs]

    def summary(self) -> str:
        lines = [f"{self.config_name}: {len(self.runs)} seeds x "
                 f"{self.n_iterations} iterations"]
        lines.append(f"  {'seed':>6}  {'final_best':>10}  {'first_impr':>10}  "
                     f"{'attempts':>8}  {'wall_s':>7}")
        for r in self.runs:
            fi = "-" if r.first_improvement is None else str(r.first_improvement)
            lines.append(f"  {r.seed:>6}  {r.final_best:>10.4f}  {fi:>10}  "
                         f"{r.total_attempts:>8}  {r.wall_time_s:>7.2f}")
        if self.runs:
            from groundhog.bench.stats import mean_std
            m, s = mean_std(self.metric("final_best"))
            lines.append(f"  final_best: mean {m:.4f} +/- {s:.4f}")
        return "\n".join(lines)


def run_config(config: BenchConfig, seeds: Union[int, Sequence[int]],
               quiet: bool = True, progress: bool = False) -> BenchResult:
    """Run ``config`` once per seed; ``seeds`` is a count (0..N-1) or a list."""
    runs = []
    for seed in _normalize_seeds(seeds):
        run = _run_seed(config, seed, quiet=quiet)
        if progress:
            fi = "-" if run.first_improvement is None else run.first_improvement
            print(f"  [{config.name}] seed {seed}: best={run.final_best:.4f} "
                  f"first_improvement={fi} attempts={run.total_attempts} "
                  f"({run.wall_time_s:.2f}s)")
        runs.append(run)
    return BenchResult(config_name=config.name,
                       n_iterations=config.n_iterations, runs=runs)


def _normalize_seeds(seeds) -> List[int]:
    if isinstance(seeds, int):
        return list(range(seeds))
    return list(seeds)


def _normalize_schedule(strategies) -> list:
    if not isinstance(strategies, (list, tuple)):
        return [(strategies, 1)]
    return [item if isinstance(item, tuple) else (item, 1) for item in strategies]


def _toolkit_seed(seed: int) -> int:
    # Decorrelated from the task seed on purpose: a task built from
    # Random(seed) (e.g. MockTask's target) and a toolkit rng seeded
    # identically replay the same draw sequence — the first uniform(0,100)
    # a strategy samples would land exactly on MockTask's target.
    return random.Random(f"groundhog-bench:{seed}").randrange(2**31)


def _run_seed(config: BenchConfig, seed: int, quiet: bool = True) -> SeedRun:
    task = config.task_factory(seed)
    sink = io.StringIO()
    silence = redirect_stdout(sink) if quiet else nullcontext()
    with tempfile.TemporaryDirectory(prefix="ghg_bench_",
                                     ignore_cleanup_errors=True) as d:
        # Wall time covers the optimization itself, measured inside the
        # tempdir block — teardown (slow on Windows) is not the strategy's.
        t0 = time.perf_counter()
        with silence:
            toolkit = assemble_toolkit(task, path=Path(d),
                                       seed=_toolkit_seed(seed),
                                       through=config.through)
            if quiet:
                # AttemptLog's default stream is bound at import time, so
                # redirect_stdout alone does not silence the console renderer.
                console = AttemptLog(out=sink)
                toolkit.attempt_log = console
                toolkit.attempt_logger = MarkdownAttemptLogger(console=console)
            if config.configure_toolkit is not None:
                config.configure_toolkit(toolkit)
            optimizer = SimpleOptimizer(
                toolkit,
                strategies=_normalize_schedule(config.strategies),
                seed_strategy=config.seed_strategy,
            )
            optimizer.run(n=config.n_iterations)
        scorer = scorer_for(task, config.through)
        attempts = toolkit.history.list(only_done=False)
        scores = [_attempt_score(a, scorer) for a in attempts]
        wall = time.perf_counter() - t0

    first_improvement = None
    for i, s in enumerate(scores[1:], start=2):
        if s > scores[0]:
            first_improvement = i
            break

    return SeedRun(
        seed=seed,
        scores=scores,
        trajectory=list(accumulate(scores, max)),
        final_best=max(scores, default=-1.0),
        first_improvement=first_improvement,
        total_attempts=len(scores),
        wall_time_s=wall,
    )


def _attempt_score(attempt, scorer) -> float:
    try:
        result = attempt.result
    except Exception:
        return -1.0
    if not result.completed or not result.stages:
        return -1.0
    return scorer(list(result.stages.values())[-1])
