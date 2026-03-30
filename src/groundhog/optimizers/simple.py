"""Simple optimizer: weighted strategy rotation, potential-based prior selection."""

import random
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from groundhog.base.types import Task
from groundhog.base.strategy import Strategy
from groundhog.base.optimizer import Optimizer
from groundhog.base.attempt_history import AttemptHistory
from groundhog.histories.folder import FolderAttemptHistory
from groundhog.base.toolkit import Toolkit
from groundhog.base.learnings import Learnings
from groundhog.learnings.markdown import MarkdownLearnings
from groundhog.tools.log import StrategyLog
from groundhog.utils.selection import select_prior


class SimpleOptimizer(Optimizer):
    """Runs strategies in weighted rotation with potential-based prior selection.

    Accepts either a single strategy or a list of (strategy, repeats) tuples
    that define a rotation schedule. The schedule cycles — e.g. 14 Improve +
    5 CrossPollinate + 1 Fresh = 20 per cycle.

    Prior selection uses potential scoring across trunk leaders: high-scoring
    trunks are favored but short/unexplored trunks get an exploration bonus.
    Set via toolkit.get_prior.

    At end of run, prints trunk summary showing improvement chains.
    """

    def __init__(self, task: Task,
                 strategy: Union[Strategy, None] = None,
                 strategies: Optional[List[Tuple[Strategy, int]]] = None,
                 seed: int = 42,
                 history: Optional[AttemptHistory] = None,
                 learnings: Optional[Learnings] = None,
                 through: Optional[str] = None,
                 seed_strategy="default"):
        from groundhog.strategies.fresh import FreshApproach
        self.task = task
        self.seed = seed
        self.through = through
        self.history = history or FolderAttemptHistory(Path(task.name))
        self.learnings = learnings or MarkdownLearnings(Path(task.name))
        self.seed_strategy = FreshApproach() if seed_strategy == "default" else seed_strategy

        # Build rotation schedule from strategies or single strategy
        if strategies:
            self._schedule = []
            for strat, repeats in strategies:
                self._schedule.extend([strat] * repeats)
        elif strategy:
            self._schedule = [strategy]
        else:
            from groundhog.strategies.improve import Improve
            self._schedule = [Improve()]

        # Build toolkit
        self.toolkit = Toolkit(task=self.task, history=self.history)
        self.toolkit.learnings = self.learnings
        self.toolkit.log = StrategyLog()
        if self.through:
            self.toolkit.through = self.through

    def _get_scorer(self):
        stages = self.task.evaluator.eval_stages(self.task.data, through=self.through)
        return stages[-1].score

    def _print_header(self):
        stages = self.task.evaluator.eval_stages(self.task.data, through=self.through)
        stage_names = [s.name for s in stages]
        existing = len(self.history.list())

        # Summarize strategy rotation
        strat_counts = {}
        for s in self._schedule:
            name = s.__class__.__name__
            strat_counts[name] = strat_counts.get(name, 0) + 1
        rotation = " + ".join(f"{count}x{name}" for name, count in strat_counts.items())

        print(f"{self.task.name} | {rotation} | {' → '.join(stage_names)} | {existing} existing")
        print()

    def _score_attempt(self, attempt, scorer):
        if not attempt.result.completed:
            return -1.0
        last = list(attempt.result.stages.values())[-1]
        return scorer(last)

    INDENT = "         "
    MAX_WIDTH = 100

    def _format_metrics(self, stage_result):
        m = stage_result.metrics
        parts = []
        for k, v in m.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.2f}")
            else:
                parts.append(f"{k}={v}")
        lines = []
        line = self.INDENT
        for part in parts:
            if len(line) + len(part) + 1 > self.MAX_WIDTH and line != self.INDENT:
                lines.append(line)
                line = self.INDENT
            line += part + " "
        if line.strip():
            lines.append(line.rstrip())
        return "\n".join(lines)

    def _get_attempt_cost(self, attempt):
        try:
            return attempt.metadata.get("cost", 0.0)
        except Exception:
            return 0.0

    def _log_attempt(self, attempt, scorer, best_score, cumulative_cost):
        cost = self._get_attempt_cost(attempt)
        result = attempt.result
        if not result.completed:
            errors = result.stages[result.failed_stage].errors
            print(f"  [{attempt.number:3d}] FAIL  {result.failed_stage}: {errors}  ${cost:.4f} (${cumulative_cost:.4f})")
            print()
            return

        score = self._score_attempt(attempt, scorer)
        delta = score - best_score if best_score is not None else 0
        marker = " *" if delta > 0 else ""
        sign = "+" if delta >= 0 else ""
        print(f"  [{attempt.number:3d}] {score:.4f} ({sign}{delta:.4f}){marker}  ${cost:.4f} (${cumulative_cost:.4f})")

        last = list(result.stages.values())[-1]
        print(self._format_metrics(last))
        print()

    def _print_trunks(self, scorer):
        trunks = self.history.derive_trunks(scorer)
        if not trunks:
            return

        # Sort by best score descending
        scored_trunks = []
        for trunk in trunks:
            best = max(self._score_attempt(a, scorer) for a in trunk)
            scored_trunks.append((trunk, best))
        scored_trunks.sort(key=lambda t: t[1], reverse=True)

        print("Trunks:")
        for trunk, best_score in scored_trunks:
            chain = " → ".join(f"#{a.number}" for a in trunk)
            print(f"  {chain} (best: {best_score:.4f}, {len(trunk)} attempts)")
        print()

    def run(self, n: int = 10):
        self.toolkit.rng = random.Random(self.seed)
        scorer = self._get_scorer()
        best_score = None
        total_cost = 0.0

        # Set up potential-based prior selection on toolkit
        self.toolkit.get_prior = lambda tk: select_prior(tk.history, scorer, tk.rng)

        self._print_header()

        # Check existing history for current best
        best = self.history.best(scorer)
        if best:
            best_score = self._score_attempt(best, scorer)
            print(f"Resuming from best: {best_score:.4f}")
            print()

        # Seed with fresh approach if no history
        if not best and self.seed_strategy:
            print("Seeding with fresh approach...")
            log = self.seed_strategy(self.toolkit)
            self.toolkit.log.end()
            if log.get("skipped"):
                print(f"  Seed skipped: {log['skipped']}")
            else:
                best = self.history.best(scorer)
                if best:
                    best_score = self._score_attempt(best, scorer)
                    cost = self._get_attempt_cost(best)
                    total_cost += cost
                    self._log_attempt(best, scorer, best_score, total_cost)
            print()

        # Rotation: cycle through schedule
        rotation = cycle(self._schedule)

        for i in range(n):
            strategy = next(rotation)
            count_before = len(self.history.list())
            strategy(self.toolkit)
            self.toolkit.log.end()

            # Some strategies (Analyse) don't create attempts
            attempts = self.history.list()
            if len(attempts) > count_before:
                latest = attempts[-1]
                cost = self._get_attempt_cost(latest)
                total_cost += cost
                self._log_attempt(latest, scorer, best_score, total_cost)

                # Update best
                current_best = self.history.best(scorer)
                if current_best:
                    current_score = self._score_attempt(current_best, scorer)
                    if best_score is None or current_score > best_score:
                        best_score = current_score

        # Print trunk summary
        self._print_trunks(scorer)

        best_str = f"Best: {best_score:.4f}" if best_score is not None else "No successful attempts"
        print(f"{best_str}  Total cost: ${total_cost:.4f}")
