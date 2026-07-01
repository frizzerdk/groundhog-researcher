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
from groundhog.tools.attempt_log import AttemptLog
from groundhog.tools.attempt_logger import MarkdownAttemptLogger
from groundhog.tools.log import StrategyLog
from groundhog.tools.queue import read_next as read_queue
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
                 extras: Optional[List[Strategy]] = None,
                 seed: int = 42,
                 path: Optional[Path] = None,
                 history: Optional[AttemptHistory] = None,
                 learnings: Optional[Learnings] = None,
                 through: Optional[str] = None,
                 agent_through: Optional[str] = None,
                 seed_strategy="default",
                 direction_weight: float = 0.5,
                 direction_decay: float = 0.1,
                 exclude_non_promotable: bool = True,
                 direction_bonus: Optional[float] = None,
                 skip_non_promotable: Optional[bool] = None):
        """Configure the optimizer.

        ``extras`` registers strategies that are reachable from the queue but
        do not appear in the rotation schedule — useful for one-shot strategies
        like ``Analyse`` or queue-only ``FreshApproach`` invocations. Rotation
        wins on name collision; a warning is logged for any extra whose name
        is already taken.
        """
        from groundhog.strategies.fresh import FreshApproach
        self.task = task
        self.seed = seed
        self.through = through
        self.agent_through = agent_through
        if direction_bonus is not None:
            direction_weight = direction_bonus
        if skip_non_promotable is not None:
            exclude_non_promotable = skip_non_promotable
        self.direction_weight = direction_weight
        self.direction_decay = direction_decay
        self.exclude_non_promotable = exclude_non_promotable
        self.path = Path(path) if path else Path(".")
        self.history = history or FolderAttemptHistory(self.path)
        self.learnings = learnings or MarkdownLearnings(self.path)
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

        # Strategy registry: queue items resolve here. Rotation strategies
        # populate first; extras fill any remaining slots without overwriting.
        self._strategy_registry: Dict[str, Strategy] = {}
        for s in self._schedule:
            self._register_strategy(s)
        for s in (extras or []):
            self._register_strategy(s, allow_overwrite=False)

        # Build toolkit
        self.toolkit = Toolkit(task=self.task, history=self.history, path=self.path)
        self.toolkit.learnings = self.learnings
        self.toolkit.log = StrategyLog()
        # Per-attempt event stream: strategies emit events through
        # attempt_logger, which fans out to attemptlog.jsonl/.md and the
        # live console renderer (AttemptLog — auto-disables ANSI/heartbeat
        # on non-TTY so CI logs stay clean).
        console = AttemptLog()
        self.toolkit.attempt_log = console
        self.toolkit.attempt_logger = MarkdownAttemptLogger(console=console)
        if self.through:
            self.toolkit.through = self.through

        # Agent backends (optional — strategies check hasattr)
        from groundhog.backends.discover import auto_agent_registry
        agent_registry = auto_agent_registry()
        if agent_registry:
            self.toolkit.agent = agent_registry

        # Default agent tools from toolkit capabilities.
        # agent_through limits which eval stages the agent gets as tools
        # (e.g. "validate" for fast iteration), independent of `through`
        # which controls final scoring.
        if self.agent_through:
            self.toolkit.agent_through = self.agent_through
        from groundhog.agents.tools import build_default_agent_tools
        self.toolkit.agent_tools = build_default_agent_tools(self.toolkit)

        # Seed rng and install default prior selector. Set here (not in run())
        # so users can override before calling .run(). Anything the user can
        # customise belongs on the toolkit at construction time.
        self.toolkit.rng = random.Random(self.seed)
        scorer = self._get_scorer()
        self.toolkit.get_prior = lambda tk: select_prior(
            tk.history,
            scorer,
            tk.rng,
            direction_weight=self.direction_weight,
            direction_decay=self.direction_decay,
            exclude_non_promotable=self.exclude_non_promotable,
        )

    def _register_strategy(self, strategy: Strategy, allow_overwrite: bool = True) -> None:
        """Add a strategy to the queue-resolution registry under both its
        CamelCase-lower and snake_case names (e.g. ``FreshApproach`` registers
        as both ``"freshapproach"`` and ``"fresh_approach"``).

        With ``allow_overwrite=False`` (used for ``extras``), an existing
        registration wins and the new one is logged as skipped.
        """
        import re
        cls_name = strategy.__class__.__name__
        names = {
            cls_name.lower(),
            re.sub(r'(?<!^)(?=[A-Z])', '_', cls_name).lower(),
        }
        for name in names:
            if not allow_overwrite and name in self._strategy_registry:
                # Rotation already owns this name; keep it.
                if hasattr(self, "toolkit"):
                    self.toolkit.log.info(
                        f"[extras] skipping {cls_name}: {name!r} already registered"
                    )
                continue
            self._strategy_registry[name] = strategy

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

        print(f"{self.task.name} | {rotation} | {' ->'.join(stage_names)} | {existing} existing")

        # Print backend tiers if LLM is configured
        if hasattr(self.toolkit, 'llm'):
            from groundhog.base.backend import BackendRegistry
            reg = self.toolkit.llm
            if isinstance(reg, BackendRegistry):
                tiers = []
                for tier in ["max", "high", "default", "budget", "cheap"]:
                    b = reg.get(tier)
                    # Only show if it's not just the default fallback
                    if tier == "default" or b is not reg.get("default"):
                        tiers.append(f"{tier}={b.model}")
                print(f"LLM: {' | '.join(tiers)}")
            else:
                print(f"LLM: {reg.__class__.__name__}")
        else:
            print("LLM: none")
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
        """Print per-attempt summary via AttemptLog so the score is fresh
        (computed via the current scorer) and the metric dump compresses
        to a single highlight line. The full metrics still live in
        result.json for anyone who wants the dump."""
        cost = self._get_attempt_cost(attempt)
        result = attempt.result
        log = getattr(self.toolkit, "attempt_log", None)

        if not result.completed:
            errors = result.stages[result.failed_stage].errors
            if log is not None:
                log.attempt_failed(
                    attempt_num=attempt.id,
                    stage=result.failed_stage,
                    errors=str(errors),
                    total_cost=cost,
                    cumulative_cost=cumulative_cost,
                )
            else:
                print(f"  [{attempt.id:>3}] FAIL  {result.failed_stage}: {errors}  "
                      f"${cost:.4f} (${cumulative_cost:.4f})")
                print()
            return

        score = self._score_attempt(attempt, scorer)
        delta = score - best_score if best_score is not None else 0
        last = list(result.stages.values())[-1]

        if log is not None:
            log.attempt_done(
                attempt_num=attempt.id,
                score=score, delta=delta,
                total_cost=cost, cumulative_cost=cumulative_cost,
                summary_line=self._summary_line(last),
            )
        else:
            marker = " *" if delta > 0 else ""
            sign = "+" if delta >= 0 else ""
            print(f"  [{attempt.id:>3}] {score:.4f} ({sign}{delta:.4f}){marker}  "
                  f"${cost:.4f} (${cumulative_cost:.4f})")
            print(self._format_metrics(last))
            print()

    def _summary_line(self, stage_result, max_pairs: int = 5) -> str:
        """Compress a stage's metrics dict to a single short line.

        Picks the first ``max_pairs`` items from the dict (insertion order =
        the task's preferred ordering) and joins them with ``|``. ASCII-only
        on purpose: Windows consoles vary by codepage and ``·`` renders as
        ``?`` / ``�`` on cp437 (default cmd.exe). The full dict still
        ends up in result.json — this is just the at-a-glance summary."""
        m = stage_result.metrics
        if not m:
            return ""
        parts = []
        for k, v in list(m.items())[:max_pairs]:
            if isinstance(v, float):
                parts.append(f"{k}={v:.2f}")
            else:
                parts.append(f"{k}={v}")
        return " | ".join(parts)

    def status(self):
        """Print current optimization status — best score, attempt count, trunks."""
        scorer = self._get_scorer()
        attempts = self.history.list()
        best = self.history.best(scorer)

        print(f"{self.task.name} | {len(attempts)} attempts")
        if best:
            best_score = self._score_attempt(best, scorer)
            print(f"Best: {best_score:.4f} (#{best.id})")
        else:
            print("No successful attempts")

        # Total cost from metadata
        total_cost = sum(self._get_attempt_cost(a) for a in attempts)
        if total_cost > 0:
            print(f"Total cost: ${total_cost:.4f}")

        # Strategy counts
        strategy_counts = {}
        for a in attempts:
            s = a.metadata.get("strategy", "unknown")
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
        if strategy_counts:
            counts = ", ".join(f"{v} {k}" for k, v in strategy_counts.items())
            print(f"Strategies: {counts}")

        print()
        self._print_trunks(scorer)

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
            chain = " ->".join(f"#{a.id}" for a in trunk)
            # Show the family's core direction (1st line) from the trunk root.
            root = trunk[0]
            from groundhog.utils.direction import read_direction, direction_title
            text = read_direction(root.path) if hasattr(root, 'path') else None
            direction = direction_title(text or "")
            direction_str = f" | {direction}" if direction != "(no direction)" else ""
            print(f"  {chain} (best: {best_score:.4f}, {len(trunk)} attempts){direction_str}")
        print()

        # Direction families — orthogonal grouping (one direction may span
        # multiple trunks; one trunk may contain multiple directions if a
        # cross-pollinate child outscored a different-family parent).
        families = self.history.derive_families()
        if families and any(self._family_key(f) is not None for f in families):
            from groundhog.utils.direction import read_direction, direction_title
            print("Direction families:")
            family_rows = []
            for members in families:
                key = self._family_key(members)
                if key is None:
                    title = "(no direction)"
                else:
                    sample = read_direction(members[0].path)
                    title = direction_title(sample or "")
                best_score = max(self._score_attempt(a, scorer) for a in members)
                best_attempt = max(members, key=lambda a: self._score_attempt(a, scorer))
                family_rows.append((title, len(members), best_score, best_attempt.id))
            # Sort by best score descending; sentinel last.
            family_rows.sort(
                key=lambda r: (r[0] == "(no direction)", -r[2])
            )
            for title, count, best, best_num in family_rows:
                print(f"  [{count:>3}] best #{best_num} ({best:.4f}) | {title}")
            print()

    @staticmethod
    def _family_key(members):
        """Return the family's normalized direction key (None for sentinel)."""
        from groundhog.utils.direction import read_direction, normalize_direction
        if not members:
            return None
        text = read_direction(members[0].path) if hasattr(members[0], "path") else None
        return normalize_direction(text) if text else None

    def run(self, n: int = 10):
        scorer = self._get_scorer()
        best_score = None
        total_cost = 0.0

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
        queue_path = self.path

        for i in range(n):
            # Check queue first — override rotation if there's a queued item
            queue_item = read_queue(queue_path)
            queue_label = ""
            if queue_item:
                strategy_name = queue_item.get("strategy", "")
                strategy = self._strategy_registry.get(strategy_name)
                if strategy:
                    config = queue_item.get("config")
                    queue_label = f"{strategy_name} from {queue_item.get('source', '?')}"
                    self.toolkit.log.info(f"[queue] {queue_label}")
                else:
                    self.toolkit.log.info(f"[queue] unknown strategy: {strategy_name}, skipping")
                    strategy = next(rotation)
                    config = None
            else:
                strategy = next(rotation)
                config = None
            # Stash the queue label so AgentStrategy can pass it to
            # attempt_log.attempt_start. Empty string when not from queue.
            self.toolkit._current_queue_label = queue_label

            count_before = len(self.history.list())
            try:
                if config is not None:
                    strategy(self.toolkit, config=config)
                else:
                    strategy(self.toolkit)
            except KeyboardInterrupt:
                self.toolkit.log.end()
                print(f"\n  Interrupted by user")
                break
            except Exception as e:
                self.toolkit.log.end()
                strategy_name = strategy.__class__.__name__
                print(f"\n  [{strategy_name}] ERROR: {e}")
                continue
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
