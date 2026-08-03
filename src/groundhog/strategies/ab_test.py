"""A/B testing meta-strategy — two strategies head-to-head in one rotation slot.

Each call runs one paired trial: the toolkit's standard selector picks a
prior ONCE, then arm A and arm B each run from that same prior (paired
comparison — prior quality is controlled for). The arms are ordinary
strategies running their own full loop — select, generate, evaluate,
finalize — untouched. ABTest only pins the prior selector for the trial and
stamps ``ab_test`` / ``ab_arm`` / ``ab_pair`` metadata onto whatever the
arms commit, via the ``toolkit._extra_attempt_metadata`` pass-through the
standard finalize merges.

Per-pair isolation: the WORLD is frozen for the trial's duration. Both
arms read the history (``list()``) and the learnings as they were at
pair start, and the arm ORDER is randomized per pair (``toolkit.rng``),
so the second arm is neither contaminated by the first arm's fresh
commit/notes nor systematically the later one. Writes pass through —
every commit and learning lands in the real store.

Stale-view tradeoff, accepted deliberately: only ``list()`` (and the
learnings reads) are frozen. History members that bypass the filtered
``list`` — ``get``/``best`` bound to the real backend, note reads — see
the live store, and a fresh arm's duplicate-direction gate cannot see
the sibling arm's brand-new direction. Prior selection is safe anyway
(the pin outranks ``best``).

Fresh-style arms ignore the pinned prior (they always start from scratch),
so for them pairing degrades to same-call: time-of-run is still controlled,
prior quality is moot. Unpaired mode (``paired=False``) skips the pinning
and alternates arms across calls (balanced by CALLS, so a skipping arm
cannot starve the other); ``ab_pair`` then numbers each arm's own
trials, so the summary's "pairs" are same-round trials, not shared-prior
ones.

The scoreboard (:meth:`ABTest.summary`) is derived from attempt records on
demand — no state files. A round where only one arm committed (the other
skipped or failed to record) is an INCOMPLETE PAIR: excluded from the
paired stats, reported via ``incomplete_pairs`` and the per-arm
``skips_a``/``skips_b`` counts and ``skip_rate_a``/``skip_rate_b``.
"""

import random as rand_module
from dataclasses import dataclass

from groundhog.base.strategy import Strategy, StrategyConfig, param
from groundhog.bench.stats import paired_stats
from groundhog.learnings.markdown import SEPARATOR


class _FrozenHistoryView:
    """Read view pinned to the attempt ids present at pair start.

    ``list()`` filters to the frozen set; everything else (writes like
    ``workspace``/``resume``/``set_note``, plus ``get``/``best``) passes
    through to the live history. See the module docstring for the
    stale-view tradeoff.
    """

    def __init__(self, history, frozen_ids):
        self._history = history
        self._frozen_ids = frozen_ids

    def list(self, only_done=True):
        return [a for a in self._history.list(only_done=only_done)
                if a.id in self._frozen_ids]

    def __getattr__(self, name):
        return getattr(self._history, name)


class _FrozenLearnings:
    """Learnings view serving the entries present at pair start.

    ``get``/``count`` read the frozen snapshot (same last/random sampling
    contract as MarkdownLearnings); ``add`` and everything else pass
    through, so an arm's notes land in the real store without the sibling
    arm reading them mid-pair.
    """

    def __init__(self, learnings):
        self._learnings = learnings
        text = learnings.get() if hasattr(learnings, "get") else ""
        self._entries = [e.strip() for e in (text or "").split(SEPARATOR)
                         if e.strip()]

    def get(self, last=0, random=0):
        entries = self._entries
        if not entries:
            return ""
        if last <= 0 and random <= 0:
            return SEPARATOR.join(entries)
        recent = entries[-last:] if last > 0 else []
        older = entries[:-last] if last > 0 else list(entries)
        sampled = []
        if random > 0 and older:
            sampled = rand_module.sample(older, min(random, len(older)))
        return SEPARATOR.join(sampled + recent)

    def count(self):
        return len(self._entries)

    def __getattr__(self, name):
        return getattr(self._learnings, name)


@dataclass
class ABTestConfig(StrategyConfig):
    """Configuration for the ABTest meta-strategy."""
    pairs_per_call: int = param(1, "Trials per call: paired A+B runs, or single alternating-arm runs when unpaired")
    min_pairs: int = param(5, "Completed pairs required before a verdict line is logged")
    paired: bool = param(True, "Pin one prior for both arms per trial; False alternates arms instead")
    test_name: str = param("", "ab_test metadata value; default '<a.name>-vs-<b.name>'")


class ABTest(Strategy):
    """Run two strategies as arms of an A/B test, one paired trial per call.

    Composed method pattern:
        init -> [per trial: next pair -> select prior once -> pin -> arm A -> arm B] -> summary -> verdict
    """

    Config = ABTestConfig
    name = "ab_test"  # derived would be "a_b_test"

    def __init__(self, strategy_a: Strategy, strategy_b: Strategy, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.strategy_a = strategy_a
        self.strategy_b = strategy_b

    def __call__(self, toolkit, config=None):
        self._init(toolkit, config)
        trials = []
        for _ in range(self.cfg.pairs_per_call):
            if self.cfg.paired:
                trials.append(self._run_pair(toolkit))
            else:
                trials.append(self._run_next_arm(toolkit))
        report = self.summary(toolkit)
        verdict = self._log_verdict(report)
        return {"strategy": self.name, "test": self._test_name(),
                "trials": trials, "summary": report, "verdict": verdict}

    # --- Init ---

    def _init(self, toolkit, config):
        from groundhog.tools.log import StrategyLog
        self.cfg = self._resolve_config(config)
        self.through = getattr(toolkit, 'through', None)
        self.log = toolkit.log if hasattr(toolkit, 'log') else StrategyLog()

    def _test_name(self):
        return self.cfg.test_name or f"{self.strategy_a.name}-vs-{self.strategy_b.name}"

    # --- Paired trial ---

    def _run_pair(self, toolkit):
        pair = self._next_pair(toolkit)
        prior = self._select_prior(toolkit)
        unpin = self._pin_prior(toolkit, prior)
        unfreeze = self._freeze_world(toolkit)
        trial = {"pair": pair}
        try:
            for arm, strategy in self._arm_order(toolkit):
                trial[arm] = self._run_arm(toolkit, strategy, arm, pair)
        finally:
            unfreeze()
            unpin()
        return trial

    def _arm_order(self, toolkit):
        """Randomize which arm runs first, per pair (``toolkit.rng`` when
        present) — otherwise the same arm is systematically the later,
        possibly resource-warmer run."""
        arms = [("a", self.strategy_a), ("b", self.strategy_b)]
        rng = getattr(toolkit, "rng", None) or rand_module.Random()
        rng.shuffle(arms)
        return arms

    def _freeze_world(self, toolkit):
        """Freeze the pair's world: both arms read history and learnings as
        they were at pair start; writes pass through. object.__setattr__
        skips Toolkit's override print — a scoped swap restored in the
        caller's finally, not a configuration change."""
        swapped = []
        history = getattr(toolkit, "history", None)
        if history is not None:
            frozen_ids = {a.id for a in history.list(only_done=False)}
            object.__setattr__(
                toolkit, "history", _FrozenHistoryView(history, frozen_ids))
            swapped.append(("history", history))
        learnings = getattr(toolkit, "learnings", None)
        if learnings is not None:
            object.__setattr__(toolkit, "learnings", _FrozenLearnings(learnings))
            swapped.append(("learnings", learnings))

        def unfreeze():
            for name, original in swapped:
                object.__setattr__(toolkit, name, original)

        return unfreeze

    def _select_prior(self, toolkit):
        if hasattr(toolkit, 'get_prior'):
            return toolkit.get_prior(toolkit)
        if hasattr(toolkit, 'history') and hasattr(toolkit, 'task'):
            stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=self.through)
            return toolkit.history.best(stages[-1].score)
        return None

    def _pin_prior(self, toolkit, prior):
        """Pin ``toolkit.get_prior`` to one pick for the trial's duration.

        Without the pin, arm A's freshly committed attempt could become the
        best and arm B would build on A's output instead of the shared prior.
        object.__setattr__ skips Toolkit's override print — this is a scoped
        swap restored in the caller's finally, not a configuration change.
        """
        had = hasattr(toolkit, 'get_prior')
        original = getattr(toolkit, 'get_prior', None)
        object.__setattr__(toolkit, 'get_prior', lambda tk: prior)

        def unpin():
            if had:
                object.__setattr__(toolkit, 'get_prior', original)
            else:
                delattr(toolkit, 'get_prior')

        return unpin

    # --- Unpaired trial ---

    def _run_next_arm(self, toolkit):
        # Balance by CALLS, not committed trials: a skipping arm commits
        # nothing, and commit-count balancing re-picked it forever. Call
        # counts live on the instance, seeded once from history so a
        # resumed run continues balanced.
        if not hasattr(self, "_arm_calls"):
            self._arm_calls = {"a": 0, "b": 0}
            for meta in self._test_metadata(toolkit):
                arm = meta.get("ab_arm")
                if arm in self._arm_calls:
                    self._arm_calls[arm] += 1
        arm = "a" if self._arm_calls["a"] <= self._arm_calls["b"] else "b"
        self._arm_calls[arm] += 1
        strategy = self.strategy_a if arm == "a" else self.strategy_b
        pair = self._arm_calls[arm]
        return {"pair": pair, arm: self._run_arm(toolkit, strategy, arm, pair)}

    # --- Arm execution ---

    def _run_arm(self, toolkit, strategy, arm, pair):
        # Save/restore like _pin_prior: a wrapping producer (nested ABTest,
        # a session) may have its own pass-through metadata pinned.
        saved = getattr(toolkit, "_extra_attempt_metadata", None)
        toolkit._extra_attempt_metadata = {
            "ab_test": self._test_name(), "ab_arm": arm, "ab_pair": pair,
        }
        try:
            return strategy(toolkit)
        finally:
            toolkit._extra_attempt_metadata = saved

    # --- Scoreboard (read-side, derived from history) ---

    def summary(self, toolkit):
        """Per-arm stats derived from attempt records carrying this test's key.

        Read-side entry point — callable without a prior ``__call__``.
        """
        if not hasattr(self, "cfg"):
            self._init(toolkit, None)
        by_arm = {"a": {}, "b": {}}
        for attempt, meta in self._test_attempts(toolkit):
            arm, pair = meta.get("ab_arm"), meta.get("ab_pair")
            if arm in by_arm and pair is not None:
                by_arm[arm][pair] = self._score_attempt(toolkit, attempt)
        pairs = sorted(set(by_arm["a"]) & set(by_arm["b"]))
        stats = paired_stats(
            "score",
            [by_arm["a"][p] for p in pairs],
            [by_arm["b"][p] for p in pairs],
        )
        # A round with only one arm recorded (the other skipped) is an
        # incomplete pair: excluded from the paired stats above, reported
        # here so attrition is visible instead of silently vanishing.
        rounds = set(by_arm["a"]) | set(by_arm["b"])
        skips_a = len(rounds - set(by_arm["a"]))
        skips_b = len(rounds - set(by_arm["b"]))
        return {
            "test": self._test_name(),
            "arm_a": self.strategy_a.name,
            "arm_b": self.strategy_b.name,
            "trials_a": len(by_arm["a"]),
            "trials_b": len(by_arm["b"]),
            "pairs": stats.n,
            "incomplete_pairs": skips_a + skips_b,
            "skips_a": skips_a,
            "skips_b": skips_b,
            "skip_rate_a": skips_a / len(rounds) if rounds else 0.0,
            "skip_rate_b": skips_b / len(rounds) if rounds else 0.0,
            "wins_a": stats.wins_a,
            "wins_b": stats.wins_b,
            "ties": stats.ties,
            "mean_a": stats.mean_a,
            "mean_b": stats.mean_b,
            "mean_delta": stats.mean_delta,
        }

    def _next_pair(self, toolkit):
        used = [m.get("ab_pair", 0) for m in self._test_metadata(toolkit)]
        return max(used, default=0) + 1

    def _test_attempts(self, toolkit):
        history = getattr(toolkit, 'history', None)
        if history is None:
            return []
        name = self._test_name()
        found = []
        for a in history.list(only_done=False):
            if a.status == "in-progress":
                continue
            meta = a.metadata
            if meta.get("ab_test") == name:
                found.append((a, meta))
        return found

    def _test_metadata(self, toolkit):
        return [meta for _, meta in self._test_attempts(toolkit)]

    def _score_attempt(self, toolkit, attempt):
        result = attempt.result
        if not result.completed:
            return None
        stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=self.through)
        for stage in reversed(stages):
            stage_result = result.stages.get(stage.name)
            if stage_result is not None:
                return stage.score(stage_result)
        return None

    # --- Verdict ---

    def _log_verdict(self, report):
        if report["pairs"] < self.cfg.min_pairs:
            return None
        n, wins_a, wins_b = report["pairs"], report["wins_a"], report["wins_b"]
        delta = report["mean_delta"]
        mean = f"{delta:+.3f}" if delta is not None else "n/a"
        if wins_a == wins_b:
            line = f"ABTest {report['test']}: tied on {n} pairs, mean {mean}"
        else:
            leader = "A" if wins_a > wins_b else "B"
            line = (f"ABTest {report['test']}: {leader} better on "
                    f"{max(wins_a, wins_b)}/{n} pairs, mean {mean}")
        self.log.info(line)
        return line
