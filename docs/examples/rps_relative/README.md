# RPS tournament — relative evaluation worked example

A minimal run dir for a task with **no absolute metric**. The candidate is a
rock-paper-scissors policy; there is no "correct score", so the Evaluator scores
each candidate by its **win-rate against a reference pool** — the fixed baseline
strategy files in `baselines/` plus every prior candidate in `toolkit.history`.

Read [`docs/relative_evaluation.md`](../../relative_evaluation.md) for the full
pattern (win-rate vs. Elo, pitfalls, mitigations). This directory is the
concrete wiring.

## What to look at

- **`task.py`** — the whole pattern:
  - `TournamentEvaluator.evaluate()` plays the candidate against the pool with
    `groundhog.utils.relative.round_robin`, using **symmetric seats** (each
    pairing played twice with players swapped) and **N-game batches** to damp
    seat/seed variance.
  - `build_toolkit()` contains the one wire that makes it relative:
    `task.evaluator.pool = tk.history`. That is how a pure Evaluator reaches the
    optimizer's memory of prior candidates.
  - The stage `scorer` reads `win_rate` out of the raw metrics — scores stay
    read-side and un-persisted, so switching the pool or the metric re-ranks
    history for free.
- **`baselines/*.py`** — the fixed reference anchors (always-rock, cycle,
  beat-last). Each is a strategy file exposing `move(my_moves, opp_moves)`.

## Run the offline demo (no LLM, no API calls)

```bash
uv run task.py
```

It scores each baseline against the pool and prints per-opponent results — e.g.
`beat_last` demolishes `always_rock` (it plays the counter every round) but only
ties `cycle`. No opponent wins every matchup, so ranking depends entirely on
*which* rivals are in the pool. That pool-sensitivity — and the intransitive
cycles that appear once smarter policies enter — is the headline pitfall of
relative evaluation, and the reason the metric is a win-rate over a diverse pool
rather than a duel against one champion.

## Wiring it to a real optimizer

`build_toolkit()` returns a ready toolkit. To actually optimize, add an
`auto_registry()` for LLM/agent backends and a `SimpleOptimizer` in `__main__`,
exactly like `examples/02_recommended/task.py` — the only task-specific part is
the relative Evaluator already in `task.py`. As candidates commit to history the
pool grows, so later candidates are judged against a stronger, more diverse
field. Periodically re-anchor the pool to the current trunk leaders
(`history.derive_trunks(scorer)`) to keep it from going stale.
