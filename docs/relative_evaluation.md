# Relative evaluation

How to optimize a task that has **no absolute metric** — a game-playing policy,
a negotiation agent, a generative style, anything where "good" only means
"beats the others". A candidate is scored by *comparison* against a pool of
prior candidates, not by a fixed number.

The pattern reuses everything groundhog already gives you (immutable history,
read-side scoring, trunks). The only new pieces are:

- `groundhog.utils.relative` — pure helpers (`round_robin`, `elo_update`, …)
  that turn a pairwise *play function* into a win-rate or an Elo rating.
- an **Evaluator that reads `toolkit.history`** to find its opponents.

Worked, runnable example: [`docs/examples/rps_relative/`](examples/rps_relative/).

## The core idea

A normal Evaluator maps one candidate to a number:

```
E(candidate) -> score
```

A relative Evaluator can't — there is no scale. It maps a candidate *and a
reference pool* to a number:

```
E(candidate, pool) -> win_rate   # or Elo, or Bradley-Terry, …
```

The pool is the set of opponents the candidate plays. Groundhog already stores
every prior candidate in the immutable attempt history, so the pool is *free* —
it is `toolkit.history`, filtered to a reference set.

Because groundhog **never persists scores** (they are always recomputed
read-side), relative scoring fits the grain of the framework: change the pool or
switch win-rate → Elo and the entire history re-ranks with no re-runs of the
expensive part, exactly as swapping a stage `scorer` does for absolute tasks.

## How an Evaluator reaches `toolkit.history`

The `Evaluator.evaluate(code_or_path, data)` signature has no `toolkit`
argument on purpose — base stays minimal, and most evaluators are pure. A
relative evaluator needs the history, so **the run dir injects it in
`build_toolkit()`**, after the toolkit is assembled:

```python
def build_toolkit() -> Toolkit:
    tk = assemble_toolkit(task, path=HERE, through="tournament")
    task.evaluator.pool = tk.history      # <-- the one wire that makes it relative
    return tk
```

`assemble_toolkit` guarantees `tk.history` exists (it defaults to a
`FolderAttemptHistory` rooted at `path`), and per-task tools already close over
`toolkit.history` the same way — so this is an established seam, not a new hole
in the abstraction. The evaluator holds the reference and, at eval time, draws
opponents from it:

```python
class TournamentEvaluator(Evaluator):
    pool = None          # set by build_toolkit(); None until wired

    def evaluate(self, code_or_path, data):
        candidate = load(code_or_path)
        opponents = self._reference_pool()          # from self.pool.list()
        r = round_robin(candidate, opponents, play_fn=self._play,
                        games_per_pair=data.get_test()["games"])
        return StageResult(metrics={"win_rate": r["win_rate"],
                                    "per_opponent": r["per_opponent"]})
```

The candidate being evaluated is **not yet in history** (it is still an
uncommitted workspace), so it never plays itself — the pool is exactly the
committed opponents. On the very first attempt the pool is empty; seed it with a
fixed set of built-in baselines so the first candidate still has something to
beat (see [Mitigations](#pitfalls-and-mitigations)).

The stage `scorer` then reads `win_rate` (or the Elo rating) out of the metrics,
just like any other stage — no special casing in the optimizer.

## The helpers (`groundhog.utils.relative`)

Pure functions, zero framework coupling — they never import groundhog, never
touch disk, and don't know what a "candidate" is. You supply a **play function**
`play_fn(candidate, opponent) -> float` that runs one game and returns the
*first* player's score: `1.0` win, `0.5` draw, `0.0` loss (fractional scores such
as a best-of-N win share are fine).

```python
from groundhog.utils.relative import round_robin, elo_update, elo_rating

# Win-rate over a pool
r = round_robin(candidate, opponents, play_fn=play, games_per_pair=4)
r["win_rate"]      # mean score across all games
r["per_opponent"]  # {opponent_key: mean_score} — see which rivals it loses to
r["games"]         # total games played

# Elo: fold a batch of (opponent_rating, score) games into a rating
rating = elo_rating(1500, [(opp_rating, score) for ...], k=32)
# …or one step at a time
rating = elo_update(rating, opp_rating, score, k=32)
```

`round_robin` keys `per_opponent` by list index by default; pass
`key=lambda a: a.id` to label rivals by their attempt id.

**Win-rate vs. Elo** — which to store as the metric:

| | win-rate over a pool | Elo rating |
|---|---|---|
| reads as | share of games won | a rating on a scale |
| needs | a fixed reference pool | opponent ratings to update against |
| strength | simple, bounded `[0,1]`, poolable | absorbs many games, comparable across time |
| weakness | meaningless if the pool shifts | drifts; sensitive to game order & schedule |

A common setup: store **win-rate against a fixed reference pool** as the primary
metric (stable, re-anchorable), and keep Elo as a secondary read for tracking
progress across a long run.

## Pitfalls and mitigations

Relative evaluation is noisier and less stable than an absolute metric. The four
failure modes that actually bite, and what to do:

### 1. Non-transitivity (rock-paper-scissors cycles)
A beats B, B beats C, C beats A — there is no total order, so "best" is
ill-defined and a single incumbent can be gamed by a counter-strategy.
- **Mitigation:** score against a **diverse pool**, not a single champion.
  A win-rate over many opponents can't be won by countering just one. Keep
  losers in the pool (groundhog already keeps every attempt) so styles they
  exploited stay represented.

### 2. Rating drift
Elo ratings inflate/deflate over a long run as the population changes; a "1600"
early means something different from a "1600" late.
- **Mitigation:** anchor to a **fixed reference pool** with known ratings and
  re-derive candidate ratings against it, rather than chaining updates through
  an ever-shifting field. Prefer win-rate-vs-fixed-pool as the *decision* metric
  and treat live Elo as display-only.

### 3. Incumbent staleness
If the reference pool is frozen forever, candidates overfit to beating a stale
set and stop improving in any real sense.
- **Mitigation:** **periodic re-anchor** — every N attempts, refresh the
  reference pool from the current trunk leaders (`history.derive_trunks(scorer)`)
  while keeping a few permanent baselines for continuity. Re-anchoring re-scores
  history for free (scores aren't persisted); it never invalidates stored data.

### 4. Seed / seat variance
One game is mostly noise: first-move advantage, a lucky RNG seed, or an
unbalanced map can swing a single match.
- **Mitigation:** play **N-game batches** (`games_per_pair`) and average, and
  use **symmetric seats** — play each pairing twice with the players swapped (or
  with mirrored seeds) and average both, so seat advantage cancels. Fix RNG
  seeds per pairing so the comparison is apples-to-apples across candidates.

## Selection for tournament tasks

`toolkit.get_prior` picks which attempt the next iteration builds on. The
default potential-weighted selection works on the derived win-rate scorer out of
the box. If you track a **live rating**, you can replace `get_prior` with a
rating-based pick — this is the sanctioned reason to swap the function rather
than just tune the `SelectionPolicy` data (see `base/toolkit.py`).

Trunks still apply: a trunk is a chain where each step *out-scores* its parent
under the scorer, so with a win-rate scorer a trunk is a genuine
beat-your-predecessor lineage — the natural shape of a tournament ladder.
