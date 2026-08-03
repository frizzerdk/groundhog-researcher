"""Relative evaluation — score a candidate by comparison, not by an absolute metric.

Some tasks have no fixed number to maximize: a game-playing policy, a negotiator,
a generative style. "Good" only means "better than the others". These helpers turn
a *play function* — one that pits two candidates against each other — into a
win-rate over a reference pool, or an Elo rating.

Everything here is pure and framework-free: nothing imports groundhog, nothing
touches disk, nothing knows what a candidate *is*. A candidate and an opponent are
whatever ``play_fn`` understands (a code string, a callable, a path). See
``docs/relative_evaluation.md`` for how an Evaluator feeds these from
``toolkit.history``.

Score convention (used throughout): a game returns the *first* player's score in
``[0, 1]`` — ``1.0`` win, ``0.5`` draw, ``0.0`` loss. Fractional scores (e.g. a
best-of-N win share) are fine; the mean is still meaningful.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Tuple


def round_robin(
    candidate: Any,
    opponents: Iterable[Any],
    play_fn: Callable[[Any, Any], float],
    games_per_pair: int = 1,
    key: Optional[Callable[[Any], Any]] = None,
) -> Dict[str, Any]:
    """Play ``candidate`` against every opponent and report its win-rate.

    ``play_fn(candidate, opponent)`` returns the candidate's score for one game
    (``1.0`` / ``0.5`` / ``0.0`` — see the module docstring). Each pairing is
    played ``games_per_pair`` times and averaged, so a stochastic game gets a
    steadier estimate.

    Returns ``{"win_rate", "games", "per_opponent"}`` where ``win_rate`` is the
    candidate's mean score across *all* games, ``games`` the total played, and
    ``per_opponent`` maps each opponent to its mean score. Opponent keys are the
    list index by default; pass ``key`` (e.g. ``lambda a: a.id``) for stable
    labels. An empty pool yields ``win_rate == 0.0`` and no games — the caller
    decides what a candidate with no one to beat is worth.
    """
    if games_per_pair < 1:
        raise ValueError("games_per_pair must be >= 1")

    opponents = list(opponents)
    per_opponent: Dict[Any, float] = {}
    total = 0.0
    games = 0
    for i, opp in enumerate(opponents):
        label = key(opp) if key is not None else i
        pair_total = 0.0
        for _ in range(games_per_pair):
            pair_total += play_fn(candidate, opp)
        per_opponent[label] = pair_total / games_per_pair
        total += pair_total
        games += games_per_pair

    return {
        "win_rate": total / games if games else 0.0,
        "games": games,
        "per_opponent": per_opponent,
    }


def expected_score(rating: float, opponent_rating: float) -> float:
    """Elo expectation: the probability ``rating``'s player scores against
    ``opponent_rating``, on the standard logistic 400-point scale. A 400-point
    lead predicts a ~0.91 score; equal ratings predict 0.5.
    """
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


def elo_update(
    rating: float,
    opponent_rating: float,
    score: float,
    k: float = 32.0,
) -> float:
    """One Elo step. ``score`` is the actual outcome for ``rating``'s player
    (``1.0`` / ``0.5`` / ``0.0``); ``k`` is the update sensitivity (32 is the
    common default — smaller settles slower but drifts less). Returns the new
    rating. The opponent's symmetric loss (``-k * (score - expected)``) is the
    caller's to apply if they track a live rating for opponents too.
    """
    return rating + k * (score - expected_score(rating, opponent_rating))


def elo_rating(
    rating: float,
    results: Iterable[Tuple[float, float]],
    k: float = 32.0,
) -> float:
    """Fold a batch of games into a single rating by sequential Elo updates.

    ``results`` is an iterable of ``(opponent_rating, score)`` pairs. Updates are
    order-sensitive (each game shifts the rating the next expectation is measured
    against), so pass games in the order they were played. Returns the final
    rating; the per-step trail is available via :func:`elo_update` if you need it.
    """
    r = rating
    for opponent_rating, score in results:
        r = elo_update(r, opponent_rating, score, k=k)
    return r
