"""Relative-evaluation helpers (utils/relative.py): pure, framework-free.

The helpers turn a pairwise play function into a win-rate or an Elo rating.
These tests pin the score convention (first player's outcome in [0,1]), the
aggregation, and the Elo math — no groundhog imports, no disk.
"""

import pytest

from groundhog.utils.relative import (
    elo_rating,
    elo_update,
    expected_score,
    round_robin,
)


# --- round_robin -------------------------------------------------------------

def test_round_robin_all_wins():
    r = round_robin("c", ["a", "b", "c"], play_fn=lambda _c, _o: 1.0)
    assert r["win_rate"] == 1.0
    assert r["games"] == 3
    assert r["per_opponent"] == {0: 1.0, 1: 1.0, 2: 1.0}


def test_round_robin_all_losses():
    r = round_robin("c", ["a", "b"], play_fn=lambda _c, _o: 0.0)
    assert r["win_rate"] == 0.0
    assert r["games"] == 2


def test_round_robin_mixed_mean():
    scores = {"a": 1.0, "b": 0.0, "c": 0.5}
    r = round_robin("x", ["a", "b", "c"], play_fn=lambda _c, o: scores[o])
    assert r["win_rate"] == pytest.approx(0.5)
    assert r["per_opponent"] == {0: 1.0, 1: 0.0, 2: 0.5}


def test_round_robin_games_per_pair_averages():
    # Alternating win/loss across the 4 games per pair -> mean 0.5 per opponent.
    seq = iter([1.0, 0.0, 1.0, 0.0])
    r = round_robin("x", ["only"], play_fn=lambda _c, _o: next(seq),
                    games_per_pair=4)
    assert r["games"] == 4
    assert r["per_opponent"][0] == pytest.approx(0.5)
    assert r["win_rate"] == pytest.approx(0.5)


def test_round_robin_key_labels():
    class Opp:
        def __init__(self, id):
            self.id = id

    opps = [Opp("7_a"), Opp("9_b")]
    r = round_robin("x", opps, play_fn=lambda _c, _o: 1.0, key=lambda o: o.id)
    assert set(r["per_opponent"]) == {"7_a", "9_b"}


def test_round_robin_empty_pool():
    r = round_robin("x", [], play_fn=lambda _c, _o: 1.0)
    assert r["win_rate"] == 0.0
    assert r["games"] == 0
    assert r["per_opponent"] == {}


def test_round_robin_rejects_zero_games():
    with pytest.raises(ValueError):
        round_robin("x", ["a"], play_fn=lambda _c, _o: 1.0, games_per_pair=0)


def test_round_robin_plays_candidate_as_first_arg():
    seen = []
    round_robin("ME", ["a", "b"],
                play_fn=lambda c, o: seen.append((c, o)) or 1.0)
    assert seen == [("ME", "a"), ("ME", "b")]


# --- Elo ---------------------------------------------------------------------

def test_expected_score_equal_ratings():
    assert expected_score(1500, 1500) == pytest.approx(0.5)


def test_expected_score_symmetry():
    # Expectations of a matchup must sum to 1.
    assert expected_score(1600, 1400) + expected_score(1400, 1600) == \
        pytest.approx(1.0)


def test_expected_score_400_point_lead():
    assert expected_score(1800, 1400) == pytest.approx(1 / (1 + 10 ** -1), abs=1e-9)


def test_elo_update_win_raises_rating():
    new = elo_update(1500, 1500, score=1.0, k=32)
    assert new == pytest.approx(1516.0)  # 1500 + 32*(1 - 0.5)


def test_elo_update_loss_lowers_rating():
    new = elo_update(1500, 1500, score=0.0, k=32)
    assert new == pytest.approx(1484.0)


def test_elo_update_draw_between_equals_is_noop():
    assert elo_update(1500, 1500, score=0.5, k=32) == pytest.approx(1500.0)


def test_elo_update_expected_win_barely_moves():
    # Beating a much weaker opponent gains little.
    gain = elo_update(1800, 1400, score=1.0, k=32) - 1800
    assert 0 < gain < 3.5


def test_elo_rating_folds_batch_in_order():
    result = elo_rating(1500, [(1500, 1.0), (1500, 0.0)], k=32)
    # win then loss from even ratings: +16, then a loss scored against 1516.
    step1 = elo_update(1500, 1500, 1.0, k=32)
    step2 = elo_update(step1, 1500, 0.0, k=32)
    assert result == pytest.approx(step2)


def test_elo_rating_empty_is_unchanged():
    assert elo_rating(1500, [], k=32) == 1500
