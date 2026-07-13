"""End-to-end tests for the offline bench harness (groundhog.bench).

Everything runs on the seedable MockTask with LLM-free strategies —
deterministic and fast, safe for CI.
"""

import pytest

from groundhog.bench import BenchConfig, compare, run_config
from groundhog.bench.cli import bench_group
from groundhog.bench.comparison import compare_results
from groundhog.bench.runner import BenchResult, SeedRun
from groundhog.bench.stats import paired_stats
from groundhog.templates.mock_strategy import MockStrategy
from groundhog.templates.mock_task import MockTask


class OracleStrategy(MockStrategy):
    """Returns the (rounded) train target — deterministically near-perfect."""

    def _do_work(self, toolkit, ws):
        target = float(toolkit.task.data.get_train()["target"])
        (ws.path / "solution.py").write_text(
            f"def solve():\n    return {target}", encoding="utf-8")


def _random_config(name="random", n_iterations=4):
    return BenchConfig(
        name=name,
        task_factory=lambda seed: MockTask(seed=seed),
        strategies=[(MockStrategy(), 1)],
        n_iterations=n_iterations,
    )


def test_run_config_end_to_end():
    result = run_config(_random_config(), seeds=2)
    assert result.config_name == "random"
    assert result.seeds == [0, 1]
    for run in result.runs:
        assert run.total_attempts == 4
        assert len(run.scores) == 4
        assert run.trajectory == [max(run.scores[:i + 1]) for i in range(4)]
        assert run.final_best == max(run.scores)
        assert 0.0 <= run.final_best <= 1.0
        assert run.wall_time_s > 0
    assert "final_best" in result.summary()


def test_run_config_deterministic_per_seed():
    first = run_config(_random_config(), seeds=[3])
    second = run_config(_random_config(), seeds=[3])
    assert first.runs[0].scores == second.runs[0].scores
    assert first.runs[0].final_best == second.runs[0].final_best


def test_compare_oracle_beats_random():
    oracle = BenchConfig(
        name="oracle",
        task_factory=lambda seed: MockTask(seed=seed),
        strategies=[(OracleStrategy(), 1)],
        n_iterations=3,
    )
    comparison = compare(oracle, _random_config(n_iterations=3), seeds=2)
    fb = comparison.metrics["final_best"]
    assert fb.n == 2
    assert fb.wins_a == 2
    assert fb.mean_delta > 0
    assert comparison.verdict.startswith("A (oracle) better on 2/2 seeds")
    formatted = comparison.format()
    assert "final_best" in formatted
    assert comparison.verdict in formatted


def test_paired_stats_none_loses():
    ps = paired_stats("first_improvement", [2, None], [None, None],
                      higher_is_better=False)
    assert (ps.wins_a, ps.wins_b, ps.ties) == (1, 0, 1)
    assert ps.mean_delta is None


def test_paired_stats_counts_one_sided_pairs_and_sd():
    """One-sided pairs are excluded from the delta but counted explicitly —
    win counts and mean_delta can disagree, and the fields say why."""
    ps = paired_stats("final_best",
                      [0.9, None, 0.5, 0.8],
                      [None, None, 0.6, 0.6])
    assert ps.n == 4
    assert (ps.n_a, ps.n_b) == (3, 2)
    assert (ps.only_a, ps.only_b) == (1, 0)
    assert (ps.wins_a, ps.wins_b, ps.ties) == (2, 1, 1)
    assert ps.deltas == [None, None, pytest.approx(-0.1), pytest.approx(0.2)]
    assert ps.mean_delta == pytest.approx(0.05)
    assert ps.sd_delta == pytest.approx(0.2121, abs=1e-3)


def _bench_result(name, final_bests, first_improvements=None):
    fi = first_improvements or [None] * len(final_bests)
    runs = [SeedRun(seed=i, scores=[f], trajectory=[f], final_best=f,
                    first_improvement=fi[i], total_attempts=1,
                    wall_time_s=0.01)
            for i, f in enumerate(final_bests)]
    return BenchResult(config_name=name, n_iterations=1, runs=runs)


def test_verdict_requires_a_win_margin():
    """A one-seed win lead is noise: the verdict must not crown a winner."""
    close = compare_results(_bench_result("a", [0.5, 0.5, 0.5]),
                            _bench_result("b", [0.4, 0.6, 0.4]))
    assert close.verdict.startswith("no clear winner")
    assert "2" in close.verdict and "1" in close.verdict

    clear = compare_results(_bench_result("a", [0.5, 0.7, 0.6]),
                            _bench_result("b", [0.4, 0.4, 0.4]))
    assert clear.verdict.startswith("A (a) better on 3/3 seeds")


def test_verdict_reports_one_sided_improvements():
    a = _bench_result("a", [0.5, 0.6, 0.7], first_improvements=[2, 3, None])
    b = _bench_result("b", [0.4, 0.4, 0.4], first_improvements=[None, None, None])
    comparison = compare_results(a, b)
    fi = comparison.metrics["first_improvement"]
    assert (fi.only_a, fi.only_b) == (2, 0)
    assert "A-only" in comparison.format()


def test_compare_results_rejects_mismatched_seeds():
    a = _bench_result("a", [0.5, 0.6])
    b = BenchResult(config_name="b", n_iterations=1, runs=[
        SeedRun(seed=7, scores=[0.4], trajectory=[0.4], final_best=0.4,
                first_improvement=None, total_attempts=1, wall_time_s=0.01)])
    with pytest.raises(ValueError, match="different seeds"):
        compare_results(a, b)


CONFIG_TEMPLATE = """\
from groundhog.bench import BenchConfig
from groundhog.templates.mock_task import MockTask
from groundhog.templates.mock_strategy import MockStrategy


def bench_config():
    return BenchConfig(
        name="{name}",
        task_factory=lambda seed: MockTask(seed=seed),
        strategies=[(MockStrategy(), 1)],
        n_iterations=2,
    )
"""


def test_cli_bench_run_and_compare(tmp_path, capsys):
    config_a = tmp_path / "a.py"
    config_b = tmp_path / "b.py"
    config_a.write_text(CONFIG_TEMPLATE.format(name="cfg-a"), encoding="utf-8")
    config_b.write_text(CONFIG_TEMPLATE.format(name="cfg-b"), encoding="utf-8")

    assert bench_group(["run", str(config_a), "--seeds", "2"]) == 0
    out = capsys.readouterr().out
    assert "cfg-a: 2 seeds x 2 iterations" in out

    assert bench_group(["compare", str(config_a), str(config_b),
                        "--seeds", "2"]) == 0
    out = capsys.readouterr().out
    assert "verdict:" in out
    assert "final_best" in out


def test_cli_bench_bad_config(tmp_path, capsys):
    bad = tmp_path / "bad.py"
    bad.write_text("x = 1\n", encoding="utf-8")
    assert bench_group(["run", str(bad)]) == 1
    assert "bench_config" in capsys.readouterr().out
