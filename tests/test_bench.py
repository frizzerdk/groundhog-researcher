"""End-to-end tests for the offline bench harness (groundhog.bench).

Everything runs on the seedable MockTask with LLM-free strategies —
deterministic and fast, safe for CI.
"""

from groundhog.bench import BenchConfig, compare, run_config
from groundhog.bench.cli import bench_group
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
