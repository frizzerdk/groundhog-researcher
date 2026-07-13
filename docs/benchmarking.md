# Benchmarking strategies offline

`groundhog bench` measures strategy/optimizer changes without API spend: it
runs `SimpleOptimizer` on a cheap deterministic task, once per seed, in a
throwaway run dir, and reports per-seed metrics. The primary interface is
`bench compare` — two configs run on the SAME seeds, results paired per
seed, so "did my change make the loop better?" gets a direct answer.

Per-seed metrics: final best score, score trajectory (running best),
attempts-to-first-improvement (1-based attempt index that first beat
attempt 1, `-` if never), total attempts, wall time.

```bash
groundhog bench run my_config.py --seeds 10
groundhog bench compare baseline.py candidate.py --seeds 10
```

## Config files

A config is a python module exposing `def bench_config() -> BenchConfig`:

- `name` — label used in output.
- `task_factory(seed) -> Task` — a fresh task per seed. `MockTask` from
  `groundhog.templates.mock_task` is seedable and needs no API. (The
  toolkit rng is seeded from a value derived from the same seed, so runs
  are fully deterministic; it is deliberately decorrelated from the task
  seed so a strategy's random draws never replay the task's own.)
- `strategies` — the `SimpleOptimizer` rotation schedule
  `[(strategy, repeats), ...]`; a bare strategy means `[(strategy, 1)]`.
- `n_iterations` — optimizer iterations per seed.
- `seed_strategy` — defaults to `None`, which disables seeding (the first
  rotation strategy runs against an empty history). The optimizer's own
  default is the string `"default"`, which seeds with `FreshApproach` and
  needs an LLM — pass that string or a strategy instance to opt in.
- `configure_toolkit(toolkit)` — optional post-assembly hook, the
  injection point for LLM-backed configs.

## Worked example: pure random vs a local-search mix

`random_mix.py`:

```python
from groundhog.bench import BenchConfig
from groundhog.templates.mock_task import MockTask
from groundhog.templates.mock_strategy import MockStrategy


def bench_config():
    return BenchConfig(
        name="pure-random",
        task_factory=lambda seed: MockTask(seed=seed),
        strategies=[(MockStrategy(), 1)],
        n_iterations=8,
    )
```

`local_mix.py` — mixes in a strategy that perturbs the best-so-far value:

```python
from groundhog.bench import BenchConfig
from groundhog.templates.mock_task import MockTask
from groundhog.templates.mock_strategy import MockStrategy


class LocalSearch(MockStrategy):
    """Perturb the best-so-far value instead of sampling uniformly."""

    def _do_work(self, toolkit, ws):
        solution = ws.path / "solution.py"
        if solution.exists():
            prior = float(solution.read_text(encoding="utf-8").rsplit("return", 1)[1])
            value = min(100.0, max(0.0, prior + toolkit.rng.uniform(-5.0, 5.0)))
        else:
            value = toolkit.rng.uniform(0, 100)
        solution.write_text(f"def solve():\n    return {value}", encoding="utf-8")


def bench_config():
    return BenchConfig(
        name="random+local",
        task_factory=lambda seed: MockTask(seed=seed),
        strategies=[(MockStrategy(), 1), (LocalSearch(), 3)],
        n_iterations=8,
    )
```

```
$ groundhog bench compare random_mix.py local_mix.py --seeds 10
  [pure-random] seed 0: best=0.9167 first_improvement=- attempts=8 (0.10s)
  ...
  [random+local] seed 9: best=0.8357 first_improvement=4 attempts=8 (0.12s)

A: pure-random | B: random+local | 10 seeds x 8 iterations
  metric             A mean+/-sd          B mean+/-sd          delta(A-B)  A/B/tie
  final_best         0.9533 +/- 0.0229    0.8570 +/- 0.1891    +0.0963     5/5/0
  first_improvement  2.4286 +/- 1.1339    3.3000 +/- 1.4181    -1.1429     3/3/4
  total_attempts     8.0000 +/- 0.0000    8.0000 +/- 0.0000    +0.0000     0/0/10
  wall_time_s        0.1166 +/- 0.0134    0.1199 +/- 0.0102    -0.0033     4/6/0
verdict: A (pure-random) and B (random+local) tied: 5 wins each, 0 ties, mean +0.0963 best score
```

Reading it: win counts and the mean are separate signals. Here B wins as
many seeds as A, but its spread is 8x larger (on one seed local search got
stuck at 0.37) — a high-variance mix, not an improvement. Ties in win
counts with a large mean delta are a hint to look at the per-seed lines.

## Real-LLM configs

`configure_toolkit` makes LLM-dependence injectable:

```python
def bench_config():
    from groundhog import auto_registry, Improve, FreshApproach

    def with_llm(toolkit):
        toolkit.llm = auto_registry()

    return BenchConfig(
        name="improve-heavy",
        task_factory=lambda seed: MockTask(seed=seed),
        strategies=[(Improve(), 3), (FreshApproach(), 1)],
        n_iterations=6,
        seed_strategy=FreshApproach(),
        configure_toolkit=with_llm,
    )
```

This works, but every seed x iteration is a real API call — costs money
and is only as deterministic as the model. Keep CI benches LLM-free.

## Programmatic use

```python
from groundhog.bench import BenchConfig, run_config, compare, compare_results
result = run_config(config, seeds=10)          # or seeds=[0, 3, 7]
comparison = compare(config_a, config_b, seeds=10)
comparison = compare_results(result_a, result_b)   # precomputed results
```

`groundhog.bench.stats` (`paired_stats`, `mean_std`) is stdlib-only with no
runner imports, so in-run consumers — e.g. an A/B meta-strategy — can reuse
the paired-comparison math on their own value sequences.
