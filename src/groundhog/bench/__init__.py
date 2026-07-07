"""Offline benchmark harness — deterministic strategy comparison, no API spend.

``run_config`` measures one setup (task factory + strategy schedule) across
seeds; ``compare`` runs two setups on the SAME seeds and pairs the results.
Configs default to LLM-free strategies; real-LLM configs work through
``BenchConfig.configure_toolkit`` but cost money. See docs/benchmarking.md.

``groundhog.bench.stats`` is stdlib-only and importable on its own, so
in-run consumers (e.g. an A/B meta-strategy) can reuse the paired stats
without pulling in the offline runner.
"""

from groundhog.bench.runner import BenchConfig, BenchResult, SeedRun, run_config
from groundhog.bench.comparison import Comparison, compare, compare_results
from groundhog.bench.loader import load_bench_config
from groundhog.bench.stats import PairedStats, mean_std, paired_stats

__all__ = [
    "BenchConfig", "BenchResult", "SeedRun", "run_config",
    "Comparison", "compare", "compare_results",
    "load_bench_config",
    "PairedStats", "mean_std", "paired_stats",
]
