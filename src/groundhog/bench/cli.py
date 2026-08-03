"""``groundhog bench`` — offline strategy benchmarking from config files."""

USAGE = """Usage: groundhog bench <subcommand>

  run <config.py> [--seeds N]           Run one config, print per-seed metrics
  compare <a.py> <b.py> [--seeds N]     Paired A/B comparison over the same seeds

A config file is a python module exposing `def bench_config() -> BenchConfig`.
Seeds default to 5 (seeds 0..N-1). See docs/benchmarking.md.
"""


def bench_group(args) -> int:
    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    sub, rest = args[0], list(args[1:])

    seeds = 5
    if "--seeds" in rest:
        i = rest.index("--seeds")
        if i + 1 >= len(rest):
            print("--seeds needs a value")
            return 1
        try:
            seeds = int(rest[i + 1])
        except ValueError:
            print(f"--seeds needs an integer, got {rest[i + 1]!r}")
            return 1
        rest = rest[:i] + rest[i + 2:]

    if sub == "run":
        if len(rest) != 1:
            print("Usage: groundhog bench run <config.py> [--seeds N]")
            return 1
        config = _load(rest[0])
        if config is None:
            return 1
        from groundhog.bench.runner import run_config
        result = run_config(config, seeds, progress=True)
        print()
        print(result.summary())
        return 0

    if sub == "compare":
        if len(rest) != 2:
            print("Usage: groundhog bench compare <configA.py> <configB.py> "
                  "[--seeds N]")
            return 1
        config_a = _load(rest[0])
        config_b = _load(rest[1])
        if config_a is None or config_b is None:
            return 1
        from groundhog.bench.comparison import compare
        comparison = compare(config_a, config_b, seeds, progress=True)
        print()
        print(comparison.format())
        return 0

    print(f"Unknown bench subcommand: {sub!r}")
    print(USAGE)
    return 1


def _load(path):
    from groundhog.bench.loader import load_bench_config
    try:
        return load_bench_config(path)
    except Exception as e:  # noqa: BLE001 — CLI boundary
        print(f"Could not load bench config {path}: {e}")
        return None
