"""Load a bench config file — a .py module exposing ``bench_config()``.

Mirrors the run-dir loader's discipline: import under a throwaway module
name, with the file's parent on sys.path so sibling imports resolve, and
deregister afterwards.
"""

import importlib.util
import itertools
import sys
from pathlib import Path

from groundhog.bench.runner import BenchConfig

_LOAD_COUNTER = itertools.count()


def load_bench_config(path) -> BenchConfig:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No bench config file at {path}")
    parent = str(path.parent)
    mod_name = f"groundhog_benchcfg_{next(_LOAD_COUNTER)}"
    sys.path.insert(0, parent)
    try:
        spec = importlib.util.spec_from_file_location(mod_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load a module spec from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
            hook = getattr(module, "bench_config", None)
            if not callable(hook):
                raise RuntimeError(
                    f"{path} does not define `def bench_config() -> BenchConfig`"
                )
            config = hook()
            if not isinstance(config, BenchConfig):
                raise RuntimeError(
                    f"{path}: bench_config() returned {type(config).__name__}, "
                    f"expected a BenchConfig"
                )
        finally:
            sys.modules.pop(mod_name, None)
    finally:
        try:
            sys.path.remove(parent)
        except ValueError:
            pass
    return config
