"""Strategy discovery — a scan, not a registry.

``discover_strategies`` walks the built-in strategies package (and,
optionally, a task module) for concrete :class:`Strategy` subclasses and
describes each one: name, class, source, first docstring line, and the
Config parameters. No ``__init_subclass__`` hooks, no global state — the
scan reads what exists at call time, so a task module's strategies are
discovered exactly like the built-ins.
"""

import importlib
import inspect
import pkgutil

from groundhog.base.strategy import Strategy


def discover_strategies(module=None):
    """Scan ``groundhog.strategies`` plus an optional task module.

    Args:
        module: an imported module (typically a run dir's ``task.py``) whose
            module-level Strategy subclasses are included with source "task".

    Returns a list of dicts, builtins first, each sorted by name:
        {name, cls, source: "builtin"|"task", doc: first docstring line,
         params: cls.Config().describe()}
    """
    import groundhog.strategies as pkg

    found = {}
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{pkg.__name__}.{info.name}")
        for cls in _strategy_classes(mod):
            found.setdefault(cls, "builtin")
    if module is not None:
        for cls in _strategy_classes(module):
            found.setdefault(cls, "task")

    entries = [
        {
            "name": cls.name,
            "cls": cls,
            "source": source,
            "doc": _first_doc_line(cls),
            "params": cls.Config().describe(),
        }
        for cls, source in found.items()
    ]
    entries.sort(key=lambda e: (e["source"] != "builtin", e["name"]))
    return entries


def _strategy_classes(module):
    return [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, Strategy) and not inspect.isabstract(obj)
    ]


def _first_doc_line(cls):
    doc = inspect.getdoc(cls)
    return doc.splitlines()[0].strip() if doc else ""
