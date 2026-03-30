"""Toolkit — the capabilities available to strategies during execution.

The operations available during an attempt — *what you can do*. The Workspace
is *where* you work; the Toolkit is what tools are available.

Just a container. Tools are added by the optimizer from various sources.
Built on SimpleNamespace: strategies access what they need via attributes.
Overrides are tracked and printed.
"""

from types import SimpleNamespace


class Toolkit(SimpleNamespace):
    """Dynamic namespace for strategy capabilities.

    The optimizer builds a Toolkit and adds tools to it.
    Strategies access what they need via attributes.
    Overrides are tracked and printed.
    """

    def __setattr__(self, name, value):
        if hasattr(self, name) and name != '_overrides':
            if not hasattr(self, '_overrides'):
                super().__setattr__('_overrides', {})
            old = getattr(self, name)
            self._overrides[name] = old
            print(f"Toolkit: overriding '{name}' (was {old!r})")
        super().__setattr__(name, value)
