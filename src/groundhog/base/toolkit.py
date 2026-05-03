"""Toolkit — the capabilities available to strategies during execution.

The operations available during an attempt — *what you can do*. The Workspace
is *where* you work; the Toolkit is what tools are available.

Just a container. Tools are added by the optimizer from various sources.
Built on SimpleNamespace: strategies access what they need via attributes.
Overrides are tracked and printed.

User-overridable attributes (set after constructing the optimizer, before
``run()``):

    toolkit.get_prior(toolkit) -> Optional[Attempt]
        Picks which attempt the next strategy iteration should build on.
        The default (potential-weighted scoring across trunk leaders) is
        installed by ``SimpleOptimizer.__init__``. Override to plug in
        custom selection — e.g. live-rating-based picks for a tournament-
        style task. The override is preserved across all iterations of
        ``run()``.

    toolkit.agent_through: Optional[str]
        Names the eval stage the agent's eval tools cap at — e.g. ``"validate"``
        keeps cheap-iteration loops fast while final commit-time scoring still
        runs ``through``'s full chain. Set via the ``agent_through=`` kwarg on
        ``SimpleOptimizer`` or directly on the toolkit afterward.

    toolkit.through: Optional[str]
        Names the eval stage the optimizer scores against. Defaults to the
        last stage in ``task.evaluator.eval_stages(...)``.

    toolkit.llm: BackendRegistry
        LLM backend lookup for non-agent strategies. Tier names like
        ``"default"`` / ``"high"`` / ``"budget"`` are conventional.

    toolkit.agent: AgentRegistry
        Agent backend lookup. Same tier convention as ``llm``.
"""

from types import SimpleNamespace


class Toolkit(SimpleNamespace):
    """Dynamic namespace for strategy capabilities.

    The optimizer builds a Toolkit and adds tools to it.
    Strategies access what they need via attributes.
    Overrides are tracked and printed.
    """

    def __setattr__(self, name, value):
        # Skip override tracking for private attributes — these are internal
        # bookkeeping (e.g. the optimizer's per-iteration queue label) that
        # can change every loop and would otherwise spam the console.
        if name.startswith('_') or name == '_overrides':
            super().__setattr__(name, value)
            return
        if hasattr(self, name):
            if not hasattr(self, '_overrides'):
                super().__setattr__('_overrides', {})
            old = getattr(self, name)
            self._overrides[name] = old
            print(f"Toolkit: overriding '{name}' (was {old!r})")
        super().__setattr__(name, value)
