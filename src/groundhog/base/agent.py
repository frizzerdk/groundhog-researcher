"""Agent Backend — interface for autonomous multi-turn agent sessions.

Parallel to backend.py (stateless LLM calls), this defines the contract for
agents that run autonomously with tool access: Claude Code CLI, Gemini CLI, etc.

Tools are created via the agent_tool() factory — wrap any callable, no
subclassing needed. The factory handles type coercion (bash passes strings),
return conversion, and error handling.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# --- Tool result ---

@dataclass
class ToolResult:
    """Result from executing an agent tool."""
    success: bool
    output: str
    error: Optional[str] = None


# --- Agent tool (concrete, created via factory) ---

# Type coercion map: string name -> converter function
_TYPE_COERCIONS = {
    "str": str,
    "path": str,  # resolved to absolute in bash wrapper, passed as string
    "int": int,
    "float": float,
    "bool": lambda v: v.lower() in ("true", "1", "yes") if isinstance(v, str) else bool(v),
}


class AgentTool:
    """A tool that an agent can invoke. Created via agent_tool() factory.

    Wraps any callable with parameter descriptions, type coercion, and
    error handling. The bash wrapper and tool server use get_parameters()
    to build the CLI interface; execute() handles coercion and calling.

    ``inject_toolkit=True`` marks a tool whose wrapped function takes the
    toolkit as its FIRST parameter (named ``toolkit``). The toolkit is bound
    once by the agent_tools hook collection and supplied at invoke time —
    the agent's schema never sees it.
    """

    def __init__(self, name: str, description: str, func: Callable,
                 params: Optional[Dict[str, Dict[str, Any]]] = None,
                 inject_toolkit: bool = False):
        self.name = name
        self.description = description
        self._func = func
        self._params = params or {}
        self._inject_toolkit = inject_toolkit
        self._toolkit = None

    def bind_toolkit(self, toolkit) -> None:
        """Supply the toolkit an injecting tool receives at invoke time.
        Called by the hook collection in assemble_toolkit; no-op otherwise."""
        if self._inject_toolkit:
            self._toolkit = toolkit

    def get_parameters(self) -> Dict[str, Any]:
        """Return parameter schema: {name: {type, default?, description?}}."""
        return dict(self._params)

    def execute(self, **kwargs) -> ToolResult:
        """Execute the wrapped function with type coercion.

        All values arrive as strings from bash wrappers. Coerces to declared
        types before calling. Returns ToolResult with string output.
        """
        try:
            coerced = self._coerce_args(kwargs)
            if self._inject_toolkit:
                if self._toolkit is None:
                    raise RuntimeError(
                        f"tool {self.name!r} takes a `toolkit` parameter but "
                        f"none is bound — task tools are bound by the task.py "
                        f"agent_tools hook; framework defaults must bind "
                        f"themselves in build_default_agent_tools"
                    )
                result = self._func(self._toolkit, **coerced)
            else:
                result = self._func(**coerced)
            output = self._format_output(result)
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _coerce_args(self, kwargs: dict) -> dict:
        """Coerce string arguments to declared types."""
        coerced = {}
        for name, value in kwargs.items():
            param_def = self._params.get(name, {})
            type_name = param_def.get("type", "str")
            coercer = _TYPE_COERCIONS.get(type_name, str)
            try:
                coerced[name] = coercer(value)
            except (ValueError, TypeError):
                coerced[name] = value  # pass through if coercion fails
        return coerced

    @staticmethod
    def _format_output(result: Any) -> str:
        """Convert function return value to string for ToolResult."""
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2, default=str)
        return str(result)


# Annotation -> schema type name for the derived form. Anything else is a
# loud build-time error (the coercion layer only understands these).
_ANNOTATION_TYPES = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
    Path: "path",
}


def _derive_tool(f: Callable, *, name: Optional[str] = None,
                 description: Optional[str] = None,
                 params: Optional[Dict[str, Dict[str, Any]]] = None) -> AgentTool:
    """Build an AgentTool from a plain function via introspection.

    One source of truth: the schema comes from the function itself —
    name from ``__name__`` (kebab-cased), description from the docstring,
    params/types/defaults from the signature. A first parameter named
    ``toolkit`` marks toolkit injection and is hidden from the agent.
    Explicit kwargs override any derived field.
    """
    import inspect

    if description is None:
        description = inspect.getdoc(f)
        if not description:
            raise ValueError(
                f"agent_tool({f.__name__}): a derived tool needs a docstring — "
                f"it IS the description the agent reads (or pass description=...)"
            )

    sig = inspect.signature(f)
    inject = False
    derived_params: Dict[str, Dict[str, Any]] = {}
    for p in sig.parameters.values():
        if p.name == "toolkit":
            inject = True
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue  # *args/**kwargs are not agent-addressable
        if p.annotation is inspect.Parameter.empty:
            type_name = "str"
        else:
            type_name = _ANNOTATION_TYPES.get(p.annotation)
            if type_name is None:
                raise ValueError(
                    f"agent_tool({f.__name__}): unsupported annotation for "
                    f"parameter {p.name!r}: {p.annotation!r} "
                    f"(use str / int / float / bool / Path)"
                )
        spec: Dict[str, Any] = {"type": type_name}
        if p.default is not inspect.Parameter.empty:
            spec["default"] = p.default
        derived_params[p.name] = spec

    return AgentTool(
        name=name or f.__name__.replace("_", "-"),
        description=description,
        func=f,
        params=params if params is not None else derived_params,
        inject_toolkit=inject,
    )


def agent_tool(func_or_name=None, description: Optional[str] = None,
               func: Optional[Callable] = None,
               params: Optional[Dict[str, Dict[str, Any]]] = None,
               *, name: Optional[str] = None) -> AgentTool:
    """Create an AgentTool — derived from a function, or fully explicit.

    Derived form (preferred): pass the function; name, description, and the
    param schema come from the function itself, so they cannot drift. A
    first parameter named ``toolkit`` is injected at invoke time and hidden
    from the agent:

        def render_digits(toolkit, n: int = 16) -> str:
            \"\"\"Render a strip of digits to a PNG for inspection.\"\"\"
            ...
        tool = agent_tool(render_digits)
        tool = agent_tool(render_digits, name="show-digits")   # per-field override

    Explicit form (unchanged — for lambdas, bound methods, rich per-param
    descriptions):

        tool = agent_tool(
            name="get-learnings",
            description="Read accumulated learnings",
            func=toolkit.learnings.get,
            params={
                "last": {"type": "int", "default": 20, "description": "Recent entries"},
                "random": {"type": "int", "default": 10, "description": "Random older entries"},
            },
        )
    """
    if callable(func_or_name):
        return _derive_tool(func_or_name, name=name,
                            description=description, params=params)
    resolved_name = name if name is not None else func_or_name
    if resolved_name is None or func is None:
        raise TypeError(
            "agent_tool: pass a function (derived form) or name= + "
            "description= + func= (explicit form)"
        )
    return AgentTool(name=resolved_name, description=description or "",
                     func=func, params=params)


# --- Agent spec and result ---

@dataclass
class AgentSpec:
    """Specification for a single agent run.

    The strategy builds this per phase. The backend interprets it.
    """
    goal: str
    workspace_path: Path
    tools: List[AgentTool] = field(default_factory=list)
    model: Optional[str] = None             # override backend default
    effort: Optional[str] = None            # "low"/"medium"/"high"/"max" — backend maps to equivalent
    allowed_tools: List[str] = field(default_factory=list)   # e.g. ["Bash(uv run *)"]
    denied_tools: List[str] = field(default_factory=list)    # e.g. ["Bash(rm -rf *)"]
    timeout: Optional[int] = None           # seconds
    budget_usd: Optional[float] = None      # cost cap — backend enforces if supported
    session_id: Optional[str] = None        # for resume — opaque, backend interprets
    env: Dict[str, str] = field(default_factory=dict)
    on_event: Optional[Callable] = None     # callback(event_dict) for live progress


@dataclass
class AgentResult:
    """Result from a single agent run."""
    success: bool
    output: str
    session_id: Optional[str] = None        # for resume — backend returns this
    cost: float = 0.0
    turns: int = 0
    duration_ms: int = 0
    error: Optional[str] = None
    steps: List[Dict] = field(default_factory=list)


# --- Agent backend interface ---

class AgentBackend(ABC):
    """Interface for an autonomous agent backend.

    Unlike LLMBackend (single-turn text generation), an AgentBackend runs
    a multi-turn session where the agent reasons, calls tools, and acts
    autonomously. Each run() call is one job — the strategy decides phasing.

    cost_model: how the backend charges. Strategy adapts execution pattern;
    display sites annotate the reported cost honestly:
        "per_token"   — API-billed, cost scales with work. Multi-phase calls
                        are fine; shown as a plain dollar amount.
        "per_request" — fixed cost per call (subscription request credits).
                        One big call is cheapest; shown as "$X (plan value)".
        "none"        — backend reports no usable cost (e.g. codex). Shown as
                        "unreported (subscription)" instead of a false $0.00.
    """
    cost_model: str = "per_token"

    @abstractmethod
    def run(self, spec: AgentSpec) -> AgentResult: ...


# --- Agent registry ---

class AgentRegistry:
    """Maps tier names to agent backends. Parallel to BackendRegistry.

    Usage:
        registry = AgentRegistry(default=claude_code, budget=gemini_cli)
        agent = registry.get("default")
        result = agent.run(spec)
    """

    def __init__(self, **tiers: AgentBackend):
        self._tiers = tiers

    def set(self, tier: str, backend: AgentBackend):
        """Set or override a tier's agent backend."""
        self._tiers[tier] = backend

    def get(self, tier: str = "default") -> AgentBackend:
        if tier in self._tiers:
            return self._tiers[tier]
        if "default" in self._tiers:
            return self._tiers["default"]
        raise KeyError(
            f"No agent backend for tier '{tier}' and no default. "
            f"Available: {list(self._tiers.keys())}"
        )
