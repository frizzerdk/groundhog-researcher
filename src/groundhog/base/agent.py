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
from typing import Any, Callable, Dict, List, Optional, Union


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
    "int": int,
    "float": float,
    "bool": lambda v: v.lower() in ("true", "1", "yes") if isinstance(v, str) else bool(v),
}


class AgentTool:
    """A tool that an agent can invoke. Created via agent_tool() factory.

    Wraps any callable with parameter descriptions, type coercion, and
    error handling. The bash wrapper and tool server use get_parameters()
    to build the CLI interface; execute() handles coercion and calling.
    """

    def __init__(self, name: str, description: str, func: Callable,
                 params: Optional[Dict[str, Dict[str, Any]]] = None):
        self.name = name
        self.description = description
        self._func = func
        self._params = params or {}

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


def agent_tool(name: str, description: str, func: Callable,
               params: Optional[Dict[str, Dict[str, Any]]] = None) -> AgentTool:
    """Create an AgentTool wrapping any callable.

    Usage:
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
    return AgentTool(name=name, description=description, func=func, params=params)


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
    """

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
