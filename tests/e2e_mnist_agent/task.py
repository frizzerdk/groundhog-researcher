"""E2E test: MNIST + every strategy type across backends.

Usage:
    cd tests/e2e_mnist_agent
    uv run task.py claude 1       # 1 iter, AgentStrategy via Claude Code haiku
    uv run task.py copilot 1      # 1 iter, AgentStrategy via Copilot gpt-5-mini
    uv run task.py llm 4          # LLM strategies: Improve x2, CrossPollinate, Fresh (haiku)
    uv run task.py status
"""

import sys

from dotenv import load_dotenv
load_dotenv()

from groundhog.templates.mnist_task import MNISTTask

from groundhog import (
    assemble_toolkit,
    SimpleOptimizer,
    AgentStrategy,
    AnthropicBackend,
    BackendRegistry,
    ClaudeCodeAgentBackend,
    CopilotAgentBackend,
    CodexCliAgentBackend,
    CrossPollinate,
    FreshApproach,
    Improve,
)
from groundhog.base.agent import AgentRegistry


backend_name = sys.argv[1] if len(sys.argv) > 1 else "claude"

if backend_name == "claude":
    agent_backend = ClaudeCodeAgentBackend(model="haiku", max_budget_usd=0.25)
elif backend_name == "copilot":
    agent_backend = CopilotAgentBackend(model="gpt-5-mini")
elif backend_name == "codex":
    agent_backend = CodexCliAgentBackend(effort="medium")
elif backend_name in ("llm", "status"):
    agent_backend = None
else:
    raise SystemExit(f"Unknown backend: {backend_name!r} (expected: claude, copilot, codex, llm, status)")

task = MNISTTask()

if backend_name == "llm":
    # LLM-strategy rotation — one cheap backend on every tier, picked from
    # whatever auth this machine has.
    import os
    from groundhog import ClaudeCodeBackend, OpenAICompatibleBackend

    if os.environ.get("ANTHROPIC_API_KEY"):
        llm = AnthropicBackend(model="claude-haiku-4-5-20250414")
    elif os.environ.get("OPENROUTER_API_KEY"):
        llm = OpenAICompatibleBackend.openrouter(model="google/gemini-3-flash-preview")
    else:
        llm = ClaudeCodeBackend(model="haiku")

    optimizer = SimpleOptimizer(
        assemble_toolkit(task, through="evaluate"),
        strategies=[(Improve(), 2), (CrossPollinate(), 1), (FreshApproach(), 1)],
    )
    optimizer.toolkit.llm = BackendRegistry(default=llm, high=llm, cheap=llm)
else:
    agent_strategy = AgentStrategy()
    optimizer = SimpleOptimizer(
        assemble_toolkit(task, through="evaluate", agent_through="validate"),
        strategy=agent_strategy,
        seed_strategy=agent_strategy,
    )
    optimizer.toolkit.llm = BackendRegistry(
        default=AnthropicBackend(model="claude-sonnet-4-6-20260217"),
        cheap=AnthropicBackend(model="claude-haiku-4-5-20250414"),
    )

if agent_backend is not None:
    optimizer.toolkit.agent = AgentRegistry(
        default=agent_backend,
        high=agent_backend,
        budget=agent_backend,
    )

if backend_name == "status":
    optimizer.status()
else:
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    optimizer.run(n=n)
