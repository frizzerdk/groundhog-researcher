# /// script
# dependencies = [
#     "groundhog-researcher",
#     "python-dotenv",
#     "numpy",
#     "scikit-learn",
#     "torch",
# ]
#
# [tool.uv.sources]
# groundhog-researcher = { path = "../.." }
# ///
"""E2E test: MNIST + AgentStrategy across backends.

Usage:
    cd tests/e2e_mnist_agent
    uv run task.py claude 1       # 1 iter, Claude Code with haiku
    uv run task.py copilot 1      # 1 iter, Copilot with gpt-5-mini
    uv run task.py status
"""

import sys

from dotenv import load_dotenv
load_dotenv()

from groundhog.templates.mnist_task import MNISTTask

from groundhog import (
    SimpleOptimizer,
    AgentStrategy,
    AnthropicBackend,
    BackendRegistry,
    ClaudeCodeAgentBackend,
    CopilotAgentBackend,
)
from groundhog.base.agent import AgentRegistry


backend_name = sys.argv[1] if len(sys.argv) > 1 else "claude"

if backend_name == "claude":
    agent_backend = ClaudeCodeAgentBackend(model="haiku", max_budget_usd=0.25)
elif backend_name == "copilot":
    agent_backend = CopilotAgentBackend(model="gpt-5-mini")
elif backend_name == "status":
    agent_backend = None
else:
    raise SystemExit(f"Unknown backend: {backend_name!r} (expected: claude, copilot, status)")

task = MNISTTask()
agent_strategy = AgentStrategy()

optimizer = SimpleOptimizer(
    task,
    strategy=agent_strategy,
    seed_strategy=agent_strategy,
    through="evaluate",
    agent_through="validate",
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
