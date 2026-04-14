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
"""E2E test: MNIST + AgentStrategy with Claude Code (Sonnet + Haiku).

Runs 3 iterations of the agent strategy on the MNIST task.
Uses Sonnet as the default agent model and Haiku as the budget model.

Usage:
    cd tests/e2e_mnist_agent
    uv run task.py 3
    uv run task.py status
"""

import sys

from dotenv import load_dotenv
load_dotenv()

# --- Re-use the MNIST task from templates ---
from groundhog.templates.mnist_task import MNISTTask

# --- Strategy and optimizer ---
from groundhog import (
    SimpleOptimizer,
    AgentStrategy,
    AnthropicBackend,
    BackendRegistry,
    ClaudeCodeAgentBackend,
)
from groundhog.base.agent import AgentRegistry


task = MNISTTask()

# Agent strategy: Claude Code CLI with sonnet model
agent_strategy = AgentStrategy()

optimizer = SimpleOptimizer(
    task,
    strategy=agent_strategy,
    seed_strategy=agent_strategy,
    through="evaluate",        # score through the 60s eval stage
    agent_through="validate",  # agent gets fast 15s eval tool for iteration
)

# LLM backends for non-agent use (learnings, analysis, etc.)
optimizer.toolkit.llm = BackendRegistry(
    default=AnthropicBackend(model="claude-sonnet-4-6-20260217"),
    cheap=AnthropicBackend(model="claude-haiku-4-5-20250414"),
)

# Agent backends: Sonnet for default, Haiku for budget
optimizer.toolkit.agent = AgentRegistry(
    default=ClaudeCodeAgentBackend(model="sonnet", max_budget_usd=0.50),
    high=ClaudeCodeAgentBackend(model="sonnet", max_budget_usd=1.00),
    budget=ClaudeCodeAgentBackend(model="haiku", max_budget_usd=0.25),
)

if len(sys.argv) > 1 and sys.argv[1] == "status":
    optimizer.status()
else:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    optimizer.run(n=n)
