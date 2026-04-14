__version__ = "0.2.7"

# Base types and interfaces
from groundhog.base import (
    Task, Data, Context, Evaluator,
    EvalStage, StageResult, EvaluationResult,
    Attempt, Workspace, AttemptHistory,
    Strategy, StrategyConfig, param, Optimizer,
    PromptPart, TextPart, ImagePart, AudioPart, Prompt,
    LLMBackend, LLMResponse, BackendRegistry,
    Learnings, Acceptance, Toolkit,
    ToolResult, AgentTool, agent_tool,
    AgentSpec, AgentResult,
    AgentBackend, AgentRegistry,
)

# Utilities
from groundhog.utils.codegen import extract_code, Diff, build_prompt
from groundhog.utils.subprocess_runner import run_code
from groundhog.tools.cost_estimate import estimate_cost, estimate_total_cost

# Default implementations
from groundhog.optimizers.simple import SimpleOptimizer
from groundhog.histories.folder import FolderAttemptHistory
from groundhog.learnings.markdown import MarkdownLearnings
from groundhog.acceptance.default import DefaultAcceptance
from groundhog.backends.mock import MockBackend
from groundhog.backends.gemini import GeminiBackend
from groundhog.backends.openai_compat import OpenAICompatibleBackend
from groundhog.backends.anthropic import AnthropicBackend
from groundhog.backends.claude_code import ClaudeCodeBackend
from groundhog.backends.copilot import CopilotBackend
from groundhog.backends.gemini_cli import GeminiCLIBackend
from groundhog.backends.opencode import OpenCodeBackend
from groundhog.backends.discover import discover_backends, auto_registry
from groundhog.strategies.improve import Improve
from groundhog.strategies.fresh import FreshApproach
from groundhog.strategies.cross_pollinate import CrossPollinate
from groundhog.strategies.analyse import Analyse
from groundhog.strategies.agent import AgentStrategy
from groundhog.agents.claude_code import ClaudeCodeAgentBackend
from groundhog.agents.gemini_cli import GeminiCliAgentBackend
from groundhog.agents.copilot import CopilotAgentBackend
from groundhog.backends.discover import discover_agent_backends, auto_agent_registry
