# Re-export all base types and interfaces
from groundhog.base.types import (
    Task, Data, Context, Evaluator,
    EvalStage, StageResult, EvaluationResult,
)
from groundhog.base.attempt_history import Attempt, Workspace, AttemptHistory
from groundhog.base.strategy import Strategy, StrategyConfig, param
from groundhog.base.optimizer import Optimizer
from groundhog.base.backend import (
    PromptPart, TextPart, ImagePart, AudioPart, Prompt,
    LLMBackend, LLMResponse, BackendRegistry,
)
from groundhog.base.learnings import Learnings
from groundhog.base.acceptance import Acceptance
from groundhog.base.toolkit import Toolkit
from groundhog.base.agent import (
    ToolResult, AgentTool, agent_tool,
    AgentSpec, AgentResult,
    AgentBackend, AgentRegistry,
)
