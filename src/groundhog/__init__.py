# Base types and interfaces
from groundhog.base import (
    Task, Data, Context, Evaluator,
    EvalStage, StageResult, EvaluationResult,
    Attempt, Workspace, AttemptHistory,
    Strategy, StrategyConfig, param, Optimizer,
    PromptPart, TextPart, ImagePart, AudioPart, Prompt,
    LLMBackend, LLMResponse, BackendRegistry,
    Learnings, Acceptance, Toolkit,
)

# Utilities
from groundhog.utils.codegen import extract_code, parse_diff, apply_diff, build_prompt
from groundhog.utils.subprocess_runner import run_code

# Default implementations
from groundhog.optimizers.simple import SimpleOptimizer
from groundhog.histories.folder import FolderAttemptHistory
from groundhog.learnings.markdown import MarkdownLearnings
from groundhog.acceptance.default import DefaultAcceptance
from groundhog.backends.mock import MockBackend
from groundhog.backends.gemini import GeminiBackend
from groundhog.strategies.improve import Improve
from groundhog.strategies.fresh import FreshApproach
from groundhog.strategies.cross_pollinate import CrossPollinate
from groundhog.strategies.analyse import Analyse
