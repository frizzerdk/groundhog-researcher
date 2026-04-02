# /// script
# dependencies = ["groundhog-researcher", "python-dotenv"]
# ///
"""
My optimization task — detailed template.

HOW THIS WORKS
==============
Groundhog is an LLM-powered iterative optimizer. You define a task (what to
optimize), and it runs a loop: generate code -> evaluate -> learn -> repeat.

This file defines everything the optimizer needs:
  1. Data     — what the generated code works with (train/test split)
  2. Context  — what the LLM sees when generating code (the problem description)
  3. Evaluator — how generated code is scored (evaluation stages with scorers)
  4. Task     — ties Data + Context + Evaluator together
  5. Optimizer — configures strategies, LLM backends, and runs the loop

CORE PRINCIPLES
===============
- The evaluator is IMMUTABLE — the optimizer cannot modify it. This prevents
  gaming the metric. Your evaluation logic is the source of truth.
- Attempts store RAW RESULTS (metrics dicts), not scores. Scores are computed
  on the fly by per-stage scorers. Change the scorer, get different rankings
  without re-running anything.
- Every attempt is kept — nothing is discarded. Failed attempts inform future
  strategies just as much as successful ones.
- Learnings accumulate across iterations. The optimizer records what worked
  and what didn't, and feeds it back to the LLM in future prompts.

RUNNING
=======
  uv run task.py 10       # run 10 iterations
  uv run task.py status   # show current state (best score, trunks, costs)
"""

from dotenv import load_dotenv
load_dotenv()

from groundhog import (
    Task, Data, Context, Evaluator, EvalStage, StageResult,
    SimpleOptimizer, Improve, FreshApproach, CrossPollinate, Analyse,
    auto_registry, BackendRegistry, GeminiBackend, OpenAICompatibleBackend,
    AnthropicBackend, ClaudeCodeBackend,
)


# ==========================================================================
# DATA
# ==========================================================================
# Data provides train/test splits. The training data is passed to the
# generated code so it can learn from it. The test data is used by the
# evaluator to score the code's output.
#
# Both get_train() and get_test() return whatever your task needs — dicts,
# tuples, numpy arrays. The generated code receives get_train() directly.

class MyData(Data):
    def get_train(self):
        return {}  # TODO: your training data

    def get_test(self):
        return {}  # TODO: your test data


# ==========================================================================
# CONTEXT
# ==========================================================================
# Context is what the LLM sees when generating code. It's the problem
# description — what to write, what constraints to follow, what's available.
#
# get_brief() — one-line summary, used in headers and logs
# get_extended() — full instructions shown to the LLM
#
# TIPS FOR GOOD CONTEXT:
# - Be specific about the function signature the code must implement
# - List available libraries/imports explicitly
# - Mention resource constraints (time limits, memory) prominently
# - Include a simple example showing the expected interface
# - If there's a time_limit, tell the LLM to USE it (scale effort with time)

class MyContext(Context):
    def get_brief(self):
        return "Write a function that solves X. Goal: maximize Y."

    def get_extended(self):
        return """Write a function `solve(data)` that returns a result.

Input:
- data: dict with keys "inputs" (list) and "targets" (list)

Output:
- A numeric result (higher is better)

Available libraries: numpy, sklearn
No other imports allowed.

Example:
    def solve(data):
        inputs = data["inputs"]
        return sum(inputs) / len(inputs)
"""


# ==========================================================================
# EVALUATOR
# ==========================================================================
# The evaluator runs generated code and scores it. This is the most
# important component — it defines what "good" means.
#
# EVALUATION STAGES:
# Stages run in order, cheapest first. If a stage produces errors, the
# cascade stops (no point running expensive evaluation on broken code).
#
# Each stage returns a StageResult:
#   StageResult(metrics={"accuracy": 0.85, "time": 1.2})   — success
#   StageResult(errors={"runtime": "division by zero"})     — failure
#
# Each stage has a SCORER — a function that extracts a comparable score
# from the StageResult. The default scorer returns metrics["score"].
# Custom scorers let you score by any metric:
#   EvalStage("eval", "...", call, scorer=lambda r: r.metrics["accuracy"])
#
# STAGE DESIGN TIPS:
# - "smoke" stage: just compile the code, catch syntax errors (instant)
# - "validate" stage: run on a small subset, fast feedback (seconds)
# - "evaluate" stage: full evaluation, the real score (minutes)
# - Use the "through" parameter to stop at a stage during optimization:
#     optimizer = SimpleOptimizer(task, through="evaluate")
#
# SUBPROCESS EXECUTION:
# For untrusted generated code, use groundhog's subprocess runner:
#   from groundhog import run_code
#   result = run_code(code, entry_point="solve", args=(data,), timeout=60)
# This runs code in an isolated process with hard timeout.
#
# PATH OR STRING:
# evaluate() receives either a code string or a Path to the workspace
# directory. If Path, read whatever files you need (e.g. path / "solution.py").
# If string, use it directly. Use _read_code() helper for the common case.

def _read_code(code_or_path):
    """Accept code string or workspace Path, return code string."""
    from pathlib import Path
    if isinstance(code_or_path, (str, bytes)):
        return code_or_path
    return (Path(code_or_path) / "solution.py").read_text(encoding="utf-8")


class MyEvaluator(Evaluator):
    def evaluate(self, code_or_path, data):
        code = _read_code(code_or_path)
        # TODO: execute code, measure performance
        # Option 1: exec() directly (simple but no isolation)
        # Option 2: run_code() for subprocess isolation with timeout
        return StageResult(metrics={"score": 0.0})

    def get_stages(self, data):
        return [
            # Stage 1: instant syntax check
            EvalStage("smoke", "Syntax check",
                      lambda cp: self._smoke(cp)),

            # Stage 2: full evaluation (scored by "score" metric)
            EvalStage("evaluate", "Full evaluation",
                      lambda cp: self.evaluate(cp, data),
                      scorer=lambda r: r.metrics.get("score", 0.0)),
        ]

    def _smoke(self, code_or_path):
        code = _read_code(code_or_path)
        try:
            compile(code, "<string>", "exec")
            return StageResult(metrics={"compiles": 1.0})
        except SyntaxError as e:
            return StageResult(errors={"syntax": str(e)})


# ==========================================================================
# TASK
# ==========================================================================
# Task ties Data + Context + Evaluator together. The name is used for display.

task = Task(data=MyData(), context=MyContext(), evaluator=MyEvaluator(), name="MyTask")


# ==========================================================================
# OPTIMIZER
# ==========================================================================
# The optimizer runs strategies in a loop. Configure:
#
# STRATEGIES:
#   Improve()                        — refine the best attempt via diffs
#   FreshApproach(mode="blank")      — generate from scratch, no context
#   FreshApproach(mode="different")  — generate a fundamentally new approach
#   CrossPollinate()                 — combine ideas from different trunks
#   Analyse()                        — compress/reformat learnings (no code)
#
# STRATEGY ROTATION:
#   strategies=[(Improve(), 14), (CrossPollinate(), 5), (FreshApproach(), 1)]
#   This cycles: 14 improve, 5 cross-pollinate, 1 fresh = 20 per cycle.
#
# BACKEND TIERS:
#   auto_registry() discovers what's available (CLI tools, API keys, local servers)
#   and builds a registry automatically. Or configure manually:
#     BackendRegistry(
#         high=AnthropicBackend(model="claude-opus-4-6-20260205"),
#         default=GeminiBackend(model="gemini-2.5-flash"),
#         cheap=OpenAICompatibleBackend.ollama(model="llama3"),
#     )
#
#   Available backends:
#     API:  AnthropicBackend, GeminiBackend, OpenAICompatibleBackend
#           OpenAICompatibleBackend has factory methods: .openai(), .groq(),
#           .deepseek(), .ollama(), .openrouter(), .together(), .cerebras(), etc.
#     CLI:  ClaudeCodeBackend, CopilotBackend, GeminiCLIBackend, OpenCodeBackend
#
#   Strategies request tiers; missing tiers fall back to "default".
#
# STRATEGY CONFIG:
#   Each strategy has configurable parameters:
#     Improve(max_retries=5, learnings_last=10)
#   See all params: Improve.Config().describe()
#
# QUEUE:
#   Override the rotation by writing to queue.json:
#     from groundhog.tools.queue import add
#     add(".", "fresh_approach", {"mode": "blank"}, source="user")
#   The optimizer will run it next, then resume rotation.
#
# SEEDING:
#   seed_strategy runs once if there's no history. Default: FreshApproach().
#   Set seed_strategy=None to skip (e.g. if history already exists).

if __name__ == "__main__":
    import sys

    optimizer = SimpleOptimizer(
        task,
        strategies=[
            (Improve(), 14),
            (CrossPollinate(), 5),
            (FreshApproach(mode="different"), 1),
        ],
        seed_strategy=FreshApproach(mode="blank"),
    )
    # Auto-discovers available backends (CLI tools, API keys, local servers)
    # Run "groundhog backends" to see what's available on your machine
    optimizer.toolkit.llm = auto_registry()

    # Or configure manually — full control over which models power each tier:
    # optimizer.toolkit.llm = BackendRegistry(
    #     max=AnthropicBackend(model="claude-opus-4-6-20260205"),           # best reasoning ($5/$25 per MTok)
    #     high=GeminiBackend(model="gemini-3-flash-preview"),               # strong + fast ($0.50/$3)
    #     default=ClaudeCodeBackend(model="sonnet"),                        # via CLI, no API key needed
    #     budget=OpenAICompatibleBackend.deepseek(model="deepseek-chat"),   # great value ($0.28/$0.42)
    #     cheap=OpenAICompatibleBackend.ollama(model="llama3"),             # free local model
    # )
    #
    # More providers: .openai(), .groq(), .cerebras(), .xai(), .together(),
    #                 .fireworks(), .openrouter(), .mistral(), .perplexity()
    # Missing tiers fall back to "default".

    if len(sys.argv) > 1 and sys.argv[1] == "status":
        optimizer.status()
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        optimizer.run(n=n)
