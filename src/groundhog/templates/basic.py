# /// script
# dependencies = ["groundhog-researcher", "python-dotenv"]
# ///
"""My optimization task."""

from dotenv import load_dotenv
load_dotenv()

from groundhog import (
    Task, Data, Context, Evaluator, EvalStage, StageResult,
    Toolkit, agent_tool, assemble_toolkit, SimpleOptimizer, Improve, auto_registry,
)


# --- Data: what the generated code works with ---

class MyData(Data):
    def get_train(self):
        return {}  # training data passed to generated code

    def get_test(self):
        return {}  # test data used by the evaluator to score


# --- Context: what the LLM sees when generating code ---

class MyContext(Context):
    def get_brief(self):
        return "Write a function that solves X."

    def get_extended(self):
        return """Write a function `solve(data)` that maximizes Y.

Input: data dict with keys ...
Output: ...

Rules:
- ...
"""


# --- Evaluator: how generated code is scored ---

def _read_code(code_or_path):
    """Accept code string or workspace Path, return code string."""
    from pathlib import Path
    if isinstance(code_or_path, (str, bytes)):
        return code_or_path
    return (Path(code_or_path) / "solution.py").read_text(encoding="utf-8")


class MyEvaluator(Evaluator):
    def evaluate(self, code_or_path, data):
        code = _read_code(code_or_path)
        # Execute code, measure performance, return metrics
        # Higher score = better. Metrics are flexible dicts.
        return StageResult(metrics={"score": 0.0})

    def get_stages(self, data):
        # Stages run cheapest first. Cascade stops on error.
        return [
            EvalStage("smoke", "Syntax check", lambda cp: self._smoke(cp)),
            EvalStage("evaluate", "Full evaluation",
                      lambda cp: self.evaluate(cp, data),
                      # Scorer reads metrics — result.score is never
                      # persisted, so a committed attempt re-scores from
                      # metrics read-side.
                      scorer=lambda r: r.metrics.get("score", 0.0)),
        ]

    def _smoke(self, code_or_path):
        code = _read_code(code_or_path)
        try:
            compile(code, "<string>", "exec")
            return StageResult(metrics={"compiles": 1.0})
        except SyntaxError as e:
            return StageResult(errors={"syntax": str(e)})


# --- Task: ties everything together ---

task = Task(data=MyData(), context=MyContext(), evaluator=MyEvaluator(), name="MyTask")


# --- Optional per-task agent tools ---
# The blessed hook: a module-level function (NOT a method on Task — the Task
# stays a pure value object). Wired below via `agent_tools=agent_tools`;
# assemble_toolkit calls it LAST against the finished toolkit. Your tools
# shadow same-named framework defaults (logged). Return [] for none.
#
# Preferred authoring: a plain module-level function — name, description,
# and the agent-visible schema are DERIVED from it (docstring + signature),
# so they cannot drift. A first parameter named `toolkit` is injected at
# invoke time and hidden from the agent:
#
#     def render_sample(toolkit, n: int = 16) -> str:
#         """Render n input samples to a PNG for inspection."""
#         data = toolkit.task.data
#         ...
#         return f"wrote {n} samples"
#
#     def agent_tools(toolkit) -> list:
#         return [agent_tool(render_sample)]
#
# (The fully-explicit form agent_tool(name=..., description=..., func=...,
# params={...}) also works — for lambdas, bound methods, or rich per-param
# descriptions.)

def agent_tools(toolkit) -> list:
    return []


# --- Bench: the toolkit every consumer loads ---

def build_toolkit() -> Toolkit:
    """Assemble + configure this run's bench. The CLI, agents, and notebooks
    load this too — construct and configure only, never run anything here."""
    tk = assemble_toolkit(task, agent_tools=agent_tools)

    # Auto-discovers available backends (CLI tools, API keys, local servers).
    # Run "groundhog backends" to see what's available on your machine.
    # None when nothing is found — loading stays LLM-free; strategies that
    # need an LLM fail loudly at run time.
    llm = auto_registry()
    if llm:
        tk.llm = llm

    # Or configure manually — uncomment and customize:
    # from groundhog import BackendRegistry, GeminiBackend, AnthropicBackend, OpenAICompatibleBackend, ClaudeCodeBackend
    # tk.llm = BackendRegistry(
    #     high=AnthropicBackend(model="claude-opus-4-6"),                # best reasoning
    #     default=ClaudeCodeBackend(model="sonnet"),                     # via Claude Code CLI
    #     cheap=OpenAICompatibleBackend.ollama(model="llama3"),          # free local model
    # )
    return tk


# --- Run: the ONLY place anything executes ---

if __name__ == "__main__":
    import sys

    optimizer = SimpleOptimizer(build_toolkit(), strategy=Improve())

    if len(sys.argv) > 1 and sys.argv[1] == "status":
        optimizer.status()
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        optimizer.run(n=n)
