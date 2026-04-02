# /// script
# dependencies = ["groundhog-researcher", "python-dotenv"]
# ///
"""My optimization task."""

from dotenv import load_dotenv
load_dotenv()

from groundhog import (
    Task, Data, Context, Evaluator, EvalStage, StageResult,
    SimpleOptimizer, Improve, FreshApproach, CrossPollinate,
    auto_registry,
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
                      lambda cp: self.evaluate(cp, data)),
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


# --- Optimizer: runs the loop ---

if __name__ == "__main__":
    import sys

    optimizer = SimpleOptimizer(task, strategy=Improve())

    # Auto-discovers available backends (CLI tools, API keys, local servers)
    # Run "groundhog backends" to see what's available on your machine
    optimizer.toolkit.llm = auto_registry()

    # Or configure manually — uncomment and customize:
    # from groundhog import BackendRegistry, GeminiBackend, AnthropicBackend, OpenAICompatibleBackend, ClaudeCodeBackend
    # optimizer.toolkit.llm = BackendRegistry(
    #     high=AnthropicBackend(model="claude-opus-4-6-20260205"),      # best reasoning
    #     default=ClaudeCodeBackend(model="sonnet"),                     # via Claude Code CLI
    #     cheap=OpenAICompatibleBackend.ollama(model="llama3"),          # free local model
    # )

    if len(sys.argv) > 1 and sys.argv[1] == "status":
        optimizer.status()
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        optimizer.run(n=n)
