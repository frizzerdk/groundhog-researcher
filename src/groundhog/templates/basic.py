# /// script
# dependencies = ["groundhog-researcher", "python-dotenv"]
# ///
"""My optimization task."""

from dotenv import load_dotenv
load_dotenv()

from groundhog import (
    Task, Data, Context, Evaluator, EvalStage, StageResult,
    SimpleOptimizer, Improve, FreshApproach, CrossPollinate,
    GeminiBackend, BackendRegistry,
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

class MyEvaluator(Evaluator):
    def evaluate(self, code, data):
        # Execute code, measure performance, return metrics
        # Higher score = better. Metrics are flexible dicts.
        return StageResult(metrics={"score": 0.0})

    def get_stages(self, data):
        # Stages run cheapest first. Cascade stops on error.
        return [
            EvalStage("smoke", "Syntax check", lambda code: self._smoke(code)),
            EvalStage("evaluate", "Full evaluation",
                      lambda code: self.evaluate(code, data)),
        ]

    def _smoke(self, code):
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
    optimizer.toolkit.llm = BackendRegistry(
        default=GeminiBackend(model="gemini-2.5-flash"),
    )

    if len(sys.argv) > 1 and sys.argv[1] == "status":
        optimizer.status()
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        optimizer.run(n=n)
