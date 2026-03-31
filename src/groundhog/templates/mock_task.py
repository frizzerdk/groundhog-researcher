# /// script
# dependencies = ["groundhog-researcher"]
# ///
"""Mock task for testing the optimization loop.

Deterministic, fast, no LLM needed. Set seed for repeatable tests.
Task: generate a function that returns a number as close to a random target as possible.
"""

import random

from groundhog import Task, Data, Context, Evaluator, EvalStage, StageResult


class MockData(Data):
    def __init__(self, seed=42):
        import random
        rng = random.Random(seed)
        self._target = rng.uniform(0, 100)

    def get_train(self):
        return {"target": round(self._target), "range": (0, 100)}

    def get_test(self):
        return {"target": self._target, "range": (0, 100)}


class MockContext(Context):
    def get_brief(self):
        return "Write solve() that returns a float between 0 and 100."

    def get_extended(self):
        return "def solve() -> float: return a number. Closer to the hidden target is better."


class MockEvaluator(Evaluator):
    def evaluate(self, code, data):
        namespace = {}
        exec(code, namespace)
        result = namespace["solve"]()
        target = data.get_test()["target"]
        distance = abs(result - target)
        score = max(0.0, 1.0 - distance / 100.0)
        return StageResult(
            score=score,
            metrics={"distance": distance, "value": result, "target": target},
        )

    @staticmethod
    def _distance_scorer(result):
        if result.errors:
            return -1.0
        if "distance" in result.metrics:
            return max(0.0, 1.0 - result.metrics["distance"] / 100.0)
        return 0.0

    @staticmethod
    def _smoke_scorer(result):
        return -1.0 if result.errors else 1.0

    def get_stages(self, data):
        return [
            EvalStage("smoke", "Check code parses and solve() exists",
                      lambda code: self._smoke(code),
                      scorer=self._smoke_scorer),
            EvalStage("validate", "Quick eval on rounded target",
                      lambda code, d=data: self._validate(code, d),
                      scorer=self._distance_scorer),
            EvalStage("evaluate", "Full evaluation",
                      lambda code, d=data: self.evaluate(code, d),
                      scorer=self._distance_scorer),
        ]

    def _smoke(self, code):
        try:
            namespace = {}
            exec(code, namespace)
            if "solve" not in namespace:
                return StageResult(errors={"missing": "No solve() function defined"})
            result = namespace["solve"]()
            if not isinstance(result, float):
                return StageResult(errors={"type": f"solve() returned {type(result).__name__}, expected float"})
            return StageResult(score=1.0, metrics={"returns_float": 1.0})
        except Exception as e:
            return StageResult(errors={"syntax": str(e)})

    def _validate(self, code, data):
        namespace = {}
        exec(code, namespace)
        result = namespace["solve"]()
        rounded_target = round(data.get_test()["target"])
        distance = abs(result - rounded_target)
        score = max(0.0, 1.0 - distance / 100.0)
        return StageResult(
            score=score,
            metrics={"distance": distance, "value": result, "rounded_target": rounded_target},
        )


class MockTask(Task):
    def __init__(self, seed=42):
        super().__init__(
            data=MockData(seed=seed),
            context=MockContext(),
            evaluator=MockEvaluator(),
            name="MockTask",
        )


if __name__ == "__main__":
    from mock_strategy import MockStrategy
    from groundhog import SimpleOptimizer

    task = MockTask(seed=42)
    strategy = MockStrategy()

    print(f"Task: {task.name}")
    print(f"Target: {task.data.get_test()['target']:.4f}")
    print()

    import sys
    optimizer = SimpleOptimizer(task, strategy=strategy, seed=69, seed_strategy=None)

    if len(sys.argv) > 1 and sys.argv[1] == "status":
        optimizer.status()
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
        optimizer.run(n=n)
