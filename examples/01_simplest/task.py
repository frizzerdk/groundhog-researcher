# /// script
# dependencies = ["groundhog-researcher"]
# ///
"""The simplest possible groundhog run — one file, no LLM, no API keys.

Task: write ``solve()`` returning a number close to a hidden target.
Strategy: a deterministic hill-climber (stands in for an LLM strategy).
Storage: the folder backend — attempts land in ``attempts/NNN_parent/``,
accumulated notes in ``learnings.md``, both beside this file.

Run it:                 python task.py 5
Inspect it (no run):    groundhog attempt list   /   groundhog eval <id>
"""

from pathlib import Path

from groundhog import (
    Task, Data, Context, Evaluator, EvalStage, StageResult,
    Toolkit, assemble_toolkit, SimpleOptimizer,
)


TARGET = 63.25


class GuessData(Data):
    def get_train(self): return {"hint": "the target is between 0 and 100"}
    def get_test(self): return {"target": TARGET}


class GuessContext(Context):
    def get_brief(self): return "Write solve() returning a float near the hidden target."
    def get_extended(self): return "def solve() -> float   # closer is better"


class GuessEvaluator(Evaluator):
    def evaluate(self, code_or_path, data):
        code = code_or_path if isinstance(code_or_path, str) else \
            (Path(code_or_path) / "solution.py").read_text(encoding="utf-8")
        ns = {}
        try:
            exec(code, ns)
            value = float(ns["solve"]())
        except Exception as e:  # noqa: BLE001 — task contract: errors -> StageResult
            return StageResult(errors={"crash": str(e)})
        target = data.get_test()["target"]
        return StageResult(metrics={"value": value, "distance": abs(value - target)})

    def get_stages(self, data):
        return [EvalStage(
            "evaluate", "distance to target",
            lambda cp, d=data: self.evaluate(cp, d),
            scorer=lambda r: -1.0 if r.errors else max(0.0, 1.0 - r.metrics["distance"] / 100.0),
        )]


task = Task(data=GuessData(), context=GuessContext(),
            evaluator=GuessEvaluator(), name="GuessTheNumber")


# --- The run-dir contract: assemble + configure the bench, never run --------

def build_toolkit() -> Toolkit:
    return assemble_toolkit(task, path=Path(__file__).parent)


# --- A deterministic strategy (what an LLM strategy does, without the LLM) --

class HillClimb:
    """Take the best prior, nudge its value toward a better score."""

    def __call__(self, toolkit, config=None):
        from groundhog.utils.results import write_result

        prior = toolkit.get_prior(toolkit)
        base = 50.0
        if prior is not None and prior.result.completed:
            base = prior.result.stages["evaluate"].metrics["value"]
        step = (toolkit.rng.random() - 0.3) * 10.0   # biased upward wander
        guess = base + step

        ws = toolkit.history.workspace(parent=prior.id if prior else None)
        code = f"def solve():\n    return {guess!r}\n"
        (ws.path / "solution.py").write_text(code, encoding="utf-8")
        result = toolkit.task.evaluate(ws.path)
        write_result(ws.path, result, metadata={"strategy": "hillclimb", "cost": 0.0})
        toolkit.learnings.add(f"guessed {guess:.2f} -> "
                              f"distance {result.stages['evaluate'].metrics.get('distance', -1):.2f}")
        ws.commit(success=result.completed)
        return {"value": guess}


if __name__ == "__main__":
    import sys

    optimizer = SimpleOptimizer(build_toolkit(), strategy=HillClimb(), seed_strategy=None)
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        optimizer.status()
    else:
        optimizer.run(n=int(sys.argv[1]) if len(sys.argv) > 1 else 5)
