# /// script
# dependencies = ["groundhog-researcher", "python-dotenv"]
# ///
"""The recommended setup — every current best practice in one run dir.

This example is LIVING: it gains every new feature as it lands. Current kit:

- **Git attempt store** (`GitAttemptHistory`): attempts/ is a browsable git
  repo — one commit per attempt, a slug-named worktree folder each, lineage
  as the commit graph. `cd attempts && git log` just works. Add `remote=` and
  two machines share one history, conflict-free (per-origin create-only refs).
- **Compacted learnings**: every note is retained append-only, while a
  condensed view stays small enough to prompt with.
- **Per-task agent tools** (`agent_tools` module hook): your own tools ride
  along to agents AND the CLI (`groundhog tool run ...`), and can read the
  attempt in flight through `toolkit.ws`.
- **SelectionPolicy as data**: tune exploration without touching functions.
- **Lazy Data by convention**: nothing heavy at import — loading this file
  (CLI, agents, notebooks) is instant and side-effect free.
- **build_toolkit() contract**: assemble + configure the bench; only
  `__main__` ever runs anything.

Run it:                 uv run task.py 10
Inspect it (no run):    groundhog attempt list / groundhog tool list
"""

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from groundhog import (
    Task, Data, Context, Evaluator, EvalStage, StageResult,
    Toolkit, agent_tool, assemble_toolkit,
    GitAttemptHistory, Compacted, MarkdownLearnings, SelectionPolicy,
    SimpleOptimizer, Improve, CrossPollinate, FreshApproach, auto_registry,
)

HERE = Path(__file__).parent


# --- The problem (lazy Data: heavy work belongs in get_*, never __init__) ----

class PackingData(Data):
    """20 items, knapsack capacity 50 — small enough to eyeball, hard enough
    to leave headroom above the greedy baseline."""

    def __init__(self):
        self._items = None  # built lazily

    def _ensure(self):
        if self._items is None:
            import random
            rng = random.Random(7)
            self._items = [(rng.randint(1, 12), rng.randint(1, 30)) for _ in range(20)]

    def get_train(self):
        self._ensure()
        return {"items": self._items, "capacity": 50}

    def get_test(self):
        return self.get_train()


class PackingContext(Context):
    def get_brief(self):
        return ("Write solve(items, capacity) -> list of item indices "
                "maximizing total value within the weight capacity.")

    def get_extended(self):
        return (
            "items: list[(weight:int, value:int)], capacity: int.\n"
            "Return indices to take. Overweight selections score 0.\n"
            "Greedy value/weight is the baseline; beat it."
        )


class PackingEvaluator(Evaluator):
    def evaluate(self, code_or_path, data):
        code = code_or_path if isinstance(code_or_path, str) else \
            (Path(code_or_path) / "solution.py").read_text(encoding="utf-8")
        d = data.get_test()
        ns = {}
        try:
            exec(code, ns)
            picks = list(ns["solve"]([tuple(i) for i in d["items"]], d["capacity"]))
            weight = sum(d["items"][i][0] for i in picks)
            value = sum(d["items"][i][1] for i in picks)
        except Exception as e:  # noqa: BLE001 — task contract: errors -> StageResult
            return StageResult(errors={"crash": str(e)})
        if weight > d["capacity"]:
            return StageResult(metrics={"value": 0, "weight": weight, "overweight": 1})
        return StageResult(metrics={"value": value, "weight": weight, "overweight": 0})

    def get_stages(self, data):
        return [
            EvalStage("smoke", "compiles",
                      lambda cp: self._smoke(cp),
                      scorer=lambda r: 0.0 if r.errors else 1.0),
            EvalStage("evaluate", "pack value",
                      lambda cp, d=data: self.evaluate(cp, d),
                      scorer=lambda r: -1.0 if r.errors else r.metrics["value"] / 200.0),
        ]

    def _smoke(self, code_or_path):
        code = code_or_path if isinstance(code_or_path, str) else \
            (Path(code_or_path) / "solution.py").read_text(encoding="utf-8")
        try:
            compile(code, "<solution>", "exec")
            return StageResult(metrics={"compiles": 1.0})
        except SyntaxError as e:
            return StageResult(errors={"syntax": str(e)})


task = Task(data=PackingData(), context=PackingContext(),
            evaluator=PackingEvaluator(), name="Knapsack")


# --- Per-task agent tools: the module hook (never a Task method) -------------
# Preferred authoring: plain module-level functions. Name, description, and
# the agent-visible schema are DERIVED (docstring + signature) — one source
# of truth. The `toolkit` first parameter is injected at invoke time and
# hidden from the agent; through it a tool reads the attempt in flight
# (toolkit.ws), the history, the task. Tools reach agents during runs AND
# the terminal via `groundhog tool run show-pack -p indices="0 3 7"`.

def show_pack(toolkit, indices: str = "") -> str:
    """Score a candidate pack (space/comma-separated indices), or the
    current attempt's solve() when called with no argument."""
    d = toolkit.task.data.get_test()
    if indices.strip():
        picks = [int(x) for x in indices.replace(",", " ").split()]
    else:
        ns = {}
        exec((toolkit.ws.path / "solution.py").read_text(encoding="utf-8"), ns)
        picks = list(ns["solve"]([tuple(i) for i in d["items"]], d["capacity"]))
    weight = sum(d["items"][i][0] for i in picks)
    value = sum(d["items"][i][1] for i in picks)
    verdict = "OVERWEIGHT" if weight > d["capacity"] else "ok"
    return f"picks={picks} weight={weight}/{d['capacity']} value={value} [{verdict}]"


def agent_tools(toolkit) -> list:
    return [agent_tool(show_pack)]


# --- The bench: build_toolkit() — the run-dir contract ------------------------

def build_toolkit() -> Toolkit:
    history = GitAttemptHistory(
        HERE,
        # Two machines, one history: point both at a private repo —
        # remote="https://x-access-token:<token>@github.com/you/your-store.git",
    )
    learnings = Compacted(
        inner=MarkdownLearnings(HERE),          # append-only permanent record
        current_path=HERE / "learnings.current.md",  # the condensed view
        # compactor=make_llm_compactor(...)     # auto-distill via an LLM tier
    )
    tk = assemble_toolkit(
        task,
        history=history,
        learnings=learnings,
        path=HERE,
        through="evaluate",
        selection=SelectionPolicy(direction_weight=0.6),  # tuning is data
        agent_tools=agent_tools,
    )
    llm = auto_registry()      # discovers CLI tools / API keys / local servers
    if llm:                    # None on a keyless machine — loading stays LLM-free
        tk.llm = llm
    return tk


if __name__ == "__main__":
    import sys

    optimizer = SimpleOptimizer(
        build_toolkit(),
        strategies=[(Improve(), 6), (CrossPollinate(), 2), (FreshApproach(mode="different"), 1)],
        seed_strategy=FreshApproach(mode="blank"),
    )
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        optimizer.status()
    else:
        optimizer.run(n=int(sys.argv[1]) if len(sys.argv) > 1 else 10)
