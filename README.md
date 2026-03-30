# Groundhog Researcher

*Just one more iteration...*

LLM-powered function optimization. Define how to score your code, and the groundhog iterates overnight. Wake up to a better solution.

## Install

```bash
pip install groundhog-researcher
```

Requires an LLM API key in `.env`:
```
GEMINI_API_KEY=...
```

## Quick start

One file — task definition and entry point:

```python
from groundhog import (
    Task, Data, Context, Evaluator, EvalStage, StageResult,
    SimpleOptimizer, Improve, FreshApproach, GeminiBackend, BackendRegistry,
)


class MyData(Data):
    def get_train(self):
        return {"inputs": [...], "targets": [...]}

    def get_test(self):
        return {"inputs": [...], "targets": [...]}


class MyContext(Context):
    def get_brief(self):
        return "Write a function that solves X."

    def get_extended(self):
        return "Write `solve(data)` that maximizes Y. Rules: ..."


class MyEvaluator(Evaluator):
    def evaluate(self, code, data):
        # Run code, return StageResult with metrics
        return StageResult(metrics={"score": 0.85, "time": 1.2})

    def get_stages(self, data):
        return [
            EvalStage("smoke", "Quick syntax check", lambda code: ...),
            EvalStage("evaluate", "Full evaluation",
                      lambda code: self.evaluate(code, data)),
        ]


task = Task(data=MyData(), context=MyContext(), evaluator=MyEvaluator(), name="MyTask")

optimizer = SimpleOptimizer(task, strategy=Improve())
optimizer.toolkit.llm = BackendRegistry(
    default=GeminiBackend(model="gemini-2.5-flash"),
)
optimizer.run(n=100)
```

```bash
uv run python my_task.py
```

## What happens

The optimizer creates a workspace:

```
MyTask/
    learnings.md            # what the optimizer has learned
    attempts/               # every candidate
        001_none/           # first attempt (no parent)
            solution.py
            result.json
            conversation.json
        002_1/              # second attempt (parent=1)
            ...
```

Then it loops:
1. Selects a **prior** (best attempt, weighted by potential)
2. Runs a **strategy** (improve, explore, combine ideas)
3. Evaluates — raw results stored, scored via stage scorers
4. Records the attempt in an immutable **attempt tree**
5. Updates **learnings**
6. Repeats

Every attempt is kept. Nothing discarded. Change what "good" means later — the history is reinterpretable.

## Multi-strategy optimizer

Run multiple strategies in a weighted rotation:

```python
optimizer = SimpleOptimizer(
    task,
    strategies=[
        (Improve(), 14),                          # 14 refinement steps
        (CrossPollinate(), 5),                     # 5 cross-pollination steps
        (FreshApproach(mode="different"), 1),       # 1 fresh exploration
    ],
    seed_strategy=FreshApproach(mode="blank"),      # seed with pure exploration
)
```

The schedule cycles — 14 + 5 + 1 = 20 per cycle. FreshApproach creates new trunks, Improve refines them, CrossPollinate transfers ideas between them.

## Backend tiers

Register LLMs by purpose. Strategies request tiers; missing tiers fall back to "default":

```python
optimizer.toolkit.llm = BackendRegistry(
    high=GeminiBackend(model="gemini-3-flash-preview"),        # reasoning tasks
    default=GeminiBackend(model="gemini-3.1-flash-lite-preview"),  # bulk generation
    cheap=GeminiBackend(model="gemini-2.5-flash-lite"),        # cheapest
)

# In a strategy:
response = toolkit.llm.get("high").generate(prompt=prompt)     # uses high
response = toolkit.llm.get("expensive").generate(prompt=prompt) # falls back to default
```

## Building a custom strategy

Subclass `Strategy`, define a `Config`, implement `__call__`:

```python
from dataclasses import dataclass
from groundhog import Strategy, StrategyConfig, param

@dataclass
class MyConfig(StrategyConfig):
    temperature: float = param(0.7, "LLM sampling temperature")
    max_retries: int = param(3, "Retry attempts on failure")

class MyStrategy(Strategy):
    Config = MyConfig

    def __call__(self, toolkit, config=None):
        cfg = self._resolve_config(config)
        # cfg.temperature, cfg.max_retries available

        ws = toolkit.history.workspace(parent=None)
        # ... generate code, write to ws.path / "solution.py" ...
        result = toolkit.task.evaluate(code, through="evaluate")
        attempt = ws.commit(result, metadata={"strategy": "mine"})
        return {"attempt": attempt.number, "strategy": "mine"}
```

Configs are introspectable — any tool can discover what's configurable:

```python
MyStrategy.Config().describe()
# {"temperature": {"type": "float", "default": 0.7, "description": "LLM sampling temperature"}, ...}
```

## Core concepts

| Concept | What it is |
|---------|-----------|
| **Task** | Your problem: data + context + evaluator |
| **Strategy** | An action that moves the state forward (improve, explore, combine, analyse) |
| **Attempt History** | Immutable tree of every candidate and its raw results |
| **Scorer** | Per-stage callable that maps raw results to scores — change it without re-running |
| **Learnings** | What the optimizer has learned about how to optimize your task |
| **Toolkit** | Capabilities available to strategies (LLMs, history, learnings, logging) |

## Architecture

```
base/           # interfaces only — Task, Strategy, Optimizer, AttemptHistory, etc.
strategies/     # Improve, FreshApproach, CrossPollinate, Analyse
optimizers/     # SimpleOptimizer
histories/      # FolderAttemptHistory
backends/       # GeminiBackend, MockBackend
learnings/      # MarkdownLearnings
tools/          # conversation_log, cost_estimate, StrategyLog
utils/          # codegen, subprocess_runner, selection
```

`base/` defines interfaces. Everything else is implementations. Build your own by subclassing the interfaces.

## License

MIT
