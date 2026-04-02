# Groundhog Researcher

<p align="center">
  <img src="docs/grindhog.jpg" width="300" alt="The Grindhog - Just One More Iteration">
</p>

*Just one more iteration...*

LLM-powered function optimization. Define how to score your code, and the groundhog iterates overnight. Wake up to a better solution.

## Quick start

Scaffold a task folder:

```bash
uv add groundhog-researcher
groundhog init my_task
cd my_task
echo "GEMINI_API_KEY=your-key-here" > .env
# edit task.py with your task logic
uv run task.py 10
```

Check status anytime:

```bash
uv run task.py status
```

Or write a task from scratch — one file is everything:

```python
# /// script
# dependencies = ["groundhog-researcher", "python-dotenv"]
# ///

from dotenv import load_dotenv
load_dotenv()

from groundhog import (
    Task, Data, Context, Evaluator, EvalStage, StageResult,
    SimpleOptimizer, Improve, auto_registry,
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
    def evaluate(self, code_or_path, data):
        # Accepts a code string or a Path to the workspace directory
        code = code_or_path if isinstance(code_or_path, str) else (code_or_path / "solution.py").read_text()
        return StageResult(metrics={"score": 0.85, "time": 1.2})

    def get_stages(self, data):
        return [
            EvalStage("smoke", "Quick syntax check", lambda cp: ...),
            EvalStage("evaluate", "Full evaluation",
                      lambda cp: self.evaluate(cp, data)),
        ]


task = Task(data=MyData(), context=MyContext(), evaluator=MyEvaluator(), name="MyTask")

optimizer = SimpleOptimizer(task, strategy=Improve())
optimizer.toolkit.llm = auto_registry()  # uses whatever LLM you have available
optimizer.run(n=100)
```

Works with whatever LLM you have — Claude Code, API keys, Ollama. Run:

```bash
uv run task.py 10
uv run task.py status         # check progress anytime
groundhog backends            # see available LLM backends
```

## What happens

The optimizer works in the current directory:

```
my_task/
    task.py                 # your task definition + entry point
    .env                    # API keys
    learnings.md            # what the optimizer has learned
    attempts/               # every candidate
        001_none/           # first attempt (no parent)
            solution.py
            result.json
            conversation.json
        002_1/              # second attempt (parent=1)
            ...
```

The loop:
1. Selects a **prior** (best attempt, weighted by potential)
2. Runs a **strategy** (improve, explore, combine ideas)
3. Evaluates — raw results stored, scored via stage scorers
4. Records the attempt in an immutable **attempt tree**
5. Updates **learnings**
6. Repeats

Every attempt is kept. Nothing discarded. Change what "good" means later — the history is reinterpretable.

Check status anytime:

```bash
uv run task.py status
```

## Multi-strategy optimizer

Run multiple strategies in a weighted rotation:

```python
from groundhog import Improve, FreshApproach, CrossPollinate

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

`auto_registry()` discovers what's available and assigns tiers automatically. Or configure manually — strategies request tiers, missing tiers fall back to "default":

```python
from groundhog import BackendRegistry, AnthropicBackend, GeminiBackend, OpenAICompatibleBackend

optimizer.toolkit.llm = BackendRegistry(
    high=AnthropicBackend(model="claude-opus-4-6-20260205"),
    default=GeminiBackend(model="gemini-2.5-flash"),
    cheap=OpenAICompatibleBackend.ollama(model="llama3"),
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
backends/       # Gemini, Anthropic, OpenAI-compatible, Claude Code CLI, + more
learnings/      # MarkdownLearnings
acceptance/     # DefaultAcceptance
tools/          # conversation_log, cost_estimate, StrategyLog, queue
utils/          # codegen, subprocess_runner, selection
templates/      # task scaffolding templates (used by groundhog init)
```

`base/` defines interfaces. Everything else is implementations. Build your own by subclassing the interfaces.

## Install

```bash
uv add groundhog-researcher
# or
pip install groundhog-researcher
```

## License

MIT
