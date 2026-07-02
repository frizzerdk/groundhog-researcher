# Groundhog Researcher

<p align="center">
  <img src="https://raw.githubusercontent.com/frizzerdk/groundhog-researcher/master/docs/mascot.png" width="300" alt="The Grindhog - Just One More Iteration">
</p>

*They called it Groundhog Day. But what was the groundhog doing? Running experiments. Keeping notes. Getting better, one iteration at a time.*

LLM-powered function optimization. Define how to score your code, and the groundhog iterates overnight. Wake up to a better solution.

## Install

```bash
uv tool install groundhog-researcher
```

This gives you the `groundhog` CLI (and `ghg` alias).

## Quick start

```bash
# See what LLM backends you have available
groundhog backends

# Scaffold a task
groundhog init my_task           # basic template
# groundhog init-llm my_task    # detailed template with full LLM guide
# groundhog init-mock my_task   # mock task, no LLM needed, for testing
# groundhog init-mnist my_task  # 50-sample MNIST digit classification, real ML task

cd my_task
uv run python task.py 10

# Check progress anytime
uv run python task.py status
```

Works with whatever LLM you have — Claude Code, GitHub Copilot, API keys, Ollama. `auto_registry()` discovers available backends automatically.

### Prefer a backend

```bash
groundhog prefer copilot                              # use copilot for all tiers
groundhog prefer-tier max copilot claude-sonnet-4.6    # override one tier
groundhog prefer reset                                 # back to auto-discovery
```

## CLI commands

```
groundhog init [dir]              Basic task template
groundhog init-llm [dir]          Detailed template with full LLM guide
groundhog init-mock [dir]         Mock task (no LLM needed, for testing)
groundhog init-mnist [dir]        MNIST example (real ML task)

groundhog new strategy [file]     Custom strategy template
groundhog new backend [file]      Custom backend template

groundhog backends                Show available backends and tier assignments
groundhog prefer <backend>        Prefer a backend for all tiers
groundhog prefer-tier <tier> <backend> [model]
groundhog prefer reset            Reset all preferences

groundhog attempt <subcommand>    Manual attempt lifecycle: new (--fresh founds a
                                  new family) / list / show / commit (gated,
                                  --strategy labels the producer) / abort /
                                  resume / reap / in-progress / best
groundhog eval <path-or-id>       Score a solution dir, .py file, or attempt
groundhog tool list|run           Run any toolkit tool from the terminal
                                  (check-gates ships as a framework default)
groundhog skills install [dir]    Install the Claude Code session skills into a
                                  run dir (every init does this automatically)

groundhog --version
```

## Write a task from scratch

One file is everything:

```python
# /// script
# dependencies = ["groundhog-researcher", "python-dotenv"]
# ///

from dotenv import load_dotenv
load_dotenv()

from groundhog import (
    Task, Data, Context, Evaluator, EvalStage, StageResult,
    Toolkit, assemble_toolkit, SimpleOptimizer, Improve, auto_registry,
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
        code = code_or_path if isinstance(code_or_path, str) else (code_or_path / "solution.py").read_text(encoding="utf-8")
        return StageResult(metrics={"score": 0.85, "time": 1.2})

    def get_stages(self, data):
        return [
            EvalStage("smoke", "Quick syntax check", lambda cp: ...),
            EvalStage("evaluate", "Full evaluation",
                      lambda cp: self.evaluate(cp, data)),
        ]


task = Task(data=MyData(), context=MyContext(), evaluator=MyEvaluator(), name="MyTask")


def build_toolkit() -> Toolkit:
    """The run-dir contract: assemble + configure the bench, never run.
    The CLI and agents load this file through it — side-effect free."""
    tk = assemble_toolkit(task)
    llm = auto_registry()      # None on a keyless machine
    if llm:
        tk.llm = llm
    return tk


if __name__ == "__main__":
    optimizer = SimpleOptimizer(build_toolkit(), strategy=Improve())
    optimizer.run(n=100)
```

## What happens

The optimizer works in the current directory:

```
my_task/
    task.py                 # your task definition + entry point
    .env                    # API keys (optional)
    learnings.md            # what the optimizer has learned
    attempts/               # every candidate
        001_none_done/      # first attempt (no parent; committed attempts
            solution.py     #   gain a _done/_fail suffix)
            result.json
            attemptlog.jsonl    # every event: prompts, responses, evals, costs
            attemptlog.md       # the same, human-readable
        002_1_done/         # second attempt (parent=1)
            ...
```

The loop:
1. Selects a **prior** (best attempt, weighted by potential)
2. Runs a **strategy** (improve, explore, combine ideas)
3. Evaluates -- raw results stored, scored via stage scorers
4. Records the attempt in an immutable **attempt tree**
5. Updates **learnings**
6. Repeats

Every attempt is kept. Nothing discarded. Change what "good" means later -- the history is reinterpretable.

## Multi-strategy optimizer

Run multiple strategies in a weighted rotation:

```python
from groundhog import Improve, FreshApproach, CrossPollinate

optimizer = SimpleOptimizer(
    build_toolkit(),
    strategies=[
        (Improve(), 14),                          # 14 refinement steps
        (CrossPollinate(), 5),                     # 5 cross-pollination steps
        (FreshApproach(mode="different"), 1),       # 1 fresh exploration
    ],
    seed_strategy=FreshApproach(mode="blank"),      # seed with pure exploration
)
```

## Backend tiers

`auto_registry()` discovers what's available and assigns 5 tiers (max, high, default, budget, cheap). Or configure manually:

```python
from groundhog import BackendRegistry, AnthropicBackend, GeminiBackend, OpenAICompatibleBackend

tk.llm = BackendRegistry(          # inside build_toolkit()
    high=AnthropicBackend(model="claude-opus-4-6"),
    default=GeminiBackend(model="gemini-2.5-flash"),
    cheap=OpenAICompatibleBackend.ollama(model="llama3"),
)
```

Override individual tiers after auto-discovery:

```python
optimizer.toolkit.llm = auto_registry()
optimizer.toolkit.llm.set("high", AnthropicBackend(model="claude-opus-4-6"))
```

Available backends:

| Type | Backends |
|------|----------|
| **API** | `OpenAICompatibleBackend` (.openai, .deepseek, .groq, .cerebras, .xai, .together, .fireworks, .ollama, .openrouter, ...), `AnthropicBackend`, `GeminiBackend` |
| **CLI** | `ClaudeCodeBackend`, `CopilotBackend`, `GeminiCLIBackend`, `OpenCodeBackend` |

## Building a custom strategy

```bash
groundhog new strategy my_strategy.py    # generates a documented template
```

Or subclass directly:

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
        ws = toolkit.history.workspace(parent=None)
        # ... generate code, write to ws.path / "solution.py" ...
        result = toolkit.task.evaluate(ws.path, through="evaluate")
        from groundhog.utils.finalize import finalize_attempt
        attempt = finalize_attempt(toolkit, ws, result, None,
                                   strategy="mine")
        return {"attempt": attempt.id}
```

## Core concepts

| Concept | What it is |
|---------|-----------|
| **Task** | Your problem: data + context + evaluator |
| **Strategy** | An action that moves the state forward (improve, explore, combine, analyse) |
| **Attempt History** | Immutable tree of every candidate and its raw results |
| **Scorer** | Per-stage callable that maps raw results to scores -- change it without re-running |
| **Learnings** | What the optimizer has learned about how to optimize your task |
| **Toolkit** | Capabilities available to strategies (LLMs, history, learnings, logging) |

## Architecture

```
base/           # interfaces only -- Task, Strategy, Optimizer, AttemptHistory, etc.
strategies/     # Improve, FreshApproach, CrossPollinate, Analyse + agent variants
optimizers/     # SimpleOptimizer
histories/      # FolderAttemptHistory, GitAttemptHistory
backends/       # Gemini, Anthropic, OpenAI-compatible, Claude Code CLI, + more
agents/         # agent backends -- Claude Code, Gemini CLI, Copilot, Codex, OpenCode
learnings/      # MarkdownLearnings
acceptance/     # DefaultAcceptance
tools/          # attempt_logger, attempt_log (console), cost_estimate, queue
utils/          # codegen, subprocess_runner, selection, gates, finalize
templates/      # task scaffolding templates (used by groundhog init)
skills/         # packaged Claude Code session skills (groundhog skills install)
```

`base/` defines interfaces. Everything else is implementations. Build your own by subclassing the interfaces.

## License

MIT
