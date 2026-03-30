# Architecture

## Rule

`base/` contains **interfaces and base types only**. Everything else is concrete implementations.
Users import from `groundhog` directly — internal paths are not part of the public API.

## Structure

```
src/groundhog/
    __init__.py              # Public API — re-exports everything

    base/                    # Interfaces and base types (no implementations)
        types.py             # Task, Data, Context, Evaluator, EvalStage, StageResult, EvaluationResult
        attempt_history.py   # Attempt, Workspace, AttemptHistory
        strategy.py          # Strategy
        optimizer.py         # Optimizer
        backend.py           # LLMBackend, LLMResponse, BackendRegistry
        learnings.py         # Learnings
        acceptance.py        # Acceptance
        toolkit.py           # Toolkit

    histories/               # AttemptHistory implementations
        folder.py            # FolderAttemptHistory — directory per attempt

    optimizers/              # Optimizer implementations
        simple.py            # SimpleOptimizer — single strategy loop

    strategies/              # Strategy implementations
        improve.py           # Improve — refine via LLM diffs

    backends/                # LLMBackend implementations
        mock.py              # MockBackend — cycles predefined responses
        gemini.py            # GeminiBackend — REST API

    learnings/               # Learnings implementations
        markdown.py          # MarkdownLearnings — single .md file

    acceptance/              # Acceptance implementations
        default.py           # DefaultAcceptance — compare highest common stage

    utils/                   # Pure utility functions
        codegen.py           # extract_code, parse_diff, apply_diff, build_prompt
```

## Key principles

- Strategies own the full loop (select prior, workspace, generate, evaluate, commit)
- Strategies follow composed method pattern: `__call__` reads as a story
- Attempts store raw results, never scores — scoring is read-side via stage scorers
- Toolkit is a dynamic namespace — strategies check `hasattr` and use defaults
- Prior selection: `toolkit.get_prior` if available, otherwise strategy's own default
