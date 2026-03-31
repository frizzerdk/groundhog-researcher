# Release Checklist

Verification steps before publishing a new version. An agent or human should run through all of these.

## Code quality

- [ ] All tests pass: `uv run python tests/test_concepts.py`
- [ ] Package builds: `uv build`
- [ ] All imports work: `uv run python -c "from groundhog import *"`
- [ ] No orphaned files in `utils/`, `tools/`, `strategies/` (everything exported or intentionally internal)

## Templates

Each template has a specific role:

- **basic** (`init`) — Quick start for humans. Brief comments showing where to fill in logic. Minimal, clean, not overwhelming. Should be the simplest possible working task.
- **llm_guide** (`init-llm`) — Comprehensive documentation embedded in comments. An LLM agent reading this file should understand every concept, option, and best practice without needing external docs. Covers: Data, Context, Evaluator, stages, scorers, strategies, rotation, tiers, queue, config introspection, subprocess execution.
- **mock** (`init-mock`) — Working example that runs without an API key. Deterministic, fast. Demonstrates the full strategy pattern (select → workspace → work → evaluate → commit) with a mock strategy. For testing the framework itself.
- **mnist** (`init-mnist`) — Real ML task. Shows subprocess execution, multiple evaluation stages, time budgets, multi-strategy rotation with backend tiers. A complete production-style example.

Checks:
- [ ] `groundhog init /tmp/test_basic` creates a valid project
- [ ] `groundhog init --script /tmp/test_script` creates a valid script
- [ ] `groundhog init-mock /tmp/test_mock` creates a runnable mock task
- [ ] `groundhog init-llm /tmp/test_llm` creates a detailed template with up to date best practices
- [ ] Mock task runs without API key: `cd /tmp/test_mock && uv run task.py 10`
- [ ] All template files have inline script metadata (`# /// script`)
- [ ] Templates reflect current API (imports, class names, method signatures)
- [ ] basic template: brief, clean, not intimidating
- [ ] llm_guide template: comprehensive — covers all concepts, options, and patterns
- [ ] mock template: runs end-to-end with no external dependencies
- [ ] mnist template: demonstrates real multi-strategy optimization

## README

- [ ] Quick start example runs as shown
- [ ] All code snippets use current API (no stale imports or method names)
- [ ] Install instructions work (`uv add groundhog-researcher`)
- [ ] `groundhog init` workflow matches what the CLI actually does
- [ ] Architecture section lists all current directories
- [ ] Core concepts table matches vault definitions

## Vault alignment

- [ ] Every concept in the vault has a corresponding implementation
- [ ] No code contradicts what the vault says
- [ ] New features added since last release are documented in vault
- [ ] Implementation Details/ docs match actual code behavior
- [ ] Workspace layout in Optimizer.md matches actual directory structure

Vault docs to cross-check:
- `Core Idea: Task.md` — Task = Data + Context + Evaluator
- `Scorer.md` — per-stage callable, read-side only
- `Optimizer/Strategy.md` — Config pattern, composed method
- `Optimizer/Attempt History.md` — immutable, atomic, complete
- `Optimizer/Learnings.md` — global, sampling, Analyse compression
- `Optimizer/Toolkit.md` — SimpleNamespace, capabilities listed
- `Optimizer/Acceptance Process.md` — implicit, not a gate
- `Optimizer/Strategy — Selection.md` — potential-weighted, pluggable get_prior
- `Optimizer/Implementation Details/` — all docs match code

## Strategies

- [ ] All strategies follow composed method pattern (`__call__` reads like a story)
- [ ] All strategies have a Config dataclass with `param()` descriptions
- [ ] All strategies check toolkit capabilities with `hasattr()`
- [ ] Strategies that create attempts include `cost` in metadata

## Version

- [ ] Version in `pyproject.toml` is updated
- [ ] CHANGELOG entry written (if maintained)
- [ ] Git tag matches version
