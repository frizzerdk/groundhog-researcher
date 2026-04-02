# Release Checklist — Consistency Manifest

Things that can drift out of sync and can't be caught by tests.

Information is duplicated across code, templates, README, vault docs, and CLI output. When the code changes, these copies don't update themselves. This checklist tracks every place where duplicated information needs manual verification.

**What this is NOT:** a test suite. Tests catch "does the function work." This catches "does the template still describe what the function actually does."

## Project principles (check new code against these)

**base/ is interfaces only.** No implementation in base/. If you're adding logic, it goes in a named implementation folder. Base classes should be minimal — the essence of the concept, nothing more.

**Strategies own the full loop.** Select prior → workspace → generate → evaluate → record. Don't split strategy logic into the optimizer. If it's specific to your approach, it lives in your strategy.

**Composed method pattern.** Strategy `__call__` reads like a story — each step is a named method. Implementation details live in the step methods. You should be able to understand what a strategy does by reading `__call__` alone.

**Raw results, never scores.** Attempts store metrics dicts. Scoring is always read-side via per-stage scorers. Never persist a score. This keeps history reinterpretable when "good" changes.

**Toolkit is capabilities, not tools.** SimpleNamespace with `hasattr` checks. Strategies never assume a capability exists — they check and fall back gracefully.

**Config is self-documenting.** Every configurable parameter uses `param(default, "description")`. `Config.describe()` makes it introspectable. If a strategy has a knob, it goes in Config.

**Nothing is discarded.** Every attempt — success or failure — is recorded. Failed attempts inform future strategies.

**Trunks are derived, not stored.** Apply a scorer to the attempt tree to get trunks. Change the scorer, get different trunks. Never persist trunk membership.

## Automated checks (run these first)

- [ ] `uv run python tests/test_concepts.py` — all pass
- [ ] `uv build` — package builds
- [ ] `uv run python -c "from groundhog import *"` — all exports work

## Templates

Each template has a specific role:

- **basic** (`init`) — Quick start for humans. Brief comments showing where to fill in logic. Minimal, clean, not overwhelming. Should be the simplest possible working task.
- **llm_guide** (`init-llm`) — Comprehensive documentation embedded in comments. An LLM agent reading this file should understand every concept, option, and best practice without needing external docs. Covers: Data, Context, Evaluator, stages, scorers, strategies, rotation, tiers, backends, auto_registry, queue, config introspection, subprocess execution.
- **mock** (`init-mock`) — Working example that runs without an API key. Deterministic, fast. Demonstrates the full strategy pattern (select → workspace → work → evaluate → commit) with a mock strategy. For testing the framework itself.
- **mnist** (`init-mnist`) — Real ML task. Shows subprocess execution, multiple evaluation stages, multi-strategy rotation. Uses auto_registry with commented manual override.
- **strategy** (`new strategy`) — Custom strategy template. Documents all toolkit capabilities, Config pattern, composed method, retries, approach.md, cost tracking, conversation logging.
- **backend** (`new backend`) — Custom backend template. Shows both REST API and CLI subprocess patterns, LLMResponse fields, cost/usage tracking, error handling.

Checks:
- [ ] All templates reflect current API (imports, class names, method signatures)
- [ ] All task templates use `auto_registry()` with commented-out manual examples using current model IDs
- [ ] Each template fulfills its role described above
- [ ] Template `.env` files have no placeholder values mistakable for real keys

## README

- [ ] Quick start example uses current API (imports, auto_registry, code_or_path)
- [ ] All code snippets use current class names and method signatures
- [ ] CLI commands shown match what `cli.py` actually implements (init, new, backends)
- [ ] Architecture section lists all current directories
- [ ] Backend tiers section shows auto_registry, manual config, and .set() override
- [ ] Custom strategy example uses current Config/param pattern
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

- [ ] All strategies follow principles above (composed method, Config with param(), hasattr checks)
- [ ] All strategies use `extract_code(response.text, prior_code)` — no `_apply_response`, no direct `parse_diff`/`apply_diff`
- [ ] approach.md: FreshApproach generates, Improve/CrossPollinate copy from parent
- [ ] Improve system prompt reflects current research methodology

## Backends

- [ ] Default model IDs are not deprecated (check provider pricing pages)
- [ ] Pricing dicts match current provider pricing
- [ ] `auto_registry()` tier priorities reflect current model landscape
- [ ] `groundhog --help` lists all commands

## Version

- [ ] Version in `pyproject.toml` is updated
- [ ] CHANGELOG entry written (if maintained)
- [ ] Git tag matches version
