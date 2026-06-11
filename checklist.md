# Release Checklist — Consistency Manifest

Things that can drift out of sync and can't be caught by tests.

Information is duplicated across code, templates, README, vault docs, and CLI output. When the code changes, these copies don't update themselves. This checklist tracks every place where duplicated information needs manual verification.

**What this is NOT:** a test suite. Tests catch "does the function work." This catches "does the template still describe what the function actually does."

## Project principles

- [ ] New code since last release checked against the design principles
      in `CLAUDE.md` (§ Design principles — base/ purity, strategies own
      the loop, composed method, raw results, capabilities, config,
      nothing discarded, derived trunks)

## Automated checks (run these first)

- [ ] `pytest` — full suite green (local convenience; CI has no pytest)
- [ ] `uv run python tests/test_concepts.py` — all pass
- [ ] `uv run python tests/test_agent_concepts.py` — all pass
- [ ] `uv run python tests/test_attempt_logger.py` — all pass
- [ ] CI runs test files **as scripts** — a new test file needs a
      `__main__` runner, no pytest fixtures, and entries in BOTH
      workflows (test.yml + publish.yml)
- [ ] `uv build` — package builds
- [ ] `uv run python -c "from groundhog import *"` — all exports work

## Templates

Each template has a specific role:

- **basic** (`init`) — Quick start for humans. Brief comments showing where to fill in logic. Minimal, clean, not overwhelming. Should be the simplest possible working task.
- **llm_guide** (`init-llm`) — Comprehensive documentation embedded in comments. An LLM agent reading this file should understand every concept, option, and best practice without needing external docs. Covers: Data, Context, Evaluator, stages, scorers, strategies, rotation, tiers, backends, auto_registry, queue, config introspection, subprocess execution.
- **mock** (`init-mock`) — Working example that runs without an API key. Deterministic, fast. Demonstrates the full strategy pattern (select → workspace → work → evaluate → commit) with a mock strategy. For testing the framework itself.
- **mnist** (`init-mnist`) — Real ML task. Shows subprocess execution, multiple evaluation stages, multi-strategy rotation. Uses auto_registry with commented manual override.
- **strategy** (`new strategy`) — Custom strategy template. Documents all toolkit capabilities, Config pattern, composed method, retries, core_direction.md inheritance, and attempt-event logging (costs ride on events; totals derive from the log).
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
- [ ] Optimizer.md's State section still matches the persistent pieces (task / learnings / attempts); per-implementation layout lives in Folder Attempt History.md

What to walk (no list here — a list in a drift-checklist drifts):
- `_Structure.md` is the map; follow its tree
- every L3 doc in `Optimizer/Implementation Details/` must match the code it describes

## Strategies

- [ ] All strategies follow the design principles (CLAUDE.md): composed method, Config with param(), hasattr checks
- [ ] All strategies use `extract_code(response.text, prior_code)` — no `_apply_response`, no direct `parse_diff`/`apply_diff`
- [ ] core_direction.md: fresh strategies mint a new one, Improve/CrossPollinate inherit and re-enforce the parent's (legacy approach.md read as fallback)
- [ ] Improve system prompt reflects current research methodology
      (per vault `Optimizer/Research Methodology/Overview.md`)

## Backends

- [ ] Default model IDs are not deprecated (check provider pricing pages)
- [ ] Pricing dicts match current provider pricing
- [ ] `auto_registry()` tier priorities reflect current model landscape
      (consult provider pricing pages; the claude-api skill is
      authoritative for Anthropic IDs — canonical IDs have no date suffix)
- [ ] `groundhog --help` lists all commands

## Docs

- [ ] CLAUDE.md matches reality (commands, architecture summary,
      removed-patterns list, smoke instructions)
- [ ] `docs/sandboxing.md` + `docs/agent_system.md` match per-backend
      behavior (permission rules, phase flow)
- [ ] Both workflows (test.yml + publish.yml) list every test file —
      check the list, not just new additions

## Smoke (when behavior changed)

- [ ] If `strategies/`, `optimizers/`, `agents/`, or `tools/attempt_log*.py`
      changed: MNIST smoke ran — `task.py claude 1` (agent) and
      `task.py llm 4` (LLM rotation) — and the new attempt logs were
      eyeballed, not just exit codes

## Version + release

- [ ] Version bumped in BOTH `pyproject.toml` and
      `src/groundhog/__init__.py` (CI rejects a mismatch)
- [ ] Commit message describes the release; no tool attribution
- [ ] Commit approved explicitly; tag + push approved separately —
      pushing tag `v<version>` (must equal both file versions) triggers
      the PyPI publish
