# Changelog

## Unreleased — toolkit-first refactor

**Breaking:** `SimpleOptimizer(toolkit, ...)` — the optimizer now consumes a
finished toolkit instead of building one. Every `task.py` exposes the run-dir
contract `def build_toolkit() -> Toolkit` (assemble + configure, never run).

- New `assemble_toolkit(task, *, history, learnings, path, through,
  agent_through, seed, selection, agent_tools)` — the one place a complete,
  standalone toolkit is built.
- `SelectionPolicy` (data on `toolkit.selection`): prior selection is a
  standing toolkit capability; tuning replaces the policy, never the function.
- Per-task agent tools: optional module hook `agent_tools(toolkit)` in
  task.py, merged with precedence strategy > task > default (shadow-logged).
- `toolkit.ws` attempt pointer: `set_attempt()/clear()` + `attempt()` bracket;
  hard-fail reads; works in strategies AND the CLI.
- `history.materialize(attempt)`: ensure-on-disk (git worktrees recreated from
  the object store — synced clones self-heal); `GitAttempt.path` lazy.
- New CLI: `groundhog tool list` / `tool run NAME [--attempt ID] [-p k=v]`.
- Loader rewritten: import + `build_toolkit()`; the monkeypatch/var-scan
  sandbox is gone.
- Fixes: SEARCH/REPLACE edits silently dropped on trailing-whitespace
  fallback; subprocess results corrupted by user prints; reap killing live
  sessions; Direction families blank on the git backend; fix-phase agents
  missing prior tools.
- Tooling: ruff + pre-commit; publish gate runs the full suite on the full
  matrix + tag/version check; LICENSE file; version single-sourced from
  `groundhog.__version__`.

## 0.2.18 and earlier

See git history (`git log --oneline v0.2.18`).
