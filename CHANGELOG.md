# Changelog

## 0.3.0 (2026-07-02) — toolkit-first refactor + interactive attempt sessions

**Interactive attempt sessions:** a Claude Code session (or a charter-bound
subagent) is now a first-class attempt producer — same store, same
lifecycle, same gates as the automated strategies.

- `toolkit.gates` — legitimacy gates as pure facts on the bench:
  `evaluate(ws, parent) -> [GateViolation]` (direction present/unique for
  fresh attempts; modified inherited direction and byte-identical solution
  as flags). Responses stay with the caller.
- `toolkit.finalize` / `finalize_attempt(...)` — the standard finish as one
  helper (promote/restore direction → gates → record → commit → score
  note). A convention, never a contract: the Strategy contract does not
  mention it and every composed piece stays public.
- `check-gates` — first framework-default agent tool: the mid-work
  self-check, also `groundhog tool run check-gates --attempt <id>`.
- CLI `attempt commit` runs the gated standard finish on both paths (a
  hard violation records the attempt as failed, never blocks) and takes
  `--strategy LABEL` so history is analyzable by producer (manual /
  session / session-swarm / ...).
- `attempt new --fresh` — parentless workspace, the way a session founds a
  new direction family.
- `groundhog skills install` + automatic install on every `init*`: the six
  session skills (interface, in-attempt, fresh, improve, iterate,
  orchestrate) ship with the package into `<run>/.claude/skills/`.
- Hardening from a 31-agent review + a full validation campaign: failed
  attempts resolvable via `history.get`; `attempt best` tolerates
  result-less commits; post-commit note errors can no longer destroy a
  committed attempt; PlanApproaches→fresh-agent queue naming fixed;
  gemini-cli backend works on Windows; non-UTF8 direction files degrade
  instead of crashing; folder workspace allocation retries Windows
  delete-pending contention.

## Previously unreleased — toolkit-first refactor

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
