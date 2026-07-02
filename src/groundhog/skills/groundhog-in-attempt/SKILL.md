---
name: groundhog-in-attempt
description: Charter constraints for an agent working inside exactly ONE pre-opened groundhog attempt workspace. Use when an orchestrator has pinned this session to a single workspace with a one-goal charter. Covers evaluating the workspace, running task tools, and core-direction obligations, and enforces the hard boundaries — hand-write only inside the workspace, never commit or abort, never fabricate scores, never pause for input, report back instead.
---

# Groundhog in-attempt

You own exactly one attempt workspace, assigned by an orchestrator
together with a one-goal charter. Improve the solution in that
workspace, evaluate it honestly, and report back. Everything else —
committing, aborting, other attempts, learnings — belongs to the
orchestrator, not you.

## Your sandbox

- **Hand-write files ONLY inside your assigned workspace directory.**
  `solution.py` at the workspace root is the deliverable that gets
  scored. `work/` is yours for scratch — notes, experiments,
  intermediate files.
- Do not write anywhere else: not other attempts or workspaces, not
  the run's `task.py`, not `learnings.md`, not store internals.
  Reading the run is fine; writing outside your workspace is not.
- Your edits live on disk and survive crashes — no manual
  checkpointing needed.

## Evaluating your workspace

`groundhog` commands work from inside the workspace (the CLI walks up
parent dirs to find the run's `task.py`).

    groundhog eval <ws-dir> [--through STAGE] [--json]

- `<ws-dir>` — your workspace path (`.` from inside it). Also accepts
  a lone `.py` file for quick probes.
- `--through STAGE` — stop after a named pipeline stage (cheaper
  partial check).
- `--json` — machine-readable stages, scores, errors, artifacts.
- Exit 0 = completed; exit 2 = failed a stage (output names it).

**Scores come ONLY from `groundhog eval`.** Never hand-write
`result.json`, never estimate or extrapolate a score. Report eval
output verbatim.

## Task tools

    groundhog tool list
    groundhog tool run <name> [--attempt ID] [-p k=v ...]

`tool list` shows this run's tools with descriptions; `tool run`
invokes one with string params (`-p key=value`, repeatable).
`--attempt ID` points workspace-relative tools at a committed
attempt's files instead.

## Direction obligations

`core_direction.md` at the workspace root is the family identity.

- **Inherited (child attempt): preserve it byte-for-byte.** Do not
  edit, reformat, or "improve" it — the direction IS the family
  identity, and changing it forks the family. If you believe the
  direction itself is wrong, say so in your report instead.
- **Fresh (only if your charter says to create one): the FIRST LINE
  is the approach name itself** — short and specific (e.g.
  "rollout-greedy beam with learned ordering"), no "Core Direction"
  heading or label. That line becomes the display name and folder
  slug. Keep the whole file to a few lines describing the
  algorithmic backbone, not implementation details.

## Hard boundaries

- No `groundhog attempt new` / `commit` / `abort` / `resume` /
  `reap` — the attempt lifecycle belongs to your orchestrator.
- No writes to `learnings.md` — propose learnings in your report.
- No touching other attempts' files or workspaces.

## Report back — never pause

**Never stop to wait for user input, at any autonomy level.**
Questions, uncertainty, and pivot ideas travel UP in your report;
they never block work. On an anomaly (eval crashes, impossible
scores, broken workspace), safe-stop: leave files in a consistent
state, then report — do not retry forever, do not wait for help.

Your final report to the orchestrator: what you changed and why, the
last full `groundhog eval` output (overall + per-stage), direction
status (preserved, or created — quote its first line), any anomalies
or open questions, and your recommendation (commit as done, or
commit as failed). Then end your turn — the orchestrator commits.
