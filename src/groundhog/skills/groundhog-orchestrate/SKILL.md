---
name: groundhog-orchestrate
description: Fans a groundhog run out to K parallel subagent attempts - pick K angles with user approval, pre-open K workspaces, spawn one-goal charter subagents running under groundhog-in-attempt, then commit every result (losers committed as failed, never silently aborted), distill learnings, and report. Use when the user invokes /groundhog-orchestrate K or asks to try several directions in parallel on the current run. Builds on groundhog-interface; orchestrator owns the lifecycle, subagents own the files.
argument-hint: <K> [pair|checkpoint|auto]
---

# Groundhog orchestrate

You are the orchestrator for one fan-out round: K subagents, each
pinned to its own pre-opened workspace, each testing exactly one angle.
Load the `groundhog-interface` skill first (in a scaffolded run dir it
lives at `.claude/skills/groundhog-interface/SKILL.md`) —
lifecycle, gates, honesty rules, and the autonomy model all live there.
This file adds only the fan-out workflow.

**Role split — the one fixed contract.** You own the lifecycle:
only you run `groundhog attempt new` / `commit` / `resume`, and only
you write to the run's `learnings.md`. Subagents own the files: they
edit only inside their assigned workspace and score it with
`groundhog eval`. Everything inside an angle is the subagent's call.

**Autonomy default for this mode.** One checkpoint: angle selection
(the user approves the K angles before any workspace opens). Below
that, run at auto — never pause; log decisions for the final report;
on anomaly, safe-stop (commit what exists, then report — don't block
waiting). The user can move the level in plain language at any time.

## Workflow

1. **Orient.** From the run dir: `uv run task.py status` (family map,
   trunks), `groundhog attempt list`, `groundhog attempt best`, and
   read `learnings.md`. Note which families are crowded, which priors
   are strong, and which directions are untouched.

2. **Pick K angles — checkpoint.** Each angle is one sentence and is
   either *improve* (names a prior attempt id) or *fresh* (names a new
   direction that is unique vs existing families — the dedup gate
   rejects duplicates, and two fresh angles must also differ from each
   other). Present the K angles with rationale; wait for approval.
   This is the only pause in the whole mode.

3. **Pre-open K workspaces** (all before spawning anything):
   - improve angle: `groundhog attempt new --parent <id>`
     (seeds from the prior; direction is inherited)
   - fresh angle: `groundhog attempt new --fresh`
     (parentless empty workspace — commits through the fresh-direction
     gates; the charter tells the subagent to create
     `core_direction.md`)
   Record each printed wsid and workspace path — the charters need
   them, and you commit by wsid later.

4. **Spawn K subagents in parallel**, one per workspace. Each
   subagent's prompt is exactly two parts: (1) the `groundhog-in-attempt`
   skill — reference it by name, point at
   `.claude/skills/groundhog-in-attempt/SKILL.md`, or paste that file's
   contents at the top of the prompt if the subagent can't load skills;
   (2) the charter below. **Subagents never pause** for user input at
   any level — the charter says so, the in-attempt skill says so, and
   any question they have comes back to you inside their report.

5. **Collect and commit — every workspace, no exceptions.**
   Read each report, then finalize:
   - subagent produced a solution:
     `groundhog attempt commit <wsid> --eval --strategy session-swarm`
     (the eval at commit is the canonical score; done vs fail follows
     from whether it completes)
   - subagent crashed, went off-charter, or left nothing runnable:
     `groundhog attempt commit <wsid> --fail --strategy session-swarm`
   - gotcha: if `commit ... --eval` errors out ("Commit failed"), the
     workspace is still open — fall back to
     `groundhog attempt commit <wsid> --fail --strategy session-swarm`
     so the loser is recorded.

6. **Distill learnings.** Append cross-attempt insights to the run's
   `learnings.md`, one entry per insight, entries separated by `---`
   on its own line with blank lines around it (the store's separator).
   Failed angles are half the value — record what didn't work and why.

7. **Report.** One row per angle: angle → attempt id, done/fail,
   score, one-line takeaway. Compare the round's best against the
   prior best (`groundhog attempt best`), list gate failures and
   anomalies, and name the learnings entries you added.

## Charter template (one per subagent, keep it this small)

```
# Charter — attempt <wsid>
Run dir:   <absolute run dir>   (run all groundhog commands from here)
Workspace: <absolute ws path>   (your sandbox — write ONLY under it; work/ for scratch)
Goal:      <one sentence — the single angle this attempt exists to test>
Direction: inherited — preserve core_direction.md exactly as-is
       OR: create core_direction.md; FIRST LINE = the approach name itself
Evaluate:  groundhog eval <ws path> --json   (scores come ONLY from this)
Budget:    <N> eval cycles max, then leave your best state on disk and report
Report:    final eval output, what you changed, what you'd try next, open questions
Rules:     groundhog-in-attempt applies. NEVER pause for user input. No attempt
           new/commit/abort, no other attempts, no learnings.md — the
           orchestrator commits your workspace and asks your questions upward.
```

## Losers commit as failed — never silent-abort

**Every workspace a subagent touched gets committed**, as done or as
`--fail`. A failed commit is recorded history: selection skips it, the
family map shows the angle was tried, and the next round doesn't
re-spend a subagent on it. An aborted workspace is invisible and the
angle gets retried blindly. The only abortable workspace is one whose
subagent never started (spawn failed, zero writes) — and even then,
say so in the report; abort is allowed, *silent* abort is not.

## Fleet conventions — when the fan-out gets large

A big round (many workers, long runs, or more than one orchestrator)
breaks in ways a K=3 round never shows. These seven rules are the
scars of a large real campaign; `docs/fleet_conventions.md` carries the
full rationale. Hold to them and silent loss stops happening.

1. **Single-writer lifecycle.** Exactly ONE process runs `attempt new`
   / `commit` / `abort`. Parallel workers never touch the lifecycle —
   they bank their results as manifests in their own scratch dirs, and
   one collector folds every manifest into the store. Two writers on
   one store is how attempts vanish.

2. **Manifest → collector → reconcile.** Each worker writes a
   `{solves, metrics}` manifest; the fold compares realized (what
   landed in the store) against claimed (what the manifests say) and
   *investigates every mismatch* rather than trusting the total. This
   reconcile step caught every silent-loss bug in the campaign — a run
   with no reconcile is a run you can't trust the count of.

3. **Foreground-only for agent work.** Never park agent runs in the
   background and wait — background waits stall the agent. Chunk a long
   run into foreground segments instead.

4. **Never `cd` into a workspace.** A shell left parked with its CWD
   inside a workspace blocks the folder backend from committing it on
   Windows (`WinError 5`, the dir is in use). Run every `groundhog`
   command from the run dir; pass the workspace by path, never by
   walking into it.

5. **Fail-loud chaining.** Verify each pipeline step's output exists
   and is fresh before feeding the next step. A masked step failure let
   a stale artifact get submitted twice — check, don't assume the
   previous stage wrote what you expected.

6. **Resume over respawn.** A stalled agent resumed with its context
   (`groundhog attempt resume <wsid>`) beats a fresh one that has to
   rediscover everything. Reach for respawn only when there is nothing
   left to resume into.

7. **Concurrent orchestrators coordinate via a claims file.** When more
   than one orchestrator runs at once, they share a claims file
   recording who owns which dirs and workspaces, and each keeps
   one-open-workspace-at-a-time. No claim, no touch.

**Namespacing.** Parallel workers that generate code MUST give their
modules unique stems (prefix by wsid, angle, or worker id). Two workers
writing same-named modules collide in `sys.modules` — the second import
silently returns the first worker's module and shadows the file on
disk, so a worker evaluates someone else's code without any error.

## Recovery and boundaries

- If your session crashed mid-round: `groundhog attempt in-progress`
  shows the workspaces (CRASHED once the heartbeat lapses); take each
  back with `groundhog attempt resume <wsid>`, then commit per step 5.
- Never hand-write `result.json` or scores — for you or a subagent
  (interface honesty rules). If a report claims a score, the commit
  `--eval` is still what counts.
- One attempt at a time sequentially → use groundhog-iterate; a single
  attempt → groundhog-fresh or groundhog-improve. This mode is only
  for parallel fan-out.
