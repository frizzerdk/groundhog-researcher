---
name: groundhog-iterate
description: Runs N groundhog attempts back-to-back as a mini optimizer session — choosing improve vs fresh each iteration from the family map, evaluating and committing through the groundhog CLI, autonomous (auto level) by default. Use when the user asks for a batch of attempts or an unattended optimization stint, e.g. "/groundhog-iterate 8", "run 5 more attempts", "keep optimizing until score 0.9 or 3 attempts without improvement". Stop rules: max N, no-improvement-in-K, score target. Builds on the groundhog-interface skill for CLI capabilities and the autonomy model.
argument-hint: "N [no-improve=K] [score>=X] [pair|checkpoint|auto]"
---

# Groundhog iterate

Run N attempts back-to-back, one at a time, like a mini optimizer
session: orient, pick improve-or-fresh, work the attempt, eval, commit,
repeat until a stop rule fires, then safe-stop and report. This skill
adds ONE goal on top of `groundhog-interface`; everything about HOW to
work an attempt (lifecycle, honesty rules, gates, coexistence, the
autonomy model) lives there.

## Prerequisite: groundhog-interface

Apply `groundhog-interface` throughout. If it is unavailable, these
invariants still bind:

- Scores come ONLY from `groundhog eval` or
  `groundhog attempt commit <wsid> --eval`. Never hand-write
  result.json or report a number you did not get from the CLI.
- Hand-write files only inside your own open workspace (`work/` for
  scratch). Everything structural comes from `groundhog` calls.
- `core_direction.md`: the FIRST LINE is the approach name itself — no
  "Core Direction" heading, no label. Fresh attempts must create it;
  children inherit it and must preserve it.
- Gates are the law, not checkpoints: direction present/unique, no
  byte-identical children. A gate rejection is feedback — record it,
  adapt; never route around it.

## Arguments and stop rules

Parse from the invocation (plain-language phrasings map to these):

- `N` (required) — hard cap on attempts committed this session.
- `no-improve=K` (optional) — stop after K consecutive committed
  attempts that do not beat the running best. The running best starts
  at `groundhog attempt best` before iteration 1.
- `score>=X` (optional) — stop once a committed attempt's overall
  score is >= X.
- level (optional) — `pair` / `checkpoint` / `auto` per the
  groundhog-interface autonomy model. **Default here is `auto`**
  (unlike the single-attempt modes).

**Stop when ANY rule fires**, checked after each commit. Every stop —
rule, anomaly, or user interruption — goes through safe-stop below.

## Autonomy: default auto

At `auto`: never pause for input; the interface's six checkpoints
become log entries for the final report; on anomaly (repeated eval
crashes, store errors, the same gate failing twice, broken
environment) safe-stop and report — never block waiting for the user.
Honor mid-flight level switches in plain language ("go pair for the
next one") and single-checkpoint nudges ("always ask before commits").

## The loop

1. **Orient once.** `groundhog attempt list`, `groundhog attempt best`,
   `uv run task.py status` (trunks + direction families = the family
   map), and read `learnings.md` at the run root.
2. **Choose improve or fresh** for this iteration (next section). At
   auto, log the choice and one-line rationale; at checkpoint/pair,
   pause per the interface model.
3. **Open the workspace.**
   - Improve: `groundhog attempt new --parent <id>` (bare
     `groundhog attempt new` seeds from the current best). Preserve
     the inherited `core_direction.md`.
   - Fresh: `groundhog attempt new --no-seed`, then create
     `core_direction.md` opening a genuinely NEW family — family
     identity is the normalized file content, so differ in substance,
     not just the first line. (Lineage still records a parent; the direction file is
     what opens the new family.)
4. **Work the attempt** per groundhog-interface. Check progress
   honestly with `groundhog eval <ws-dir>` (`--json` to parse,
   `--through STAGE` for partial pipelines).
5. **Commit:** `groundhog attempt commit <wsid> --eval`. Commit weak
   and failed attempts too — recorded history steers later selection.
   Abort (`groundhog attempt abort <wsid>`) only workspaces with
   nothing worth recording.
6. **Log and check stop rules.** Append one row to your iteration log
   (attempt id, parent, improve|fresh, direction, score, delta vs
   running best). If an insight transfers beyond this attempt, append
   an entry to `learnings.md` (entries separated by `---` on its own
   line, blank lines around it). Refresh the family map with
   `uv run task.py status`, then go to 2.

## Improve vs fresh — read the family map

Heuristics, not a formula:

- Improve when a trunk is still climbing (a recent child beat its
  parent) or the best attempt has clear unaddressed weaknesses
  (inspect with `groundhog attempt show <id>`).
- Fresh when top families have plateaued (several children, no new
  best), the map is thin (few families), or `learnings.md` points at
  an untried angle.
- Vary priors: improving the same leader every iteration re-treads one
  trunk — consider other trunk leaders, not just the global best.
- Never re-open a direction that already exists as a family; the dedup
  gate rejects duplicates.

## Safe-stop — never leave a dangling workspace

Before ending the session, resolve the workspace you hold: commit it
(`--eval`, plus `--fail` if it should count as failed) or
`groundhog attempt abort <wsid>` — either clears its heartbeat. Leave
it open only deliberately, when it holds real uncommitted progress the
user may want (resumable via `groundhog attempt resume <wsid>`), and
say so in the report. Verify with `groundhog attempt in-progress`.
Never resume or reap a workspace that `in-progress` shows as live — it
belongs to another process.

## Final report

End every session (normal stop or safe-stop) with:

- **Stop reason** — which rule fired (max N / no-improve-in-K / score
  target / anomaly / user stop), with anomaly details if any.
- **Score chain** — one row per iteration: attempt id, parent,
  improve|fresh, direction name, overall score, new-best marker.
- **Best** — before vs after (id + score), per `groundhog attempt best`.
- **Directions tried** — families opened or extended this session.
- **Gate failures** — which gates rejected what, and how you adapted.
- **Learnings added** — the entries appended to `learnings.md`, or
  "none".
- **Workspace state** — confirmation nothing dangling remains, or the
  deliberately-open wsid.

## Out of scope

Parallel subagents (`groundhog-orchestrate`), single curated attempts
(`groundhog-fresh`, `groundhog-improve`), and any workspace owned by
another live process.
