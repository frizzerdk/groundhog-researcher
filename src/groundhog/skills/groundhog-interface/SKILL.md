---
name: groundhog-interface
description: Base interface for driving a groundhog optimization run from an interactive session - orienting in the attempt store, opening/evaluating/committing workspaces via the groundhog CLI, running task tools, and the run's standing conventions (core_direction.md, gates, honesty, coexistence) plus the shared autonomy model. Use when working inside a groundhog run dir (a folder with task.py) as an attempt producer, or when a groundhog mode skill (groundhog-fresh, groundhog-improve, groundhog-iterate, groundhog-orchestrate) references it. Capabilities only - modes add the goal.
---

# Groundhog interface

An interactive session is a first-class attempt producer for a groundhog
optimization run: same store, same lifecycle, same gates as the automated
strategies. The interface is the `groundhog` CLI, run from inside the run
dir (the folder containing task.py — commands find it from cwd, searching
up to 5 parent dirs; there is no run-dir flag). This skill teaches
capabilities and invariants only; the mode skills (groundhog-fresh,
groundhog-improve, groundhog-iterate, groundhog-orchestrate) each add one
goal on top of it.

**Hands/framework split.** Hand-write files ONLY inside a workspace you
opened (use its `work/` subdir for scratch). Everything structural —
opening, seeding, naming, evaluating, committing, aborting, metadata —
comes from `groundhog` calls. Sole exception: appending to learnings.md
(see Learnings).

**Honesty — scores come ONLY from `groundhog eval` or `groundhog attempt
commit --eval`.** Never hand-write result.json or metadata.json (`commit
--eval` writes both itself), never guess, round, or restate a score the
framework didn't print.

## Orient first

Before touching anything, build the map:

```
groundhog attempt list             # committed: id, parent, status, score, name
groundhog attempt list --all       # include failed attempts
groundhog attempt best             # current best: id, score, name
groundhog attempt show <id>        # stages, metadata, note[score], files
groundhog attempt show <id> --file core_direction.md   # read one attempt file
groundhog attempt in-progress      # open workspaces: wsid, age, live/CRASHED
uv run task.py status              # trunks + direction families
```

- `task.py status` prints `Trunks:` (chains where each child improved on
  its parent) and `Direction families:` (attempts grouped by normalized
  core_direction.md content, best score per family) — the family map for
  deciding where to work.
- `learnings.md` at the run root is accumulated knowledge from prior
  attempts. Read it before proposing anything.
- The git-backed store caches scores as git notes:
  `git -C attempts log --all --oneline --show-notes=groundhog/score`.
  Notes are a mutable cache — the canonical score is always recomputed
  read-side by the commands above. Prefer the CLI.

## CLI reference

The complete surface for attempt work. No other flags exist — never
invent one.

```
groundhog attempt new [--parent ID] [--no-seed] [--name NAME]
    Open a workspace; prints wsid + path. Default parent = current best
    (none when the store is empty). Seeding copies ONLY the parent's
    solution.py and core_direction.md; --no-seed copies nothing but the
    parent pointer remains as lineage.
groundhog attempt commit <wsid> [--fail] [--eval] [--through STAGE]
    Finalize a workspace. --eval evaluates it, writes result.json +
    metadata, prints stage scores; success = evaluation completed and
    not --fail. If no name is set, the display name derives from
    core_direction.md's first line.
groundhog attempt abort <wsid>       Discard a workspace — no trace left.
groundhog attempt resume <wsid>      Take over an open workspace (see Coexistence).
groundhog attempt reap [--ttl S]     Abort CRASHED workspaces older than S (default 300).
groundhog attempt list [--all]       List attempts (--all includes failed).
groundhog attempt show <id> [--file F]
groundhog attempt in-progress
groundhog attempt best

groundhog eval <ws-dir|attempt-id|file.py> [--through STAGE] [--json]
    Score a target; persists NOTHING. Exit 0 = completed, 2 = failed.

groundhog tool list                                     This run's task tools.
groundhog tool run <name> [--attempt ID] [-p k=v ...]   Invoke one.
```

## Attempt lifecycle

1. `groundhog attempt new ...` — record the printed wsid and path.
   - Improving a prior: default (seeded); solution.py and
     core_direction.md arrive from the parent.
   - Fresh direction: add `--no-seed`, then create core_direction.md
     yourself (see Direction) before committing.
2. Work in the printed path. `solution.py` is the deliverable; keep
   notes, experiments, and scratch under `work/`.
3. Validate loop: `groundhog eval <ws-path>` after each meaningful
   change; fix and re-run until it completes (exit 0). eval only prints —
   nothing is recorded yet.
4. `groundhog attempt commit <wsid> --eval` — the evaluation of record.
   Default to `--eval` on every commit; a commit without it records no
   fresh result.
5. Real work that didn't pan out: `commit <wsid> --fail`. A failed
   attempt is recorded history — the family map remembers it and
   selection skips it. Reserve `abort` for scrap: empty or accidental
   workspaces.

## Evaluation details

- Targets: a workspace dir, a committed attempt id (re-evaluates its
  code), or a bare `.py` file.
- `--through STAGE` stops the stage pipeline early (cheap signal). The
  run may define its own default through-stage; commit and eval both use
  it unless you override.
- `--json` emits per-stage score, metrics, errors, warnings, and
  artifact names — use it when parsing.
- A failed stage means overall score -1.0, `FAILED at stage <name>`, and
  exit code 2.

## Task tools

`groundhog tool list` names this run's tools — the task's own tools
(defined by its `agent_tools` hook) plus any framework defaults. The set
is per-run: always list before assuming. Invoke with
`groundhog tool run <name> [--attempt ID] [-p k=v ...]`; `--attempt`
points workspace-relative tools at that attempt's files for the duration
of the call.

## Direction — core_direction.md

- **The FIRST LINE is the approach name itself** (e.g. `Rollout-greedy
  beam with 2-step lookahead`) — never a heading or label like
  "# Core Direction". That line becomes the attempt's display name and
  its folder slug.
- Below it: a few lines pinning the algorithmic backbone — narrow enough
  that two implementations of it belong to the same family.
- Fresh attempts MUST create it at the workspace root before commit.
- Children inherit it and MUST preserve it byte-for-byte. A fundamental
  change of approach is a new direction — open a fresh attempt; never
  rewrite an inherited direction.
- Family identity = normalized file content. Same content, same family.

## Gates — the law at every autonomy level

The automated strategies enforce these; session attempts get no
exemption. Gates are NOT checkpoints and are never relaxed by autonomy
level:

- **Direction present and unique (fresh).** A fresh attempt with no
  core_direction.md, or one whose direction duplicates an existing
  family (normalized comparison, failed attempts included), is a FAILED
  attempt. Check the family map before writing one.
- **No byte-identical children.** A child whose solution.py equals its
  parent's proves nothing — never commit it as an improvement.

## Coexistence with other producers

The store is shared: the optimizer, other sessions, and subagents may
hold open workspaces at the same time. Heartbeats (pid + timestamp) mark
live work.

- `attempt in-progress` labels each open workspace `live` or `CRASHED`.
- A wsid owned by another LIVE process resolves read-only — never
  commit, abort, or edit it.
- `attempt resume <wsid>` is explicit takeover: it rewrites the
  heartbeat so this process owns the workspace. Resume only DEAD
  (crashed) workspaces or your own — never a live foreign one.
- Edits survive crashes on disk: a crashed workspace still holds its
  uncommitted files, so resume-then-commit loses nothing.
- `attempt reap` aborts CRASHED workspaces past the TTL; it never
  touches live ones.

## Autonomy model (shared by all modes)

Three levels x six checkpoints. Levels are orthogonal to modes: each
mode declares a default, the invocation may override it
(`/groundhog-improve pair`), the user can switch mid-flight in plain
language ("go auto from here") and nudge single checkpoints ("always ask
before commits").

| Checkpoint             | pair  | checkpoint (default) | auto               |
|------------------------|-------|----------------------|--------------------|
| direction/prior chosen | pause | **pause**            | log                |
| approach pivot         | pause | narrate              | log                |
| each eval result       | pause | narrate              | log                |
| before commit          | pause | **pause**            | log                |
| between iterations     | pause | narrate              | log                |
| anomaly                | pause | **pause**            | safe-stop + report |

- **pause** — stop, present the state and the choice, wait for the user.
- **narrate** — say what you decided and why; continue without waiting.
- **log** — record for the final report; say nothing at the time.
- **auto never blocks.** On anomaly at auto, safe-stop: finish or
  cleanly commit/abort the current workspace (no live heartbeat left
  dangling), then deliver the report. Never sit waiting for input.
- Anomaly = anything outside the plan's envelope: repeated eval crashes,
  unexplained gate failures, store conflicts, suspicious scores.
- Gates are not checkpoints — they apply identically at every level.

## Learnings

`learnings.md` at the run root. Entries are separated by a `---` line
with a blank line on each side — match that separator exactly when
appending. After committed work that taught something durable (a
technique that moved the score, a dead end with its reason), append ONE
short entry. Never rewrite or delete existing entries. Subagents inside
attempts never write it — they report up to their orchestrator.

## Out of scope

- No goal lives here — the mode or the user supplies it. Orienting is
  not a license to start optimizing.
- Don't launch the automated optimizer (`uv run task.py N`) unless
  explicitly asked.
- Don't edit task.py, the internals of `attempts/`, or files of attempts
  you don't own.
- Subagents working inside a single attempt follow the constrained
  groundhog-in-attempt skill instead of this one.
