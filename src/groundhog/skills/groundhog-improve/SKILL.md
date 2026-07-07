---
name: groundhog-improve
description: Produces one child attempt that improves a committed prior in the groundhog run in the current directory — the session does the editing, the groundhog CLI owns lifecycle, scoring, and gate enforcement at commit. Use when the user asks to improve, refine, or build on an existing attempt or the current best (e.g. "/groundhog-improve", "/groundhog-improve e65665e2", "make the best attempt better"). Inherits the parent's direction and preserves it; default autonomy level checkpoint.
argument-hint: "[prior-id] [pair|checkpoint|auto]"
---

# Groundhog improve

One goal: open a child workspace on a prior attempt, make it measurably
better, commit it with a real eval. Every capability — CLI commands,
orientation, the autonomy model, honesty rules, coexistence — is
defined in the `groundhog-interface` skill; load it first if it is not
already in context. This file adds only the improve workflow and its
defaults.

## Autonomy level

**Default: checkpoint.** Pause at exactly two planned points — prior
chosen and before commit — plus any anomaly; narrate everything else
and keep working. The levels (pair / checkpoint / auto) and the six
checkpoints are defined once in groundhog-interface. The user sets a
level at invocation ("/groundhog-improve pair") and can switch or
nudge mid-flight in plain language ("go auto from here", "always ask
before commits"). **Gates are not checkpoints** — they run identically
at every level.

## Workflow

1. **Orient.** `groundhog attempt list`, `groundhog attempt best`, read
   `learnings.md`, and `groundhog attempt show <id>` on candidate
   priors. Know where the candidate sits in the family map before
   committing to it as a base.
2. **Open the child workspace on the prior.**
   - The user named a prior (in the invocation or in chat):
     `groundhog attempt new --parent <ID>`
   - Otherwise accept the toolkit's selection: run
     `groundhog attempt new` with no `--parent`. The toolkit picks the
     parent (currently the best-scoring attempt) and the command echoes
     which one it chose.
   - CHECKPOINT (prior chosen): state the parent's id, score, and
     direction (the first line of its `core_direction.md`), and why it
     is the right base. If the user redirects, run
     `groundhog attempt abort <wsid>` and reopen with `--parent <ID>`.
3. **Study before editing.** The workspace arrives seeded from the
   parent — code plus `core_direction.md`. Read the direction and the
   parent's result (`groundhog attempt show <parent-id>`) so the
   improvement targets a real, observed weakness, not a guess.
4. **Improve and measure.** Edit only inside the workspace (`work/`
   for scratch). Score with `groundhog eval <ws-dir>` — add
   `--through STAGE` for a partial pass, `--json` for parseable
   output. Loop edit → eval until the child beats the parent or you
   have a clear stop reason. Each eval result is a narrate-only
   checkpoint at the default level.
5. **Commit.** CHECKPOINT (before commit): run
   `groundhog tool run check-gates --attempt <wsid>`, report its verdict
   plus the score delta vs the parent and what changed, then
   `groundhog attempt commit <wsid> --eval --strategy session`. A dead
   end still worth recording as history: add `--fail`. Not worth
   recording at all: `groundhog attempt abort <wsid>`.

## Direction obligations

- **Preserve the inherited direction.** `core_direction.md` is seeded
  from the parent. Its FIRST LINE is the approach name itself (no
  heading, no label) and doubles as the attempt's display name and
  folder slug — keep that line intact. Refine the body only while the
  work remains the same approach. The commit gate enforces exactly this:
  a changed first line is restored to the parent's and flagged
  (`direction_restored`), while body refinements are kept and recorded
  (`direction_body_refined`).
- **A pivot is not an improve.** If the changes amount to a new
  approach, that is the approach-pivot checkpoint: pause and surface
  it (pair/checkpoint) or safe-stop and report (auto). New directions
  belong to groundhog-fresh, never to a rewritten inheritance.
- **No empty children.** A byte-identical child proves nothing — the
  commit flags it non-promotable and selection skips it. check-gates
  shows the flag before you commit: diff against the parent and commit
  only after a real change.

## Non-negotiables (hold even if groundhog-interface is skimmed)

- Scores come ONLY from `groundhog eval` or `commit --eval`. Never
  hand-write `result.json` or report an unmeasured number.
- Hand-write files only inside this attempt's own workspace;
  everything structural goes through groundhog commands.
