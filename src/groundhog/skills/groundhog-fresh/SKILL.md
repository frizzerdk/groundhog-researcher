---
name: groundhog-fresh
description: Produces one fresh groundhog attempt in an unexplored direction — proposes 2-3 candidate directions from the family map and learnings, builds the chosen one, evaluates, and commits it through the groundhog CLI. Use when the user invokes /groundhog-fresh or asks to try a new approach, start a new family, or explore something no existing attempt covers. Builds on groundhog-interface for all mechanics. Default autonomy level: checkpoint; the core direction must be unique across existing families.
argument-hint: [pair|checkpoint|auto]
---

# Groundhog fresh

You are producing ONE fresh attempt — the seed of a new family — for the
groundhog run in the current directory. This skill adds only the goal and
workflow; every mechanic (attempt lifecycle, evaluation, task tools,
conventions, honesty rules, coexistence, the autonomy model) is defined
in the **groundhog-interface** skill. Read it first. If it is
unavailable, recover mechanics from `groundhog --help` and
`groundhog attempt --help`.

## Goal

One committed attempt whose core direction is new to this run — a
genuine alternative to every existing family, not a refinement of one. A
committed failure in a truly new direction is a success for this mode; a
good score in a direction the run already has is not.

## Autonomy level

**Default: checkpoint** — pause at direction-chosen, before-commit, and
anomaly; narrate the rest and work alone in between. The invocation may
name another level (`/groundhog-fresh pair`, `/groundhog-fresh auto`),
and the user can switch levels or nudge single checkpoints mid-flight in
plain language. Levels and checkpoints are defined once in
groundhog-interface. Gates are not checkpoints — they apply identically
at every level.

## Workflow

1. **Orient.** `uv run task.py status` for the family/trunk map;
   `groundhog attempt list` and `learnings.md` for what was tried and
   what was learned. You are mapping the explored space to find holes.
2. **Propose 2-3 candidate directions**, each 2-3 sentences: the
   approach, why it could score well, and why no existing family covers
   it. Direction-chosen checkpoint: present the candidates, the user
   picks. At auto level, pick the strongest yourself and log why.
3. **Open a workspace:** `groundhog attempt new --no-seed`. The parent
   defaults to the current best; `--no-seed` keeps its files — including
   its `core_direction.md` — out of your workspace.
4. **Write `core_direction.md` first.** First line = the approach name
   itself (no "Core Direction" heading or label); it becomes the display
   name and folder slug. Then build `solution.py`; scratch goes in
   `work/`.
5. **Evaluate as you iterate:** `groundhog eval <ws-dir>`. Scores come
   only from eval — never hand-write results.
6. **Before-commit checkpoint:** report the direction, the score, and
   what you tried; recommend commit — a recorded failed family is
   information; abort (`groundhog attempt abort <wsid>`) only for
   unusable scraps. Then `groundhog attempt commit <wsid> --eval`.

## Uniqueness obligation

**The direction must be unique vs every existing family** (family
identity = the normalized file content) and `core_direction.md` must
exist. The automated strategies enforce this in code; **on the session
path YOU are the gate** — before committing, re-check the family map
(`groundhog attempt list` + read the leaders' directions) and confirm:
direction file present, first-line convention followed, genuinely new
family, solution not byte-identical to any attempt. If your work
drifted into an existing family mid-build, that is an approach-pivot
checkpoint — surface it and re-aim; never reword the direction
cosmetically to make it look distinct.

## Out of scope

Improving an existing attempt (`groundhog-improve`), multi-attempt loops
(`groundhog-iterate`), parallel subagents (`groundhog-orchestrate`).
