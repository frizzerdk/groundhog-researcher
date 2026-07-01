# Git Attempt History — Goal & Flow (DRAFT v0 — edit me)

> Draft for Frederik to edit. The principle, in his words: *"the storage should
> basically be a wrapper for normal git operations, similar to the old one — make
> a new branch on the parent when we start, then when we're done commit it and
> sync. It should behave like a person would do it."*

## Goal

- The optimizer's **memory**, stored as an **ordinary git repo**. The backend is a
  thin wrapper around the git commands a person would run by hand: **branch → work
  → commit → push**.
- **Primary user: the optimizer loop** (storage/memory for strategies). Secondary:
  you, inspecting occasionally with normal git tools.

## Model

- The attempts repo is a **normal git repo**.
- **One attempt = one branch off its parent, with one commit** (the finished work).
- **Lineage = the commit graph** — you branched from the parent, so the parent
  commit *is* the logical parent. `git log --all --graph` draws the tree.
- **Identity = the commit sha**; the **branch name** is the human-readable label.

## Flow of one attempt — what a person would do

1. **START** — the optimizer picks a strategy and a parent attempt.
   - `git checkout -b <branch> <parent-commit>`   (fresh attempt → branch from `main`)
   - The working tree now holds the parent's files to build on.

2. **WORK** — the strategy/agent edits `solution.py`, runs the task's eval (writes
   `result.json`), records the core direction + name + metadata.
   - An agent may make intermediate commits along the way; they get **squashed**
     into one at the end (one attempt = one published commit).

3. **DONE** — finish the piece of work:
   - `git add -A && git commit -m "<name>"`   (the single attempt commit)
   - `git push <origin> <branch>`             (best-effort sync; degrade to local)
   - The branch now points at the attempt's commit; its parent is the branch point.
   - **ABORT instead**: discard the branch + working changes — nothing is published.

4. The optimizer recomputes **derived views** (trunks / families / best) by reading
   the commit graph + metadata. Nothing extra is stored.

5. Repeat for the next attempt.

## On disc

```
<task>/
  attempts/                 ← the git repo
    .git/                   ← every attempt branch; git log --all --graph = the tree
    solution.py  result.json  metadata.json   ← the working tree = the attempt
                                                 currently checked out
  learnings.md  queue.json  ← optimizer state (siblings of the repo, not inside it)
```

One working tree (the optimizer is sequential — one attempt at a time). To look at
another attempt you check out its branch, like a person would.

## Inspect (you, occasionally, with normal git)

```
git branch                 → every attempt
git log --all --graph      → the lineage tree
git checkout <branch>      → look at a specific attempt's files
git switch -               → back
```

## Sync

Each machine pushes its attempt branches. Branch names carry an **origin tag** so two
machines never collide (`<origin>/<name>`), and commits are immutable — so sync is
conflict-free by construction. Pull to see another machine's attempts.

## Open points (please decide / edit)

- **Branch naming.** The direction/name often isn't known until *during* the work
  (a fresh attempt mints its direction as it goes). Options: start the branch with a
  provisional name (`wip/<id>`) and **rename** at commit (`git branch -m`, a normal
  person move) → `<slug>-<short>`; or just name branches `attempt/<n>`. Recommend:
  rename-at-commit to a readable `<slug>-<short>`.
- **What is `main`?** The base the fresh attempts branch from (an initial commit, maybe
  holding the task context). Do we **promote the best to `main`** like the old version,
  or leave `main` as the base and keep "best" purely derived? Recommend: `main` = base;
  best stays derived; optionally a moving `best` branch/tag for convenience.
- **Parallelism.** One working tree handles the sequential loop. If we ever run
  attempts in parallel, that's where worktrees come back (one per concurrent attempt).
  Recommend: sequential for now; revisit only if needed.
- **The agent `.git` gate (deferred).** When real agents run, they work in the repo's
  working tree and see its `.git`. Parked for now; fix when we wire agents.
- **Failed attempts.** Still committed (a branch + commit, marked failed in metadata),
  or left as an un-pushed branch? Recommend: commit + mark `status: fail` so failures
  are kept and analyzable.
