# Fleet Conventions

Operating conventions for running a *fleet* of agents against a single
groundhog store — many parallel workers, long runs, and sometimes more
than one orchestrator at once. A K=3 fan-out (see the
`groundhog-orchestrate` skill) rarely trips any of these; a large
campaign trips all of them.

Every rule below earned its place in one real campaign. The failure
mode each prevents is stated as a one-line incident, because the rule
only makes sense once you've seen it break. The short form of these
rules also lives in the `groundhog-orchestrate` skill under "Fleet
conventions"; this file is the rationale.

## The invariant behind all of it

A groundhog store is an immutable attempt tree with a *single* lifecycle
surface: `groundhog attempt new` opens a workspace, `commit` finalizes
it, `abort` discards it. That surface is not built for concurrent
writers — the folder backend records no heartbeats, and even the git
backend's coordination is optimistic. So the whole fleet design is one
idea: **fan work out widely, but funnel every mutation of the store
through a single throat.** The conventions are the guardrails that keep
that funnel intact under scale.

## 1. Single-writer lifecycle

**Rule.** Exactly one process runs `attempt new` / `commit` / `abort`.
Parallel workers never call the lifecycle at all — they produce results
into their own scratch dirs and hand them off. One collector folds
those results into the store.

**Incident.** With two orchestrators both committing against the same
store, attempts went missing — a workspace opened by one and committed
by the other, races on the parent pointer, results that were produced
but never appeared in `attempt list`. The count of committed attempts
did not match the count of finished workers, and there was no single
place to look for why.

**Why it works.** A single writer serializes every mutation. Workers
become pure producers — they can crash, retry, or run on another machine
without endangering the store, because they never touch it. The store's
consistency is the collector's sole responsibility, and there is exactly
one collector.

## 2. Manifest → collector → reconcile

**Rule.** Each worker writes a manifest describing what it produced —
minimally `{solves, metrics}`: which candidates it believes it solved
and the metrics it measured. The collector folds every manifest into the
store, then **reconciles**: it compares *realized* (what actually landed
as committed attempts) against *claimed* (what the manifests say should
be there) and investigates every mismatch. A discrepancy is a bug to
chase, never a number to paper over.

**Incident.** This reconcile step caught *every* silent-loss bug in the
campaign. Without it, a worker would report success, its manifest would
say five solves, and only four would be in the store — and nothing
surfaced the gap, because each stage locally looked fine. The reconcile
turned an invisible off-by-one into a loud, located failure.

**Why it works.** Realized-vs-claimed is a cross-check between two
independently derived counts. Any single-point failure — a dropped
commit, a worker that lied, a fold that skipped a manifest — shows up as
a mismatch. The fleet's trustworthiness comes from this comparison, not
from any one component being correct.

**Shape of a manifest.** Keep it flat and machine-foldable: the worker's
id, the workspace(s) it produced, the candidates it claims to have
solved, and the raw metrics it measured (never a score — scores are
computed read-side, per the store's honesty rule). The collector reads
manifests, commits with `--eval` so the store's own evaluation is the
number of record, and then reconciles its committed set against the
union of manifest claims.

## 3. Foreground-only for agent work

**Rule.** Run agent work in the foreground. Do not park an agent run in
the background and wait on it. If a run is too long for one foreground
segment, chunk it — break it into segments that each complete in the
foreground.

**Incident.** Agents launched into the background and waited on stalled:
the background wait blocked without progressing the agent, and the fleet
sat idle behind a job that was never going to advance on its own.

**Why it works.** Agent runs are interactive processes that need their
turns driven; a background wait is not an event loop for them. Chunking
keeps every segment observable and interruptible, and keeps the
orchestrator in control of pacing instead of blocked behind an opaque
wait.

## 4. Never `cd` into a workspace

**Rule.** Never change a shell's working directory into a workspace.
Run every `groundhog` command from the run dir and pass the workspace by
path. A shell left parked with its CWD inside a workspace must be moved
out before that workspace is committed.

**Incident.** On Windows, a parked shell CWD inside a workspace dir
blocked the folder backend from committing that workspace: the commit
tries to move/remove the directory and Windows refuses because the
directory is *in use* by the parked shell — `WinError 5` (access
denied). The workspace could not be finalized until the shell moved out.

**Why it works.** Windows locks a directory that is any process's
current directory; the folder backend's commit relocates the workspace
dir, which fails while it is locked. Keeping every shell rooted at the
run dir and addressing workspaces purely by path means nothing ever
holds a lock on the thing being committed. (This is a Windows-specific
lock, but the discipline is cheap and portable, so it is universal.)

## 5. Fail-loud chaining

**Rule.** In any multi-step pipeline, verify each step's output — it
exists, and it is fresh — before feeding the next step. A step that
fails must halt the chain loudly, never let the next step run on stale
inputs.

**Incident.** A masked step failure let a stale artifact get submitted
twice: one stage failed silently, left last run's artifact in place, and
the next stage happily consumed it as if it were new — so the same old
result was submitted a second time under a new identity. Nothing
errored; the wrong thing simply propagated.

**Why it works.** Pipelines fail at the seams. Checking freshness at
each seam converts a silent stale-data propagation into an immediate,
located stop. The cost is one existence/timestamp check per step; the
payoff is that "it ran without error" actually means "it produced fresh
output".

## 6. Resume over respawn

**Rule.** When an agent stalls, prefer resuming it with its context
(`groundhog attempt resume <wsid>`, which takes over the open workspace)
over killing it and spawning a fresh one. Respawn only when there is
genuinely nothing to resume into.

**Incident.** A stalled agent resumed with its accumulated context
consistently outperformed a fresh agent started from scratch on the same
angle — the fresh one had to rediscover the workspace state, the
approach, and the dead ends the stalled one had already mapped.

**Why it works.** An agent's value is mostly in the context it has built
— what it has read, tried, and ruled out. A crash or stall does not
destroy the workspace files (edits survive on disk), so resuming
reclaims all of that work. Respawning throws it away and pays the
rediscovery cost again.

## 7. Concurrent orchestrators coordinate via a claims file

**Rule.** When more than one orchestrator runs against the store at
once, they coordinate through a shared **claims file** that records who
owns which directories and workspaces. Each orchestrator keeps
**one open workspace at a time**. You touch only what you have claimed.

**Incident.** Uncoordinated orchestrators stepped on each other — two
opened workspaces in overlapping territory, and ownership of a given
workspace was ambiguous, which fed straight back into the missing-attempt
failure of rule 1.

**Why it works.** The claims file is an explicit, inspectable ownership
map — the coordination that the store's own lifecycle does not provide
across processes. One-open-workspace-at-a-time bounds each
orchestrator's blast radius: a crash strands at most one workspace, and
there is never a question of who owns what. It is the multi-writer
extension of the single-writer rule: still one writer *per claimed
region*.

## Namespacing — unique module stems for generated code

**Rule.** Parallel workers that generate Python code MUST give their
modules unique stems — prefix by wsid, angle, or worker id. Two workers
must never write same-named modules.

**Incident.** Two workers each wrote a module with the same stem (say
`solution.py` or `model.py`). When both were imported in one process,
Python's `sys.modules` cache returned the first worker's already-imported
module for the second import — silently shadowing the second worker's
file. The second worker evaluated the *first* worker's code, with no
error anywhere: same name, cache hit, wrong bytes.

**Why it works.** `sys.modules` is keyed by module name, not by file
path. Same name = same cache slot = the first one loaded wins, forever,
in that process. Unique stems give each worker its own cache slot, so an
import always resolves to that worker's own file. This bites hardest when
a single collector process imports many workers' outputs to evaluate
them — exactly the fold in rule 2.

## The through-line

Rules 1, 2, and 7 are one idea at three scales: **one writer** (1), **a
cross-check that the one writer didn't drop anything** (2), and **how
several writers partition the store without colliding** (7). Rules 3–6
and namespacing are the operational hygiene that keeps individual workers
honest and observable so the funnel above them stays clean. Together they
turn a fleet from a source of silent loss into something whose every
result you can account for.
