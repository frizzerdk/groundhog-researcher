# Git Attempt History — Implementation Plan (v2)

> Status: **decisions locked, ready to build.** v2 incorporates a second review (2026-06-18) and nine settled decisions. One gate remains: the parked `.git`-siting problem must be solved before real agent runs (it does **not** block the storage-layer MVP or the test suite). Phases are sequenced so the suite stays green at the end of each.

> **Implementation status (2026-06-21): P0, P1, P3 landed and green — 164 tests pass.**
> The storage layer is complete: the contract is re-keyed to string `id`; `histories/git.py` implements the git backend (commit-via-plumbing, no-checkout reads, chronological `list()`); a parametrized contract suite (`tests/test_history_contract.py`) runs the same assertions against folder and git; sync (`tests/test_git_sync.py`) is built storage-only over per-origin create-only refs. **Only P2 remains** — threading attempt *names* from the strategies (producer-side) and persisting `name` in the folder backend. P2 reaches into the real-agent strategies, which sit behind the parked `.git`-siting gate, so it is held for a working session. Git already stores/parses the `Attempt-Name`/`Status`/`Groundhog-Created` trailers, so P2's storage half is done. Decision made during P1: `created_at` is a real-time float carried in a `Groundhog-Created` trailer (monotonic-guarded), not git's 1-second `%ct`, so `list()[-1]` is reliably the newest.

> **Update (2026-06-22): P2 storage half landed — 168 tests pass.** A dedicated `metadata.json` per attempt now holds the human-readable `name` (and attempt metadata); `result.json` is eval-only; `Attempt.name`/`.metadata` read `metadata.json` backend-agnostically with a legacy fallback to `result.json` metadata. `name` now persists for the folder backend (gap closed) and round-trips for both. Folder dir names stay structural (`id`/`parent`). **Remaining P2:** the *producer* side — strategies setting `ws.name` (agent attempts from the `plan_approaches` slug; organic attempts auto-slugged) — plus the parked **`.git`-siting gate**, which must be resolved before real agent runs. Both touch the gated agent strategies.

> **Update (2026-06-22, later): P2 complete — 173 tests pass. All four phases (P0–P3) implemented and green.** Producer wiring landed: organic strategies (Improve / Fresh / CrossPollinate) set `ws.name` to a slug of the core-direction title; agent strategies set it from the planned `plan_approaches` name (now threaded through the queued config) or fall back to the direction slug, via `AgentStrategy._finalize`. Naming = direction-title slug (`slugify` + `workspace_name` in `utils/direction.py`); the `plan_approaches` LLM slug takes precedence. Verified end-to-end: direction → slug → `ws.name` → `metadata.json` → `attempt.name`, and the git commit subject. **Only open item: the parked `.git`-siting gate** — re-investigated 2026-06-22 and found smaller than parked: the storage layer never runs agents, agent workspaces under git are identical to the folder backend's attempt dirs today, and the one leaky backend (codex on Windows) is bounded by Windows ACLs, **not** `.git` (so a `.git` sentinel can't fix it; WSL2/Docker would). Deliberately deferred by the user.

> ## v3 redesign (2026-06-22): a real, browsable git repo with worktree folders
>
> After a design session (`.review-sessions/2026-06-22-git-ah-disk-layout`), the on-disk model changes from a **bare object store** to a **normal git repo** the user views and manipulates with regular git tools. The git-native core (below) is already built; the rework adds the browsable layer. **Locked decisions:**
>
> - **`<task>/attempts/` is a normal (non-bare) git repo** — programmatically managed but ordinary: `git log` / `git show` / GUIs / push-pull all work, you `cd` in and use git.
> - **Git-native lineage (already implemented):** each attempt = one commit, **commit-parent = the logical parent**, **id = the commit sha**. `git log --graph` draws the lineage. `id`/`parent` are *also* mirrored into `metadata.json` for durability + readability (survives history rewrites; lineage stays the metadata's job for `derive_trunks`).
> - **One attempt = one published commit.** Multi-step builds (agent phases / recovery checkpoints) are **squashed** to a single commit before publishing.
> - **Browsable folders = git worktrees.** Each materialized attempt is a worktree checked out to its commit: `attempts/done/<id>-<name>/`, `attempts/fail/<id>-<name>/`. Default = materialize **all** attempts; "best+recent on disc" later = keep worktrees for those and `git worktree remove` the rest (commits stay in history). This is the one mechanism that gives git-lineage **and** browsable folders at once.
> - **metadata.json** per attempt: id (=sha), parent, name, status, strategy/cost; `result.json` stays eval-only.
> - **Derived** trunks/families/best read-side (git + metadata); **sync** keeps the per-origin create-only refs (already built), conflict-free.
> - **in-progress + the agent `.git` gate are DEFERRED.** Build storage first, wire agents after. The eventual gate fix: give the agents that walk up to `.git` (opencode) a throwaway `.git` per workspace so they fence to it instead of re-anchoring to `attempts/.git`. claude_code/gemini are already fenced by their tool systems; codex-on-Windows is an OS-ACL limit (WSL/Docker), unaffected by this layout.
> - **Deferred shelf:** agent gate / in-progress siting · meta-repo decision log · index cache · best+recent pruning · display numbers.
>
> **Carries over from P0–P3 (built, 173 green):** commit DAG mechanics (parent=lineage, sha=id), the `_git` chokepoint (config-neutral reproducible hashes), reads via `git show`, `metadata.json`/name/status plumbing, create-only sync refs, the ABC contract, derived views, the contract suite. **Reworked:** bare → normal repo, worktree-materialized browsable folders, default folder structure. **Implementation nuance to nail in the build:** what the repo's *primary* worktree is checked out to — proposal: a small orphan branch carrying a `.gitignore` for `done/`/`fail/`/`in-progress/`, attempt commits reachable via refs, the folders as detached worktrees.
>
> **Implemented (2026-06-22): v3 worktree backend built and green — 176 tests.** `histories/git.py` reworked from bare store to a normal repo at `<base>/attempts/`: `git init` + orphan `_root` branch holding a `.gitignore`; each attempt committed (parent = lineage, sha = id) and materialized as a `done/<sha8>-<name>/` (or `fail/`) **worktree**; reads still via `git show`; sync refs unchanged; in-progress workspaces under `in-progress/<uuid>/`. New `tests/test_git_worktrees.py` asserts browsability (real repo, worktree folders, `git log --graph` = lineage, pruning keeps history). The model was first validated by a spike (`groundhog-runs/spike_worktree.sh`) and confirmed with an end-to-end smoke. **Deferred:** in-progress agent `.git` gate (own-`.git`-per-workspace), meta-repo decision log, index cache, best+recent pruning UX, display numbers.

## Context

Adding a **git-backed `AttemptHistory` backend** behind the existing ABC (`base/attempt_history.py`). An "attempt" = one candidate + its eval result; the optimizer builds a tree of them. The current backend is folder-based (`histories/folder.py`). A predecessor (`EvaluatableExperiments`) built git tracking and abandoned it because the *implementation* tangled (rebased trunks, stored branches/worktrees, index-as-truth, a `.git` in the agent's cwd) — not because git was wrong. This is the purified version.

### Settled design

- Git is a **storage substrate**: attempt = **one immutable commit**; git-parent **is** the logical parent (DAG-native); reads come from the object store with **no checkout**.
- **Identity = the commit hash** (`id: str`), assigned **at commit**. The integer `N` is dropped.
- **`created_at: float`** (unix epoch) — native to both backends (git `%ct`, folder `st_mtime`); it's the **sort key** and the ABC annotation.
- **Name** = a human-readable label (e.g. `improve-relu-init`), strategy-generated, mutable until commit, frozen into an `Attempt-Name:` trailer. **Display-only — never identity/lookup.** Lookups resolve by id or a unique short-hash prefix.
- **Status** = an `Attempt.status` ABC property; git sources it from a `Status: done|fail` trailer, folder from its existing `_done`/`_fail` suffix. **Fail vs abort**: a failed attempt (eval ran, result bad) **is** committed (`Status: fail`, kept); an aborted attempt (discarded mid-flight) is **not**.
- **Workspace** = a **plain dir** (no `.git`); `commit()` snapshots the tree into one commit via external git plumbing (`--git-dir`/`--work-tree` + a per-attempt `GIT_INDEX_FILE`); `abort()` = `rmtree` (+ `os.replace → .trash` on a Windows lock). Git is a **sink**, not involved in-flight.
- **Local + synced = one class + a `remote` flag** (hybrid cadence): per-origin **create-only** refs `refs/attempts/<origin>/<id>`; push best-effort after commit; fetch best-effort before global reads; degrade to local on failure. Create-only ⇒ conflict-free.
- **Commit everything**: at store init, **neutralize global/system excludes** (`core.excludesFile` empty, no `.git/info/exclude`); the per-attempt `.gitignore` (empty by default) is the only deliberate trim knob. Use `git add -A` (not `-f`).
- Trunks/families/best/lineage are **derived** read-side views, never stored, never rebased.

### Decisions ledger

| Decision | Choice |
|---|---|
| Test runner | **pytest** — CI switches to `uv run pytest`; fixtures + parametrization |
| Committed-attempt reads | **Attempt.read_file()/list_files()** — no virtual `GitTreePath` |
| In-flight workspace id | **`display_id` / `name`** separate from the commit id |
| `list()` ordering | **chronological by `created_at`** from P1 (deterministic tie-break) |
| Name policy | **display-only**; resolve by id / short-hash |
| Commit content | **neutralize global excludes**; local `.gitignore` is the trim knob |
| `created_at` type | **float epoch** |
| Organic-attempt name | **auto-slug** from direction/solution summary |
| origin-id | **`uuid4().hex[:12]`** persisted in the store (`local` offline) |
| `.trash` reaping | **best-effort at next `workspace()`/init** |
| `fetch_ttl_s` | **2–5 s** |
| folder `status` source | **reuse the `_done`/`_fail` suffix**, exposed via the ABC property |

### Out of scope (deferred shelf)

In-flight checkpoint commits ("open" state) · ULID start-locked ids · materialized on-disk views · meta-repo · rebuildable perf cache.

---

## Phase 0 — ABC migration + folder.py conforms + consumer sweep + pytest

**Goal:** re-key the contract from integer `N` to string `id`, switch CI to pytest, and refactor committed-attempt reads onto the Attempt API — **without** introducing git yet. Folder backend stays green.

**Steps:**
- `base/attempt_history.py`: `Attempt.number:int → id:str`; `parent → Optional[str]`; add `created_at: float`, `name: str`, and a `status` property. `Workspace` **drops** `number`, gains `display_id`/`name` (in-flight label), `parent → str`. `workspace(parent:Optional[str])`, `get(id:str)`. `derive_trunks` re-keys by `id` (101,122,125). `derive_families` (158-159) orders by `created_at` **and** reads direction via the Attempt API (below), not `a.path`.
- **Reader refactor (kills the GitTreePath risk):** add `read_direction_from_attempt(attempt)` to `utils/direction.py` that tries each candidate path via `attempt.read_file(rel)` and returns the first hit. Repoint `derive_families` (`attempt_history.py:153`) and `selection._attempt_family_key` (`selection.py:124`) at it. No consumer does `Path(attempt.path)` on a committed attempt anymore.
- `histories/folder.py`: `id` = dir-name string (e.g. `"001"`); `parent` str; `created_at` = `st_mtime` (float); `name` persisted/read back; `status` property from the `_done`/`_fail` suffix. **Keep the integer allocator** (dir naming only). De-`int()` `list`/`get`/`lineage`.
- Consumer sweep `.number → .id`. **Real edits:** `tools.py:356-377` resolver keys by `id`, accepts id / `attempt_<id>` / short-hash (**not** name), drops `int()`; `tools.py:437-438` → `a.status`; `agent.py:522` `get(forced)`; `agent.py:534` `parent=prior.id`; `improve.py:98`, `cross_pollinate.py:131` parent; `agent.py:425/641/644` `ws.number` → `ws.display_id`/`ws.name`. Fix `:03d` format specs that break on a string id.
- **pytest:** change `.github/workflows/test.yml:27-29` to `uv run pytest tests/` (keep the per-file `__main__` blocks — harmless). Verify all existing script-style tests still **collect and pass** under pytest (they create their own tmpdirs; capsys-needing tests now actually run). Add `tests/conftest.py` scaffolding + a `pytest` config in `pyproject.toml`.

**Exit:** `uv run pytest tests/` green with folder as the only backend; no `.number` on any Attempt/Workspace; no committed-attempt `.path` filesystem use in `direction`/`selection`/`derive_families`; CI runs pytest.

## Phase 1 — GitAttemptHistory local core + parametrized contract suite

**Goal:** stand up `histories/git.py` (local-only) and prove it satisfies the same contract as folder via a parametrized fixture. Commits via plumbing; reads from the object store, no checkout; **`list()` chronological from day one**.

**Steps:**
- New `histories/git.py` (`GitAttempt`/`GitWorkspace`/`GitAttemptHistory`); one `_git()` subprocess chokepoint injecting `GIT_AUTHOR_*`/`GIT_COMMITTER_*` (the box's git config is unusable).
- `__init__`: bare store `base_path/.attempts.git` (`git init --bare`); **`core.autocrlf false`** (CRLF rewriting changes commit hashes = identity); **neutralize global excludes** (`core.excludesFile=` empty, no `.git/info/exclude`); persist `origin` = `uuid4().hex[:12]` (`local` offline); ensure `.work`/`.trash`; **lazy-sweep `.trash`** here.
- `workspace(parent)`: plain dir `base_path/.work/<uuid>` (no `.git`); seed from parent via `git archive <parent> | tar -x`; per-attempt `GIT_INDEX_FILE`; `display_id` = the temp dir id, settable `.name`.
- `commit(success)`: `add -A` (per-attempt index, no `-f`) → `write-tree` → `commit-tree -p <parent_sha> <msg>` → **create-only** `update-ref refs/attempts/<origin>/<sha>`. (Fail still commits; `success` only sets the future Status trailer.)
- `abort()`: `rmtree` (+ `os.replace → .trash/<uuid>` on lock); never publish a ref.
- Read API (by sha, no checkout): `code`/`result`/`metadata`/`read_file`/`list_files` via `git show`/`ls-tree` (`check=False` + degrade on missing); `created_at` = `%ct` (float); `parent` = `rev-parse <sha>^1`. **No GitTreePath** — `read_direction_from_attempt` already uses `read_file`.
- `list(only_done)`: `for-each-ref` → build attempts → **sort by `created_at`, tie-break by id** (deterministic; `simple.py:460` relies on `list()[-1]` = newest). `get`/`best`/`lineage` over refs; `best`/`lineage` copied from folder.
- Export from `__init__.py`. `tests/conftest.py`: `history_factory` fixture params `[folder, git]` (skip git if not on PATH); `commit_attempt(...)` helper. Parametrize the pure-contract tests; keep folder-implementation-detail tests folder-pinned; soften dense-1..20 → uniqueness; **abort test = list-count/path-disappearance, never `ws.id`** (no id in-flight).

**Exit:** parametrized contract tests pass for **both** `[folder]` and `[git]`; `list()` is chronological for both; a git attempt round-trips with `id` = the commit sha, reads with no checkout.

## Phase 2 — Name + Status trailers (end to end) + folder parity

**Goal:** name and status first-class through the full seam; **fail-vs-abort observable** via `list(only_done)`.

**Steps:**
- `git.py commit()`: emit `Attempt-Name`, `Status: done|fail`, `Parent` trailers; parse them from one `%B` read; `list(only_done=True)` filters `status != done`.
- `folder.py`: expose `name` + `status` (status from the `_done`/`_fail` suffix — no new file).
- **Name plumbing:** `plan_approaches.py:134` threads the LLM-minted `proposal['name']` slug into the queued config; `AgentConfig.name`; set-points at `agent._prepare_workspace`, `fresh.py:184`, `fresh_agent._ensure_direction`. **Organic attempts auto-slug** from the direction/solution summary already present at the seam. Display-only label; lookups never use it.
- Tests NT1/NT2/NT3/NT5: name round-trips as a trailer and `history.get(a.name)` is **not** a lookup (resolves by id/short-hash only); status filtering; fail-vs-abort dichotomy; `created_at` ordering (git ids are lexically unordered, so ordering must use `created_at`).

**Exit:** NT1/NT2/NT3/NT5 pass both backends; failures retrievable under `only_done=False`, aborts leave no trace; `Attempt-Name` trailer readable via `git log`.

## Phase 3 — Sync layer (one class, sync flag, hybrid cadence) — **storage-only**

**Goal:** per-origin create-only-ref sync as a `remote` field + thin `SyncPolicy`. Conflict-free by construction. **Built and tested across two stores, but NOT wired into the live optimizer** until the `.git`-siting gate is solved.

**Steps:**
- `__init__`: `remote:Optional[str]=None` (None ⇒ local); `SyncPolicy(push_after_commit, fetch_before_reads, timeouts, retries, fetch_ttl_s=2..5)`.
- **Push** at end of `commit()`: best-effort single-ref `git push`, timeout + retry then **swallow** (commit already succeeded locally; create-only ⇒ idempotent).
- **Fetch** at top of `list()` (the single funnel `best`/`derive_trunks`/`derive_families` use): best-effort `git fetch refs/attempts/*`, **TTL-debounced (2–5 s)** so one `status()` pass = one fetch; degrade-to-local on failure. `get`/`lineage` don't fetch.
- **Guard:** keep `GitAttemptHistory` storage-only — do **not** auto-wire `remote=...` into `optimizers/simple.py` live runs until siting is solved (a `NotImplementedError`/explicit opt-in guard documents the gate).
- Test NT4 (git-only): two stores over one bare remote, conflict-free both ways; commit succeeds with the remote unreachable.

**Exit:** NT4 passes; offline (`remote=None`) is a no-op; full suite green for both params (sync tests git-only).

---

## File manifest (~21 files)

- **New:** `histories/git.py`, `tests/conftest.py`
- **Edited (core):** `base/attempt_history.py`, `histories/folder.py`, `utils/direction.py` (reader-from-attempt), `__init__.py`, `pyproject.toml`, `.github/workflows/test.yml` (→ pytest), `tests/test_concepts.py`
- **Edited (consumers):** `utils/selection.py`, `agents/tools.py`, `strategies/{agent, improve, cross_pollinate, cross_pollinate_agent, fresh, fresh_agent, analyse, plan_approaches}.py`, `optimizers/simple.py`, `templates/{strategy, mock_strategy}.py`

---

## Risks / gotchas

1. **`core.autocrlf false` is mandatory** on the store — else `add -A` rewrites line endings → commit hash (= identity) becomes non-reproducible on Windows. Correctness bug.
2. **`list()` must be chronological from P1** — `simple.py:460` treats `list()[-1]` as the newest attempt; git refs aren't ordered by time. Sort by `created_at` + deterministic tie-break.
3. **Fail vs abort across two methods** — `commit(success=False)` writes a kept `Status:fail` commit; `abort()` writes nothing. Never collapse one into the other.
4. **pytest migration:** verify *every* existing script-style test still collects + passes under `uv run pytest` (some `capsys` tests that the old runner SKIPPED will now actually run). The `__main__` blocks can stay.
5. **Per-attempt `GIT_INDEX_FILE`** mandatory (shared index cross-contaminates concurrent trees); inject author/committer env; reads tolerate missing files (`check=False` + degrade).
6. **Strict dependency order:** P0 fully lands before P1; P2 adds the `status`/`name` trailers; P3 rides P1's create-only refs.
7. **PARKED gate (not solved here):** agents (codex/opencode) key off `.git`; even `-c project_root_markers=[]` doesn't shrink the writable area (`docs/sandboxing.md`). The plain-dir workspace + separate bare store are safe for the storage MVP + tests (no live agent runs), but `GitAttemptHistory` stays storage-only until `base_path` is sited **outside** any enclosing repo and the sandbox boundary is settled. **This gates P3 live-wiring and any real agent run.**
