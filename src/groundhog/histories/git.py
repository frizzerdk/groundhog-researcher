"""Git-backed attempt history — a browsable git repo scoped to ``attempts/``.

The run folder is a PLAIN folder holding three peers — your task (``task.py``),
the ``attempts/`` store, and the knowledge store (``learnings.md``). Git is
scoped to the attempt history alone::

    <run>/                 ← plain folder (your task dir; NOT a repo)
        task.py            ← the task (+ entry point)
        attempts/          ← the attempt store
            .git/          ← a BARE repo (objects, refs, worktrees admin)
            <slug>/        ← a worktree per attempt, on branch attempt/<sha>
        learnings.md       ← the knowledge store

So ``cd attempts && git log --all --graph`` shows the search tree, and opening
any ``attempts/<slug>/`` behaves like a worktree. ``attempts/.git`` is **bare**:
there is no primary worktree to pollute ``attempts/`` or to keep clean — the
only working trees are the attempt folders themselves.

Model:

- An attempt is **one commit**; the commit's parent is the logical parent. Fresh
  ("root") attempts branch off an empty **base commit** on ``main`` — so the
  graph is connected and rooted — yet read as ``parent = None`` (the base is the
  origin, not an attempt).
- **Identity = the commit sha**; the branch is ``attempt/<sha>`` (collision-proof,
  so there is never any name juggling on branches or commits).
- **Readable name = the slug**: the folder name (how you navigate on disc) and
  the commit subject (how ``git log`` reads), with ``metadata.json`` its source
  of truth. The folder is the only collision spot (two dirs can't share a name)
  → it borrows the commit's short hash on a clash (``dropout`` → ``dropout-a8ccd75``).
- **Lifecycle:** ``workspace(parent)`` adds a worktree on ``wip/<uuid>`` off the
  parent commit; the strategy writes files; ``commit()`` makes the single
  commit, renames the branch to ``attempt/<sha>`` and the folder to its slug;
  ``abort()`` removes the worktree and branch.
- **Reads** come from the object store (``git show`` / ``ls-tree``), never a
  worktree.
- Per-origin **create-only** refs ``refs/attempts/<origin>/<sha>`` index the
  attempts and carry optional conflict-free sync.

Git config is neutralized and identity/date injected per call, so commit hashes
never depend on machine configuration; ``core.autocrlf`` is forced off.
"""

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from groundhog.base.types import EvaluationResult, StageResult
from groundhog.base.attempt_history import (
    Attempt, Workspace, AttemptHistory, InProgress)
from groundhog.utils.results import read_result, write_metadata, read_attempt_metadata

_NOTE_KEY = re.compile(r"^[a-z0-9_-]{1,64}$")
from groundhog.utils.direction import slugify


_IDENTITY_NAME = "Groundhog"
_IDENTITY_EMAIL = "groundhog@localhost"

# Field separator for ``git show`` format strings (unit separator, 0x1f).
_FS = "\x1f"

_BINARY_EXTS = {".png", ".gif", ".jpg", ".jpeg", ".bmp", ".ico", ".pdf",
                ".zip", ".gz", ".tar", ".bin", ".pkl", ".npy", ".npz",
                ".whl", ".so", ".dll", ".exe", ".pyc"}


@dataclass
class SyncPolicy:
    """When and how a synced store talks to its remote. All best-effort —
    refs are per-origin and create-only, so sync is conflict-free."""
    push_after_commit: bool = True
    fetch_before_reads: bool = True
    timeout_s: float = 10.0
    fetch_ttl_s: float = 3.0
    push_retries: int = 2


class GitError(RuntimeError):
    """A git command exited non-zero."""

    def __init__(self, cmd, returncode, stderr):
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = (stderr or b"").decode("utf-8", "replace")
        super().__init__(
            f"git {' '.join(str(c) for c in cmd[1:])} exited {returncode}: "
            f"{self.stderr.strip()}"
        )


def _binary_placeholder(nbytes: int) -> str:
    return f"[binary file: {nbytes / 1024:.0f}KB — use your file viewer to inspect]"


class GitAttempt(Attempt):
    """A committed attempt, addressed by its commit sha.

    Content reads (``code`` / ``result`` / ``metadata`` / ``read_file``) go
    through the object store — no worktree required. ``path`` is the on-disk
    view: it materializes the attempt's worktree on demand (may run git), for
    consumers that need a real folder (CLI tools, workspace-relative reads)."""

    def __init__(self, history: "GitAttemptHistory", id: str,
                 parent: Optional[str], created_at: float, status: str):
        self._history = history
        self.id = id
        self.parent = parent
        self.created_at = created_at
        self._status = status

    @property
    def status(self) -> str:
        return self._status

    @property
    def code(self) -> str:
        return self.read_file("solution.py") or ""

    @property
    def result(self) -> EvaluationResult:
        text = self.read_file("result.json")
        return read_result(json.loads(text)) if text else read_result({})

    @property
    def metadata(self) -> dict:
        return read_attempt_metadata(self)

    @property
    def name(self) -> str:
        return self.metadata.get("name", "")

    @property
    def path(self) -> Path:
        """This attempt's worktree folder, materialized on demand.

        Disk state is dynamic (pruned folders, synced clones without
        worktrees) — the folder is ensured on first access. Prefer ``code`` /
        ``read_file`` for content; use ``path`` when a real directory is
        needed."""
        return self._history.materialize(self.id)

    def list_files(self) -> List[str]:
        return self._history._list_files(self.id)

    def read_file(self, path: str) -> Optional[str]:
        return self._history._read_file(self.id, path)

    def __repr__(self):
        p = self.parent[:8] if self.parent else None
        return f"GitAttempt({self.id[:8]}, parent={p})"


class GitWorkspace(Workspace):
    """A live workspace = a git worktree under ``attempts/<uuid>/`` on a
    throwaway ``wip/<uuid>`` branch off the parent commit. ``commit()`` makes
    the single attempt commit, renames the branch to ``attempt/<sha>`` and the
    folder to its slug; ``abort()`` removes the worktree and branch."""

    def __init__(self, history: "GitAttemptHistory", display_id: str,
                 parent: Optional[str], path: Path, wip_branch: str):
        self._history = history
        self.display_id = display_id
        self.name = ""
        self.parent = parent
        self.path = path
        self._wip_branch = wip_branch

    def commit(self, success: bool = True) -> GitAttempt:
        return self._history._commit_workspace(self, success)

    def abort(self):
        self._history._abort_workspace(self)

    def heartbeat(self):
        """Refresh the pid+timestamp liveness marker so ``reap`` can tell a
        working session from a crashed one."""
        self._history._write_heartbeat(self.display_id)


class GitAttemptHistory(AttemptHistory):
    """Attempt history as a browsable git repo scoped to ``<run>/attempts/``.
    One commit per attempt; each attempt is a worktree folder named by its
    slug, on an ``attempt/<sha>`` branch."""

    def __init__(self, base_path, remote: Optional[str] = None,
                 policy: Optional[SyncPolicy] = None):
        # base_path is the plain run/task folder. The git store lives in the
        # attempts/ subdir (mirrors the folder backend's <base>/attempts), so
        # task.py and learnings.md sit beside attempts/, outside git.
        self._root = Path(base_path)
        self._store = self._root / "attempts"
        self._repo = self._store          # default cwd for the git chokepoint
        self._git_dir = self._store / ".git"   # a BARE repo
        self._attempts = self._store      # worktree folders live directly here
        # Failed deletes are parked inside .git so they never linger on disc.
        self._trash_dir = self._git_dir / "groundhog-trash"
        # Heartbeats for in-progress (uncommitted) workspaces — outside any
        # tree, so never committed; pid + start time per open wip/<id>.
        self._wip_dir = self._git_dir / "groundhog-wip"

        self._last_created = 0.0
        self._remote = str(remote) if remote else None
        self._policy = policy or SyncPolicy()
        self._last_fetch = 0.0

        self._root.mkdir(parents=True, exist_ok=True)
        self._store.mkdir(parents=True, exist_ok=True)
        if not self._git_dir.exists():
            self._init_store()
        self._trash_dir.mkdir(exist_ok=True)
        self._wip_dir.mkdir(exist_ok=True)
        self._load_origin()
        self._base_sha = self._git_text("rev-parse", "main")
        self._reap_trash()

    # --- store setup ----------------------------------------------------

    def _init_store(self):
        # A bare repo at attempts/.git: no primary worktree, so attempts/ holds
        # only .git + the attempt worktree folders, and nothing needs excluding.
        self._git("init", "--bare", "-b", "main", str(self._git_dir),
                  cwd=self._store)
        self._git("config", "core.autocrlf", "false")
        # Empty base commit on main via plumbing (bare → no index/worktree).
        # The origin every fresh attempt branches from; main never moves off it.
        empty_tree = self._git_text("hash-object", "-t", "tree", "--stdin",
                                    input_bytes=b"")
        base = self._git_text("commit-tree", empty_tree,
                              input_bytes=b"store: task root\n")
        self._git("update-ref", "refs/heads/main", base)

    def _load_origin(self):
        origin_file = self._git_dir / "groundhog-origin"
        if origin_file.exists():
            self.origin = origin_file.read_text(encoding="utf-8").strip()
        else:
            self.origin = uuid.uuid4().hex[:12]
            origin_file.write_text(self.origin, encoding="utf-8")

    def _reap_trash(self):
        if not self._trash_dir.exists():
            return
        for child in self._trash_dir.iterdir():
            shutil.rmtree(child, ignore_errors=True)

    # --- git chokepoint -------------------------------------------------

    def _git(self, *args, input_bytes: Optional[bytes] = None,
             author_date: Optional[str] = None, check: bool = True,
             timeout: Optional[float] = None,
             cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        cmd = ["git"] + [str(a) for a in args]

        env = dict(os.environ)
        for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE",
                    "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR"):
            env.pop(var, None)
        env["GIT_CONFIG_GLOBAL"] = os.devnull
        env["GIT_CONFIG_SYSTEM"] = os.devnull
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_AUTHOR_NAME"] = env["GIT_COMMITTER_NAME"] = _IDENTITY_NAME
        env["GIT_AUTHOR_EMAIL"] = env["GIT_COMMITTER_EMAIL"] = _IDENTITY_EMAIL
        if author_date is not None:
            env["GIT_AUTHOR_DATE"] = author_date
            env["GIT_COMMITTER_DATE"] = author_date

        run_cwd = str(cwd) if cwd is not None else str(self._repo)
        result = subprocess.run(
            cmd, env=env, input=input_bytes, cwd=run_cwd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
        )
        if check and result.returncode != 0:
            raise GitError(cmd, result.returncode, result.stderr)
        return result

    def _git_text(self, *args, **kw) -> str:
        return self._git(*args, **kw).stdout.decode("utf-8", "replace").strip()

    # --- workspace lifecycle -------------------------------------------

    def workspace(self, parent: Optional[str] = None) -> GitWorkspace:
        parent = str(parent) if parent is not None else None
        self._reap_trash()
        ws_id = uuid.uuid4().hex
        ws_path = self._attempts / ws_id
        wip_branch = f"wip/{ws_id[:12]}"
        # Branch the worktree off the parent commit (or the base for a fresh
        # attempt) so its checkout seeds from the parent AND the eventual
        # commit's git-parent is the logical parent — lineage for free.
        parent_commit = parent if parent is not None else self._base_sha
        self._git("worktree", "add", "-b", wip_branch, str(ws_path),
                  parent_commit)
        # Start EMPTY like the folder backend: strip the inherited parent tree
        # so a child never carries the parent's result.json / metadata.json /
        # logs. Lineage is preserved by the branch HEAD = parent_commit (the
        # eventual commit's git-parent), NOT by leftover working-tree files —
        # the strategy then copies in only the convention set (solution.py,
        # core_direction.md, TASK_CONTEXT.md). Best-effort: a fresh attempt's
        # base is the empty tree, so there is nothing to strip.
        self._git("rm", "-r", "--quiet", "--ignore-unmatch", "--", ".",
                  cwd=ws_path, check=False)
        (ws_path / "work").mkdir(exist_ok=True)
        self._write_heartbeat(ws_id[:12])
        return GitWorkspace(self, display_id=ws_id[:12], parent=parent,
                            path=ws_path, wip_branch=wip_branch)

    def _commit_workspace(self, ws: GitWorkspace, success: bool) -> GitAttempt:
        status = "done" if success else "fail"

        ts = time.time()
        if ts <= self._last_created:
            ts = self._last_created + 1e-6
        self._last_created = ts
        date = f"@{int(ts)} +0000"

        # The readable name → metadata.json (source of truth) + the slug folder
        # + the commit subject. Status/parent mirrored into metadata so the
        # folder is self-describing on disc.
        meta = {"status": status, "parent": ws.parent or "none"}
        if ws.name:
            meta["name"] = ws.name
        write_metadata(ws.path, meta)

        subject = ws.name or "attempt"
        message = (f"{subject}\n\nStatus: {status}\n"
                   f"Groundhog-Created: {ts:.6f}\n")

        self._git("add", "-A", cwd=ws.path)
        self._git("commit", "--allow-empty", "-F", "-",
                  input_bytes=message.encode("utf-8"),
                  author_date=date, cwd=ws.path)
        sha = self._git_text("rev-parse", "HEAD", cwd=ws.path)

        # Index the attempt for reads/sync (create-only, per-origin).
        self._git("update-ref", f"refs/attempts/{self.origin}/{sha}", sha)
        # Rename the workspace branch + folder to their final names.
        self._git("branch", "-m", ws._wip_branch, f"attempt/{sha[:12]}")
        self._settle_folder(ws, sha)
        self._clear_heartbeat(ws.display_id)
        self._push_ref(sha)
        return self._load_attempt(sha)

    def _settle_folder(self, ws: GitWorkspace, sha: str):
        """Rename the worktree dir from its uuid to the readable slug; on a
        name clash borrow the commit's short hash. Best-effort — reads never
        depend on the folder."""
        slug = slugify(ws.name) if ws.name else ""
        leaf = slug or f"attempt-{sha[:12]}"
        dest = self._attempts / leaf
        if dest.exists():
            dest = self._attempts / f"{leaf}-{sha[:8]}"
        try:
            self._git("worktree", "move", str(ws.path), str(dest))
            ws.path = dest
        except (GitError, OSError):
            pass

    def _abort_workspace(self, ws: GitWorkspace):
        try:
            self._git("worktree", "remove", "--force", str(ws.path))
        except (GitError, OSError):
            self._remove_path(ws.path)
            self._git("worktree", "prune", check=False)
        self._git("branch", "-D", ws._wip_branch, check=False)
        self._clear_heartbeat(ws.display_id)

    def _remove_path(self, path: Path):
        if not path.exists():
            return
        try:
            shutil.rmtree(path)
        except OSError:
            try:
                dest = self._trash_dir / f"{path.name}-{uuid.uuid4().hex[:8]}"
                os.replace(path, dest)
            except OSError:
                pass

    # --- in-progress (list / resume / reap) ----------------------------
    # A wip/<id> branch + its worktree dir persist on disc through a crash (with
    # the uncommitted edits still in the folder), so an open workspace is
    # listable and resumable without committing it. A heartbeat (pid + start
    # time) under .git/groundhog-wip tells a crashed wip from a live one.

    def _heartbeat(self, wsid: str) -> Path:
        return self._wip_dir / wsid

    def _write_heartbeat(self, wsid: str):
        try:
            self._heartbeat(wsid).write_text(
                json.dumps({"pid": os.getpid(), "started_at": time.time()}),
                encoding="utf-8")
        except OSError:
            pass

    def _read_heartbeat(self, wsid: str) -> dict:
        try:
            return json.loads(self._heartbeat(wsid).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _clear_heartbeat(self, wsid: str):
        try:
            self._heartbeat(wsid).unlink()
        except OSError:
            pass

    @staticmethod
    def _pid_alive(pid) -> bool:
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return False
        if pid <= 0:
            return False
        try:
            if os.name == "nt":
                import ctypes
                h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
                if h:
                    ctypes.windll.kernel32.CloseHandle(h)
                    return True
                return False
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _worktree_for_branch(self, branch: str) -> Optional[Path]:
        out = self._git_text("worktree", "list", "--porcelain")
        target = f"branch refs/heads/{branch}"
        cur = None
        for line in out.splitlines():
            line = line.rstrip()
            if line.startswith("worktree "):
                cur = line[len("worktree "):].strip()
            elif line == target:
                return Path(cur) if cur else None
        return None

    def _detached_worktree_at(self, sha: str) -> Optional[Path]:
        """Find an existing DETACHED worktree checked out at ``sha``.

        Only detached ones count: a wip worktree can sit at the same commit
        (a child branches AT its parent before its first commit), and
        returning that would expose the child's dirty edits as the parent's
        content. Detached worktrees are created only by ``materialize``.
        """
        out = self._git_text("worktree", "list", "--porcelain")
        blocks = out.split("\n\n")
        for block in blocks:
            path = head = None
            detached = False
            for line in block.splitlines():
                line = line.rstrip()
                if line.startswith("worktree "):
                    path = line[len("worktree "):].strip()
                elif line.startswith("HEAD "):
                    head = line[len("HEAD "):].strip()
                elif line == "detached":
                    detached = True
            if detached and head == sha and path and Path(path).exists():
                return Path(path)
        return None

    def list_in_progress(self) -> List[InProgress]:
        out = self._git_text("for-each-ref", "--format=%(refname:short)",
                             "refs/heads/wip/")
        items = []
        for ref in out.splitlines():
            ref = ref.strip()
            if not ref.startswith("wip/"):
                continue
            wsid = ref[len("wip/"):]
            tip = self._git_text("rev-parse", ref)
            parent = None if tip == self._base_sha else tip
            hb = self._read_heartbeat(wsid)
            items.append(InProgress(
                workspace_id=wsid,
                parent=parent,
                started_at=float(hb.get("started_at", 0.0)),
                path=self._worktree_for_branch(ref),
                live=self._pid_alive(hb.get("pid")),
            ))
        items.sort(key=lambda ip: ip.started_at)
        return items

    def resume(self, workspace_id: str) -> GitWorkspace:
        branch = f"wip/{workspace_id}"
        res = self._git("rev-parse", "--verify", "--quiet", branch, check=False)
        if res.returncode != 0:
            raise KeyError(f"no in-progress workspace {workspace_id!r}")
        tip = res.stdout.decode("utf-8").strip()
        parent = None if tip == self._base_sha else tip
        path = self._worktree_for_branch(branch)
        if path is None or not Path(path).exists():
            # Worktree dir gone; rematerialize at the wip tip (the seed state —
            # uncommitted edits were only recoverable while the dir survived).
            path = self._attempts / workspace_id
            self._git("worktree", "prune", check=False)
            self._git("worktree", "add", str(path), branch)
        self._write_heartbeat(workspace_id)   # this process now owns it
        return GitWorkspace(self, display_id=workspace_id, parent=parent,
                            path=Path(path), wip_branch=branch)

    def reap_in_progress(self, ttl_s: float = 300.0) -> int:
        now = time.time()
        reaped = 0
        for ip in self.list_in_progress():
            # Contract: leave LIVE sessions alone regardless of age (a working
            # agent easily exceeds any TTL — the old `live and recent` check
            # force-killed it, audit 2026-07-01 bug #2). The TTL is a grace
            # period applied to dead pids only.
            if ip.live:
                continue
            if (now - ip.started_at) <= ttl_s:
                continue   # dead but recent — grace period
            branch = f"wip/{ip.workspace_id}"
            if ip.path is not None:
                try:
                    self._git("worktree", "remove", "--force", str(ip.path))
                except (GitError, OSError):
                    self._remove_path(Path(ip.path))
            self._git("worktree", "prune", check=False)
            self._git("branch", "-D", branch, check=False)
            self._clear_heartbeat(ip.workspace_id)
            reaped += 1
        return reaped

    def set_note(self, attempt_or_id, key: str, value: str) -> None:
        """Mutable annotation as a real git note (``refs/notes/groundhog/<key>``).

        Shows up natively in the browsable store:
        ``git log --show-notes=groundhog/<key>``. Overwrites on re-set; the
        attempt commit itself is untouched (immutability holds — notes are a
        scratch channel, e.g. the latest computed score cache).
        """
        if not _NOTE_KEY.match(key or ""):
            raise ValueError(f"invalid note key {key!r} (use [a-z0-9_-], max 64)")
        sha = attempt_or_id.id if hasattr(attempt_or_id, "id") else str(attempt_or_id)
        res = self._git("rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}",
                        check=False)
        if res.returncode != 0:
            raise KeyError(f"unknown attempt {sha!r}")
        self._git("notes", f"--ref=refs/notes/groundhog/{key}",
                  "add", "-f", "-m", str(value), sha)

    def get_note(self, attempt_or_id, key: str) -> Optional[str]:
        if not _NOTE_KEY.match(key or ""):
            return None
        sha = attempt_or_id.id if hasattr(attempt_or_id, "id") else str(attempt_or_id)
        res = self._git("notes", f"--ref=refs/notes/groundhog/{key}",
                        "show", sha, check=False)
        if res.returncode != 0:
            return None
        return res.stdout.decode("utf-8", errors="replace").strip()

    def materialize(self, attempt_or_id) -> Path:
        """Ensure the attempt's worktree folder exists on disk; return it.

        Disk is dynamic by design: worktrees may be pruned, and a store
        synced from a remote arrives as git objects with NO worktrees at all.
        This is the one place a folder is (re)created — generalizing the
        rematerialize move ``resume()`` already performs for in-progress
        workspaces. Idempotent: an existing worktree is returned as-is.
        """
        sha = attempt_or_id.id if hasattr(attempt_or_id, "id") else str(attempt_or_id)
        res = self._git("rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}",
                        check=False)
        if res.returncode != 0:
            raise KeyError(f"unknown attempt {sha!r}")
        sha = self._git_text("rev-parse", f"{sha}^{{commit}}")

        branch = f"attempt/{sha[:12]}"
        existing = self._worktree_for_branch(branch)
        if existing is not None and Path(existing).exists():
            return Path(existing)
        # Synced attempts have no local branch — their worktrees are detached.
        # Without this check every call would mint a new duplicate checkout
        # (and the collision ladder eventually hard-fails).
        existing = self._detached_worktree_at(sha)
        if existing is not None:
            return existing

        # Recreate from the object store. Use the attempt branch when it
        # exists locally (keeps the branch<->worktree binding); a synced
        # store may only have the commit, so fall back to a detached checkout.
        self._git("worktree", "prune", check=False)
        attempt = self._load_attempt(sha)
        slug = slugify(attempt.name) if attempt.name else ""
        leaf = slug or f"attempt-{sha[:12]}"
        dest = self._attempts / leaf
        if dest.exists():
            dest = self._attempts / f"{leaf}-{sha[:8]}"
        if dest.exists():
            dest = self._attempts / f"{leaf}-{sha[:12]}"

        have_branch = self._git("rev-parse", "--verify", "--quiet",
                                f"refs/heads/{branch}", check=False).returncode == 0
        if have_branch:
            self._git("worktree", "add", str(dest), branch)
        else:
            self._git("worktree", "add", "--detach", str(dest), sha)
        return dest

    # --- object-store reads (no checkout) ------------------------------

    def _read_file(self, sha: str, path: str) -> Optional[str]:
        path = path.replace("\\", "/")
        res = self._git("show", f"{sha}:{path}", check=False)
        if res.returncode != 0:
            return None
        data = res.stdout
        if Path(path).suffix.lower() in _BINARY_EXTS:
            return _binary_placeholder(len(data))
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return _binary_placeholder(len(data))

    def _list_files(self, sha: str) -> List[str]:
        out = self._git_text("ls-tree", "-r", "--name-only", sha)
        return sorted(line for line in out.splitlines() if line)

    def _load_attempt(self, sha: str) -> GitAttempt:
        fmt = _FS.join(["%H", "%P", "%ct", "%B"])
        out = self._git("show", "-s", f"--format={fmt}", sha).stdout.decode("utf-8")
        h, parents, ct, body = out.split(_FS, 3)
        parent = parents.split()[0] if parents.strip() else None
        # The base commit is the origin, not an attempt: a child of the base is
        # a "root" attempt with no logical parent.
        if parent == self._base_sha:
            parent = None
        created, status = self._parse_trailers(body)
        if not created:
            try:
                created = float(ct.strip())
            except ValueError:
                created = 0.0
        return GitAttempt(self, id=h.strip(), parent=parent,
                          created_at=created, status=status)

    @staticmethod
    def _parse_trailers(body: str):
        created, status = 0.0, "done"
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("Groundhog-Created:"):
                try:
                    created = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("Status:"):
                status = line.split(":", 1)[1].strip() or "done"
        return created, status

    # --- sync (best-effort) --------------------------------------------

    def _push_ref(self, sha: str):
        if not (self._remote and self._policy.push_after_commit):
            return
        ref = f"refs/attempts/{self.origin}/{sha}"
        for _ in range(max(1, self._policy.push_retries)):
            try:
                self._git("push", self._remote, f"{ref}:{ref}",
                          timeout=self._policy.timeout_s)
                return
            except (GitError, subprocess.SubprocessError, OSError):
                continue

    def _maybe_fetch(self):
        if not (self._remote and self._policy.fetch_before_reads):
            return
        now = time.time()
        if now - self._last_fetch < self._policy.fetch_ttl_s:
            return
        self._last_fetch = now
        try:
            self._git("fetch", self._remote,
                      "+refs/attempts/*:refs/attempts/*",
                      timeout=self._policy.timeout_s)
        except (GitError, subprocess.SubprocessError, OSError):
            pass

    # --- queries -------------------------------------------------------

    def list(self, only_done: bool = True) -> List[GitAttempt]:
        self._maybe_fetch()
        out = self._git_text("for-each-ref", "--format=%(objectname)",
                             "refs/attempts/")
        shas = list(dict.fromkeys(s for s in out.splitlines() if s))
        attempts = [self._load_attempt(s) for s in shas]
        if only_done:
            attempts = [a for a in attempts if a.status == "done"]
        attempts.sort(key=lambda a: (a.created_at, a.id))
        return attempts

    def get(self, id: str) -> Optional[GitAttempt]:
        res = self._git("rev-parse", "--verify", "--quiet", f"{id}^{{commit}}",
                        check=False)
        if res.returncode != 0:
            return None
        sha = res.stdout.decode("utf-8").strip()
        if sha == self._base_sha:
            return None
        return self._load_attempt(sha)

    def best(self, scorer: Callable[[StageResult], float]) -> Optional[GitAttempt]:
        attempts = self.list()
        if not attempts:
            return None

        def score_attempt(attempt):
            result = attempt.result
            if not result.completed:
                return -1.0
            last_stage = list(result.stages.values())[-1]
            return scorer(last_stage)

        return max(attempts, key=score_attempt)

    def lineage(self, attempt: GitAttempt) -> List[GitAttempt]:
        chain = [attempt]
        current = attempt
        while current.parent is not None:
            current = self.get(current.parent)
            if current is None:
                break
            chain.append(current)
        chain.reverse()
        return chain
