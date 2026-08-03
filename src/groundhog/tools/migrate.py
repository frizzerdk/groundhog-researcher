"""Migrate a folder-backend run to the git attempt store.

The source run is never touched. The destination is a copy of the run
(excluding the venv, caches, and the old ``attempts/``) whose store is
rebuilt by replaying every committed attempt through GitAttemptHistory's
own API (workspace -> copy files -> commit -> notes), preserving parent
lineage, status, creation time (the source dir mtime becomes the commit's
author date), every ``notes.json`` key, and the folder id as
``migrated_from_folder_id`` metadata. In-progress workspaces have no git
equivalent to replay into, so migration refuses until they are committed
or aborted in the source.

``wire_git_history`` is the shared task.py patcher: it points a template's
``build_toolkit()`` at GitAttemptHistory. ``groundhog init --git`` applies
it to a fresh scaffold; migration applies it to the copied run.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import List

COPY_EXCLUDE = {".venv", "attempts", "__pycache__", ".pytest_cache"}

# Lean by default: work/ holds scratch and large artifacts (opt in with
# full_work); notes.json is re-set through the git backend's own notes.
REPLAY_EXCLUDE = {"work", "notes.json"}

# The injected lines land inside build_toolkit(); `import pathlib as ...`
# (not `from pathlib import Path`) so a Path already used earlier in that
# function body is not shadowed into an UnboundLocalError.
_GIT_WIRING = (
    "import pathlib as _pathlib\n"
    "from groundhog import GitAttemptHistory\n"
    "# Pass remote= to sync the store with a shared bare repo:\n"
    '# history = GitAttemptHistory(_pathlib.Path(__file__).parent, remote="git@github.com:you/attempts-store.git")\n'
    "history = GitAttemptHistory(_pathlib.Path(__file__).parent)\n"
)


def wire_git_history(task_py: Path) -> bool:
    """Patch a task.py's ``assemble_toolkit(task, ...)`` call to pass a
    GitAttemptHistory. Returns False when no patchable call was found, when
    the call already passes its own ``history=`` (never produce a duplicate
    kwarg), or when the patch would not compile; True when patched or when
    the call site already carries our wiring (idempotent)."""
    task_py = Path(task_py)
    text = task_py.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("#") or "assemble_toolkit(task" not in line:
            continue
        if re.search(r"\bhistory\s*=", line):
            return "assemble_toolkit(task, history=history" in line
        indent = line[:len(line) - len(stripped)]
        prelude = "".join(indent + w + "\n" for w in _GIT_WIRING.splitlines())
        patched = line.replace("assemble_toolkit(task",
                               "assemble_toolkit(task, history=history", 1)
        new_text = "".join(lines[:i] + [prelude + patched] + lines[i + 1:])
        try:
            compile(new_text, str(task_py), "exec")
        except SyntaxError:
            return False
        task_py.write_text(new_text, encoding="utf-8")
        return True
    return False


class MigrationError(RuntimeError):
    """A refused or failed migration — message is user-facing."""


def plan_migration(src: Path) -> List[dict]:
    """Read the source folder store and return replayable entries in id
    order. Raises MigrationError on a non-folder store or open workspaces."""
    from groundhog.histories.folder import FolderAttemptHistory

    src = Path(src)
    attempts_dir = src / "attempts"
    if not attempts_dir.exists():
        raise MigrationError(f"no attempts/ store in {src}")
    if (attempts_dir / ".git").exists():
        raise MigrationError(f"{attempts_dir} is already a git attempt store")

    history = FolderAttemptHistory(src)
    open_ws = history.list_in_progress()
    if open_ws:
        ids = ", ".join(ip.workspace_id for ip in open_ws)
        raise MigrationError(
            f"in-progress workspaces exist ({ids}) - commit or abort them "
            f"in the source first, then re-run")

    entries = []
    for a in history.list(only_done=False):
        if a.status not in ("done", "fail"):
            continue
        entries.append({
            "id": a.id,
            "parent": a.parent,
            "success": a.status == "done",
            "path": a.path,
            "name": a.metadata.get("name", ""),
            "created_at": a.created_at,
            "notes": _read_notes(a.path),
        })
    entries.sort(key=lambda e: int(e["id"]))
    return entries


def _read_notes(attempt_path: Path) -> dict:
    try:
        raw = json.loads((Path(attempt_path) / "notes.json")
                         .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def copy_run(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True)
    for item in Path(src).iterdir():
        if item.name in COPY_EXCLUDE:
            continue
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target,
                            ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(item, target)


def replay(entries: List[dict], dest: Path, *, full_work: bool = False,
           out=print) -> dict:
    """Replay committed folder attempts through GitAttemptHistory in id
    order. Returns the folder-id -> git-sha map."""
    from groundhog import GitAttemptHistory

    history = GitAttemptHistory(dest)
    id_map: dict = {}
    try:
        for e in entries:
            id_map[e["id"]] = _replay_one(history, e, id_map,
                                          full_work=full_work, out=out)
    except BaseException:
        out(f"ERROR: replay stopped mid-way - {dest} holds a PARTIAL store "
            f"and must be deleted before retrying")
        raise
    return id_map


def _replay_one(history, e: dict, id_map: dict, *, full_work: bool, out) -> str:
    parent_sha = id_map.get(e["parent"])
    if e["parent"] is not None and parent_sha is None:
        out(f"  WARNING {e['id']}: parent {e['parent']} was not migrated; "
            f"committing as root")
    ws = history.workspace(parent=parent_sha)
    for item in Path(e["path"]).iterdir():
        if item.name in REPLAY_EXCLUDE:
            continue
        if item.is_dir():
            shutil.copytree(item, ws.path / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, ws.path / item.name)
    if full_work and (e["path"] / "work").exists():
        shutil.copytree(e["path"] / "work", ws.path / "work",
                        dirs_exist_ok=True)

    meta_f = ws.path / "metadata.json"
    meta = {}
    if meta_f.exists():
        try:
            meta = json.loads(meta_f.read_text(encoding="utf-8"))
        except ValueError:
            meta = {}
    meta["migrated_from_folder_id"] = int(e["id"])
    meta_f.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    ws.name = e["name"] or _direction_first_line(ws.path)
    ws.created_at = e["created_at"]
    attempt = ws.commit(success=e["success"])
    for key, value in e["notes"].items():
        history.set_note(attempt, key, value)
    noted = f", notes: {', '.join(sorted(e['notes']))}" if e["notes"] else ""
    out(f"  {e['id']:>3} -> {attempt.id[:10]}  "
        f"({'done' if e['success'] else 'fail'}{noted})")
    return attempt.id


def _direction_first_line(path: Path) -> str:
    direction = Path(path) / "core_direction.md"
    if not direction.exists():
        return ""
    try:
        lines = direction.read_text(encoding="utf-8",
                                    errors="replace").strip().splitlines()
    except OSError:
        return ""
    return lines[0].strip() if lines else ""


def migrate_store(src: Path, dest: Path, *, full_work: bool = False,
                  dry_run: bool = False, out=print) -> None:
    """The whole move: plan, copy, replay, patch. Raises MigrationError on
    any refusal (source untouched in every case)."""
    src, dest = Path(src).resolve(), Path(dest).resolve()
    if dest.exists():
        raise MigrationError(f"dest exists: {dest} - refusing to overwrite")
    if dest.is_relative_to(src):
        raise MigrationError("dest must be OUTSIDE the source run dir")

    entries = plan_migration(src)
    out(f"source: {src}")
    out(f"found {len(entries)} committed attempts "
        f"({sum(1 for e in entries if e['success'])} done, "
        f"{sum(1 for e in entries if not e['success'])} fail), "
        f"{sum(1 for e in entries if 'score' in e['notes'])} score notes")
    if dry_run:
        out("dry run - nothing written")
        return

    out(f"copying run -> {dest} (excluding {', '.join(sorted(COPY_EXCLUDE))})")
    copy_run(src, dest)
    out("migrating attempts through GitAttemptHistory:")
    replay(entries, dest, full_work=full_work, out=out)
    if wire_git_history(dest / "task.py"):
        out("task.py wired to GitAttemptHistory")
    else:
        out("WARNING: could not patch task.py automatically - pass "
            "history=GitAttemptHistory(run_dir) to assemble_toolkit yourself")
