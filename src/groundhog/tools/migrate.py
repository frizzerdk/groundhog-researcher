"""Migrate a folder-backend run to the git attempt store.

The source run is never touched. The destination is a copy of the run
(excluding the venv, caches, and the old ``attempts/``) whose store is
rebuilt by replaying every committed attempt through GitAttemptHistory's
own API (workspace -> write files -> commit -> score note), preserving
parent lineage, status, score notes, and the folder id as
``migrated_from_folder_id`` metadata. In-progress workspaces have no git
equivalent to replay into, so migration refuses until they are committed
or aborted in the source.

``wire_git_history`` is the shared task.py patcher: it points a template's
``build_toolkit()`` at GitAttemptHistory. ``groundhog init --git`` applies
it to a fresh scaffold; migration applies it to the copied run.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List

COPY_EXCLUDE = {".venv", "attempts", "__pycache__", ".pytest_cache"}

# Lean by default: work/ dirs hold scratch and large artifacts. notes.json
# stays behind — the git backend's notes are git notes, re-set at replay.
ESSENTIALS = ["solution.py", "core_direction.md", "metadata.json",
              "result.json", "attemptlog.jsonl", "attemptlog.md",
              "TASK_CONTEXT.md", "agent_steps.jsonl", "agent_summary.jsonl"]

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
    GitAttemptHistory. Returns False when no patchable call was found;
    True when patched or already git-aware (idempotent)."""
    task_py = Path(task_py)
    text = task_py.read_text(encoding="utf-8")
    if "GitAttemptHistory" in text:
        return True
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("#") or "assemble_toolkit(task" not in line:
            continue
        indent = line[:len(line) - len(stripped)]
        prelude = "".join(indent + w + "\n" for w in _GIT_WIRING.splitlines())
        patched = line.replace("assemble_toolkit(task",
                               "assemble_toolkit(task, history=history", 1)
        lines[i] = prelude + patched
        task_py.write_text("".join(lines), encoding="utf-8")
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
            "score_note": history.get_note(a, "score"),
        })
    entries.sort(key=lambda e: int(e["id"]))
    return entries


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
    for e in entries:
        parent_sha = id_map.get(e["parent"])
        if e["parent"] is not None and parent_sha is None:
            out(f"  WARNING {e['id']}: parent {e['parent']} was not migrated; "
                f"committing as root")
        ws = history.workspace(parent=parent_sha)
        for name in ESSENTIALS:
            src_f = e["path"] / name
            if src_f.exists():
                shutil.copy2(src_f, ws.path / name)
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
        attempt = ws.commit(success=e["success"])
        id_map[e["id"]] = attempt.id
        if e["score_note"] is not None:
            history.set_note(attempt, "score", str(e["score_note"]))
        out(f"  {e['id']:>3} -> {attempt.id[:10]}  "
            f"({'done' if e['success'] else 'fail'}"
            f"{', score noted' if e['score_note'] is not None else ''})")
    return id_map


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
        f"{sum(1 for e in entries if e['score_note'] is not None)} score notes")
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
