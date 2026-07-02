"""Core direction helpers — the algorithmic invariant of a family.

A family of attempts is identified by ``core_direction.md`` — a narrow,
human-readable description of the algorithmic backbone (e.g. "CNN",
"rollout-greedy"). Refine and cross-pollinate inherit it byte-for-byte
from the parent; only fresh strategies mint new directions.

Vault: ``Optimizer/Implementation Details/Family Identity.md``.

The source of truth is the file content. Hashes for grouping/lookup may
be computed at read time but are not persisted per attempt.

Filename:
    Primary:  ``core_direction.md`` at attempt root.
    Legacy:   ``approach.md`` (fallback for older attempts) at attempt
              root, or ``work/core_direction.md`` /
              ``work/approach.md`` for fresh-style runs that wrote the
              direction in the agent's workspace.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Iterable, Optional


DIRECTION_FILENAME = "core_direction.md"
LEGACY_FILENAMES = ("approach.md",)

# Search order: prefer new filename at root, then legacy at root, then
# work/ variants (agents write there during fresh sessions).
_SEARCH_PATHS = (
    DIRECTION_FILENAME,
    *LEGACY_FILENAMES,
    f"work/{DIRECTION_FILENAME}",
    *(f"work/{name}" for name in LEGACY_FILENAMES),
)


def find_direction_path(attempt_dir: Path | str) -> Optional[Path]:
    """Return the first existing direction file in ``attempt_dir``, or None.

    Prefers the canonical name and root location over legacy / workspace
    variants. Returns ``None`` if no direction file is present.
    """
    base = Path(attempt_dir)
    for rel in _SEARCH_PATHS:
        candidate = base / rel
        if candidate.exists():
            return candidate
    return None


def read_direction(attempt_dir: Path | str) -> Optional[str]:
    """Read the direction text from ``attempt_dir``, or None if absent.

    Whitespace is preserved; use :func:`normalize_direction` before
    comparing for family identity. Non-UTF8 bytes are replaced rather
    than raised: a hand-written direction in a legacy encoding must
    degrade to a comparable string, never crash the gates or the commit.
    """
    path = find_direction_path(attempt_dir)
    if path is None:
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def read_direction_from_attempt(attempt) -> Optional[str]:
    """Read the direction text from a *committed* attempt via its read_file API.

    Backend-agnostic: works for the folder backend (disk) and the git backend
    (object store, no checkout) alike, so no consumer needs ``attempt.path``
    on a committed attempt. Tries the same search order as
    :func:`find_direction_path`.
    """
    reader = getattr(attempt, "read_file", None)
    if reader is None:
        return None
    for rel in _SEARCH_PATHS:
        text = reader(rel)
        if text is not None:
            return text
    return None


def write_direction(attempt_dir: Path | str, text: str) -> Path:
    """Write ``text`` to ``attempt_dir/core_direction.md``. Returns the path."""
    base = Path(attempt_dir)
    base.mkdir(parents=True, exist_ok=True)
    target = base / DIRECTION_FILENAME
    target.write_text(text.strip() + "\n", encoding="utf-8")
    return target


def inherit_direction(prior_dir: Path | str, ws_dir: Path | str) -> Optional[Path]:
    """Copy the parent's direction into the child workspace at canonical name.

    Read-fallback applies (legacy ``approach.md`` is migrated forward to
    ``core_direction.md``). Returns the destination path on success, or
    ``None`` if the parent has no direction recorded.

    Use at workspace prep time. Pair with :func:`enforce_inherited_direction`
    at commit time to soft-gate against agents rewriting it mid-session.
    """
    text = read_direction(prior_dir)
    if text is None:
        return None
    return write_direction(ws_dir, text)


def enforce_inherited_direction(
    ws_dir: Path | str, prior_dir: Path | str
) -> Optional[Path]:
    """Re-copy parent's direction into the workspace at commit, overwriting
    any agent-written variant.

    The same operation as :func:`inherit_direction` but explicitly named
    for the commit-time soft-gate use case: refine, agent, and
    cross-pollinate strategies call this just before
    ``ws.commit(...)`` so the family invariant survives sessions where
    the agent edited the file. Fresh strategies must NOT call this —
    they're minting a new direction.
    """
    return inherit_direction(prior_dir, ws_dir)


def promote_workspace_direction(ws_dir: Path | str) -> Optional[Path]:
    """Move an agent-written ``work/core_direction.md`` to the attempt root.

    Fresh strategies that delegate direction-writing to the agent should
    call this at commit. If the agent wrote ``work/core_direction.md``
    (or legacy ``work/approach.md``) and root has no direction yet,
    promote it. Returns the destination path or ``None`` if nothing to
    promote.
    """
    base = Path(ws_dir)
    root_target = base / DIRECTION_FILENAME

    # Don't clobber an existing root direction.
    if root_target.exists():
        return root_target

    for rel in (
        f"work/{DIRECTION_FILENAME}",
        *(f"work/{name}" for name in LEGACY_FILENAMES),
    ):
        src = base / rel
        if src.exists():
            shutil.copy2(str(src), str(root_target))
            return root_target
    return None


# --- Backend-agnostic parent reads (folder AND git) --------------------
#
# The *_from_attempt helpers read the parent via the committed-read API
# (Attempt.read_file / Attempt.code) instead of ``prior.path``. GitAttempt has
# no ``.path`` (reads come from the object store), so the older path-based
# inherit/enforce/dedup silently no-op on git; these work for both backends.

def inherit_direction_from_attempt(prior, ws_dir: Path | str) -> Optional[Path]:
    """Copy a parent ATTEMPT's core direction into ``ws_dir``, backend-agnostic.

    Reads the parent's direction via :func:`read_direction_from_attempt` and
    writes it at the canonical name. Returns the destination path, or ``None``
    if the parent records no direction. Drop-in for ``inherit_direction(
    prior.path, ws_dir)`` that also works on the git backend.
    """
    if prior is None:
        return None
    text = read_direction_from_attempt(prior)
    if text is None:
        return None
    return write_direction(ws_dir, text)


def inherited_direction_changed_from(ws_dir: Path | str, prior) -> bool:
    """True when the workspace direction differs from the parent ATTEMPT's
    (backend-agnostic counterpart of :func:`inherited_direction_changed`)."""
    if prior is None:
        return False
    parent = read_direction_from_attempt(prior)
    if parent is None:
        return False
    current = read_direction(ws_dir)
    return normalize_direction(current or "") != normalize_direction(parent)


def solution_matches_attempt(ws_dir: Path | str, other) -> bool:
    """True iff ``ws_dir/solution.py`` equals the ``other`` ATTEMPT's code.

    Backend-agnostic dedup: compares the workspace's solution against
    ``other.code`` (object store on git, disk on folder) rather than reading
    ``other.path/solution.py``.
    """
    if other is None:
        return False
    ours = Path(ws_dir) / "solution.py"
    if not ours.exists():
        return False
    try:
        other_code = other.code
    except Exception:
        return False
    if not other_code:
        return False
    try:
        return ours.read_text(encoding="utf-8") == other_code
    except OSError:
        return False


def _history_list(history, *, only_done: bool = True):
    """Call ``history.list`` while tolerating older implementations."""
    try:
        return history.list(only_done=only_done)
    except TypeError:
        return history.list()


def direction_exists(
    history,
    direction: str,
    *,
    exclude: Iterable[str] = (),
    only_done: bool = False,
) -> bool:
    """Return True if history already contains this normalized direction."""
    key = normalize_direction(direction)
    if not key or history is None:
        return False
    excluded = set(exclude)
    for attempt in _history_list(history, only_done=only_done):
        if getattr(attempt, "id", None) in excluded:
            continue
        text = read_direction_from_attempt(attempt)
        if normalize_direction(text or "") == key:
            return True
    return False


def inherited_direction_changed(ws_dir: Path | str, prior_dir: Path | str) -> bool:
    """True when the workspace root direction differs from its parent."""
    parent = read_direction(prior_dir)
    if parent is None:
        return False
    current = read_direction(ws_dir)
    return normalize_direction(current or "") != normalize_direction(parent)


def mark_result_failed(result, stage_name: str, reason: str) -> None:
    """Mutate an EvaluationResult into a failed gate result."""
    from groundhog.base.types import StageResult

    result.stages[stage_name] = StageResult(errors={stage_name: reason})
    result.completed = False
    result.failed_stage = stage_name


# --- Read-side: family grouping ----------------------------------------

# Two or more consecutive blank lines collapse to one. Trailing whitespace
# on each line is stripped. Leading/trailing whitespace on the whole
# document is stripped. Used for family-identity comparison so trivial
# formatting churn doesn't fragment a family.
_BLANK_LINE_RUN = re.compile(r"\n\s*\n\s*\n+")
_TRAILING_WS = re.compile(r"[ \t]+\n")


def normalize_direction(text: str) -> str:
    """Normalize direction text for family-identity comparison.

    Strips trailing whitespace per line, collapses runs of blank lines
    to a single blank line, and strips outer whitespace. Display the
    original text; use the normalized form only as a comparison key.
    """
    if not text:
        return ""
    text = _TRAILING_WS.sub("\n", text)
    text = _BLANK_LINE_RUN.sub("\n\n", text)
    return text.strip()


def direction_title(text: str, max_len: int = 60) -> str:
    """First non-empty, non-heading-marker line of ``text``, for display."""
    if not text:
        return "(no direction)"
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading markdown heading markers for display.
        line = line.lstrip("#").strip()
        if not line:
            continue
        # Strip a redundant "Core Direction:" label — the file IS the core
        # direction, and agents often title it exactly that way. Keeping it
        # would leak into every slug (core-direction-data-augmentation-...),
        # status line, and family title. A line that is ONLY the label falls
        # through to the first real content line.
        stripped = re.sub(r"(?i)^core[\s_-]*direction\s*[:\-—]*\s*", "", line).strip()
        if stripped != line and not stripped:
            continue
        line = stripped or line
        # ASCII ellipsis on purpose: this string goes to stdout and Windows
        # consoles on legacy codepages render "…" as garbage.
        if len(line) > max_len:
            return line[: max_len - 3] + "..."
        return line
    return "(no direction)"


def slugify(text: str, max_words: int = 6) -> str:
    """Lowercase, hyphenated, alphanumeric slug for a human-readable name."""
    text = (text or "").strip().lower()
    text = re.sub(r"[`\"']", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^a-z0-9-]", "", text)
    text = re.sub(r"-+", "-", text).strip("-")
    if max_words:
        text = "-".join(text.split("-")[:max_words])
    return text


def workspace_name(ws_dir: Path | str, explicit: Optional[str] = None) -> str:
    """Resolve an attempt's display name (human-readable, never a lookup key).

    ``explicit`` (e.g. a PlanApproaches slug) wins; otherwise slug the
    workspace's core-direction title. Returns "" when neither is available.
    """
    if explicit:
        return slugify(explicit)
    text = read_direction(ws_dir)
    if not text:
        return ""
    return slugify(direction_title(text))
