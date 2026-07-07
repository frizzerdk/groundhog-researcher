"""Learnings as ledger + lens — the attempts ARE the learnings.

The ledger is the per-attempt learning record, immutable and traveling
with the attempt: ``learnings.md`` at the attempt root (LLM strategies,
via :func:`record_attempt_learning` before commit) and the agent
scratchpad ``work/learnings.md`` (agent strategies). The run-root
``learnings.md`` that :class:`MarkdownLearnings` reads is a DERIVED
digest — a lens over the ledger, rebuildable at any time with
:func:`rebuild_digest` (``groundhog learnings rebuild``). Strategies
still append to it directly for compatibility, but nothing recorded
only there is canonical: a rebuild reconstructs the digest from the
attempts alone.

Without an LLM the rebuild is a deterministic curated concatenation:
newest entries first, grouped by direction family, exact duplicates
dropped, capped at ``max_entries``. With an LLM it is one pass that
merges the collected entries into at most ``max_entries`` directive
entries. Either way the output keeps the ``---`` entry convention so
``MarkdownLearnings`` stays the reader, unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

SEPARATOR = "\n\n---\n\n"

DIGEST_HEADER = "<!-- derived digest — rebuild with: groundhog learnings rebuild -->"

ATTEMPT_LEARNINGS_FILENAME = "learnings.md"

DIGEST_SYSTEM_PROMPT = (
    "You maintain a distilled learnings digest for an iterative "
    "optimization run. Return only the digest entries, no preamble."
)

DIGEST_PROMPT_TEMPLATE = """\
Below are learning entries collected from an optimization run's attempt
ledger, newest first, tagged [attempt <id> | <direction family>].

Merge and compress them into at most {max_entries} strong entries:

1. Merge entries that make the same point; keep the most specific.
2. Directive phrasing grounded in evidence ("X hurt accuracy — avoid",
   "Y gained 3% — extend"), not vague summaries.
3. Flag contradictions rather than silently dropping either side.
4. Separate entries with a line containing only ---
5. Return only the entries, nothing else.

ENTRIES:
{entries}
"""


def record_attempt_learning(attempt_dir: Path | str, text: str) -> Optional[Path]:
    """Append a learning entry to the attempt's own ledger file.

    Call on the open workspace, before commit — the entry becomes part
    of the immutable record and travels with the attempt.
    """
    text = (text or "").strip()
    if not text:
        return None
    target = Path(attempt_dir) / ATTEMPT_LEARNINGS_FILENAME
    existing = target.read_text(encoding="utf-8") if target.exists() else ""
    if existing.strip():
        target.write_text(existing.rstrip() + SEPARATOR + text + "\n", encoding="utf-8")
    else:
        target.write_text(text + "\n", encoding="utf-8")
    return target


def attempt_learnings(attempt) -> List[str]:
    """Learning entries recorded in a committed attempt (the ledger read).

    Reads the strategy-recorded ``learnings.md`` at the attempt root and
    the agent scratchpad ``work/learnings.md`` (seed instructions
    stripped). Free-form agent notes come back as a single entry.
    """
    entries: List[str] = []
    for rel in (ATTEMPT_LEARNINGS_FILENAME, "work/" + ATTEMPT_LEARNINGS_FILENAME):
        text = attempt.read_file(rel)
        if not text or text.startswith("[binary file"):
            continue
        text = _strip_seed(text)
        entries.extend(e.strip() for e in text.split(SEPARATOR) if e.strip())
    return entries


def rebuild_digest(history, path: Path | str, max_entries: int = 50, llm=None) -> str:
    """Rebuild the derived digest at ``path`` from the per-attempt ledger.

    Walks committed attempts (done and failed — failures inform) newest
    first and collects their recorded learnings. Deterministic without
    ``llm``; with one, a single ``generate`` call merges the collected
    entries. Writes the digest (header included) to ``path`` and returns
    the written text.
    """
    collected = _collect(history)
    if llm is None:
        body = SEPARATOR.join(_tagged(item) for item in collected[:max_entries])
    else:
        body = _llm_digest(llm, collected, max_entries)
    text = DIGEST_HEADER + ("\n\n" + body if body else "") + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return text


def _collect(history) -> List[Tuple[str, str, str]]:
    """(attempt_id, family_label, entry) tuples: newest first, exact
    duplicates dropped, grouped by direction family (newest family first)."""
    from groundhog.utils.direction import (
        normalize_direction,
        read_direction_from_attempt,
    )

    attempts = [a for a in history.list(only_done=False)
                if a.status in ("done", "fail")]
    attempts.sort(key=lambda a: (a.created_at, _numeric(a.id)), reverse=True)

    seen = set()
    families: dict = {}
    order: list = []
    for a in attempts:
        entries = attempt_learnings(a)
        if not entries:
            continue
        direction = read_direction_from_attempt(a)
        key = normalize_direction(direction) if direction else None
        label = (direction.strip().splitlines()[0].strip()
                 if direction and direction.strip() else "no direction")
        for entry in entries:
            fingerprint = " ".join(entry.split())
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            if key not in families:
                families[key] = (label, [])
                order.append(key)
            families[key][1].append((a.id, label, entry))
    return [item for key in order for item in families[key][1]]


def _tagged(item: Tuple[str, str, str]) -> str:
    attempt_id, label, entry = item
    return f"[attempt {attempt_id} | {label}]\n{entry}"


def _llm_digest(llm, collected, max_entries: int) -> str:
    if not collected:
        return ""
    prompt = DIGEST_PROMPT_TEMPLATE.format(
        max_entries=max_entries,
        entries=SEPARATOR.join(_tagged(item) for item in collected),
    )
    response = llm.generate(prompt, system_prompt=DIGEST_SYSTEM_PROMPT)
    parts = [p.strip() for p in
             re.split(r"^\s*-{3,}\s*$", response.text.strip(), flags=re.MULTILINE)
             if p.strip()]
    return SEPARATOR.join(parts[:max_entries])


def _strip_seed(text: str) -> str:
    from groundhog.strategies.agent import LEARNINGS_SEED
    stripped = text.strip()
    seed = LEARNINGS_SEED.strip()
    if stripped.startswith(seed):
        return stripped[len(seed):].strip()
    return stripped


def _numeric(id_: str) -> int:
    try:
        return int(id_)
    except (TypeError, ValueError):
        return 0
