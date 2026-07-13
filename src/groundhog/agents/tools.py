"""Agent tool helpers.

Provides:
  - eval_to_dir(): run an eval stage and write results/artifacts to a directory
  - promote_best(): wrap an eval tool to snapshot the source on score improvement
  - build_default_agent_tools(): general-purpose utility tools for toolkit.agent_tools
  - build_prior_tools(): per-attempt tools for accessing prior attempt files
  - build_eval_tools(): wrap eval stages as agent tools (called by strategy, not optimizer)
  - build_learnings_tool(): wrap toolkit.learnings as an agent tool

``assemble_toolkit`` puts general utilities on toolkit.agent_tools (framework
defaults merged with the task.py ``agent_tools`` hook). Eval tools and
learnings are built by the strategy, which owns the workspace and can add
policies like promote-best.
"""

import copy
import json
import operator
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from groundhog.base.agent import agent_tool
from groundhog.utils.direction import normalize_direction, read_direction_from_attempt


def _format_eval_result(result):
    """Format a StageResult into a readable string for the agent."""
    lines = []

    # Score and key metrics
    if result.metrics:
        score = result.metrics.get("score")
        if score is not None:
            lines.append(f"Score: {score}")
        for key, val in result.metrics.items():
            if key != "score":
                lines.append(f"  {key}: {val}")

    # Errors
    if result.errors:
        lines.append("")
        lines.append("ERRORS (score = 0 if any present):")
        for key, msg in result.errors.items():
            lines.append(f"  {key}: {msg}")

    # Warnings
    if result.warnings:
        lines.append("")
        lines.append("WARNINGS:")
        for key, msg in result.warnings.items():
            lines.append(f"  {key}: {msg}")

    # Artifacts — list paths so the agent knows they exist
    if result.artifacts:
        lines.append("")
        lines.append("Artifacts:")
        for path in result.artifacts.values():
            lines.append(f"  {path}")

    return "\n".join(lines)


def eval_to_dir(stage, path, output_dir, prefix=""):
    """Run eval stage on a file, write results + artifacts to output_dir.

    Args:
        stage: EvalStage to run.
        path: Path to the .py file to evaluate.
        output_dir: Directory for results and artifacts.
        prefix: Prefix for artifact filenames (e.g. "validate_").

    Returns a formatted string with metrics, errors, warnings, and artifact paths.
    """
    code = Path(path).read_text(encoding="utf-8")
    result = stage.call(code)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Write results.json (prefixed)
    results_data = {"score": result.score, "metrics": result.metrics}
    if result.errors:
        results_data["errors"] = result.errors
    if result.warnings:
        results_data["warnings"] = result.warnings
    results_name = f"{prefix}results.json" if prefix else "results.json"
    (out / results_name).write_text(
        json.dumps(results_data, indent=2, default=str), encoding="utf-8")

    # Write artifacts with prefix, collect paths
    written = {}
    for name, content in result.artifacts.items():
        if name.startswith("_"):
            continue
        out_name = f"{prefix}{name}" if prefix else name
        dest = out / out_name
        if isinstance(content, bytes):
            dest.write_bytes(content)
        elif isinstance(content, str):
            dest.write_text(content, encoding="utf-8")
        else:
            if not out_name.endswith(".json"):
                out_name = f"{out_name}.json"
            dest = out / out_name
            dest.write_text(json.dumps(content, indent=2, default=str), encoding="utf-8")
        written[out_name] = str(dest)

    # Add results.json to written paths
    written[results_name] = str(out / results_name)

    # Format for agent
    out_result = copy.copy(result)
    out_result.artifacts = written
    return _format_eval_result(out_result)


def build_default_agent_tools(toolkit) -> list:
    """Build general-purpose framework utility tools for toolkit.agent_tools.

    Eval tools and learnings are built by the strategy (which owns the workspace
    and can add policies like promote-best); task-specific tools come from the
    task.py ``agent_tools(toolkit)`` hook via ``assemble_toolkit``.

    Tools built here bind the toolkit THEMSELVES: assemble_toolkit only
    binds task-hook tools (collect_task_tools), so a toolkit-injecting
    default left unbound would raise at invoke time.
    """
    from groundhog.base.agent import agent_tool

    tools = [agent_tool(check_gates), agent_tool(raise_insight)]
    tools.append(agent_tool(search_attempts))
    for t in tools:
        t.bind_toolkit(toolkit)
    return tools


def check_gates(toolkit) -> str:
    """Check the attempt in flight against the run's legitimacy gates —
    core direction present and unique (fresh attempts), inherited direction
    unmodified, solution not byte-identical to the parent. Reports exactly
    what the commit-time gate would find, so a failure surfaces mid-work
    instead of at commit. Read-only; changes nothing.
    """
    from groundhog.utils.gates import evaluate_gates

    handle = getattr(toolkit, "ws", None)
    if handle is None or not handle.is_set():
        return (
            "No attempt is in flight — check-gates reads the current "
            "workspace via toolkit.ws. From the CLI, point it at one with "
            "`groundhog tool run check-gates --attempt <id>`."
        )

    ws = handle.current
    history = getattr(toolkit, "history", None)
    parent, parent_known = _resolve_parent(ws, history)
    committed = getattr(ws, "attempt", None) is not None

    lines = []
    if not parent_known:
        lines.append(
            "note: this workspace's parent could not be determined — "
            "checking it as a FRESH attempt (direction gates apply)."
        )
    if committed:
        # A committed record already founded (or joined) its family — its
        # descendants share its direction, so re-running the duplicate
        # check against today's store would report the family it created
        # as a violation.
        lines.append(
            "note: committed record — the duplicate-direction gate is "
            "skipped (its family already exists in the store)."
        )

    # Exclude the attempt itself from the duplicate check under every
    # identity it may carry: a live workspace's display id, and — when the
    # handle points at a committed attempt (read-only view) — the record's
    # own id (display_id is the NAME there, which direction_exists ignores).
    self_ids = {
        getattr(ws, "display_id", None),
        getattr(getattr(ws, "attempt", None), "id", None),
    }
    self_ids.discard(None)

    violations = evaluate_gates(
        handle.path,
        parent,
        history=None if committed else history,
        exclude=self_ids,
    )
    if not violations:
        lines.append("All gates pass: this attempt would commit clean.")
        return "\n".join(lines)

    lines.append(f"{len(violations)} gate violation(s):")
    for v in violations:
        lines.append(f"- [{v.severity.upper()}] {v.gate}: {v.message}")
        if v.severity == "fail":
            lines.append(
                "    -> the standard finish would commit this attempt as FAILED"
            )
        else:
            lines.append(
                "    -> recorded in metadata; the commit itself stays done"
            )
    return "\n".join(lines)


INSIGHT_KINDS = ("insight", "tool-request", "blocker", "idea")
_INSIGHT_MAX_CHARS = 4000


def raise_insight(toolkit, kind: str = "insight", text: str = "") -> str:
    """Raise a note out of the sandbox to the humans running this optimization:
    a general observation (``insight``), a wish for a tool that would have
    helped (``tool-request``), something that blocked progress (``blocker``),
    or an idea worth trying later (``idea``). Appends a stamped entry to the
    run's ``insights.md`` and records it in the attempt log. Use it whenever
    you hit friction or notice something the framework's authors should know —
    it changes nothing about the solution.
    """
    text = (text or "").strip()
    if not text:
        return "raise-insight: nothing recorded (text was empty)."
    if len(text) > _INSIGHT_MAX_CHARS:
        text = text[:_INSIGHT_MAX_CHARS].rstrip() + "\n[truncated]"
    kind = (kind or "").strip().lower() or "insight"
    if kind not in INSIGHT_KINDS:
        kind = "insight"  # unknown kinds fold into a plain insight, never rejected

    stamp = datetime.now().isoformat(timespec="seconds")
    ws_id = _insight_workspace_id(toolkit)
    phase = _insight_phase(toolkit)

    header = f"## {stamp} | {kind}"
    if ws_id:
        header += f" | attempt {ws_id}"
    if phase:
        header += f" | phase {phase}"
    entry = f"{header}\n\n{text}"

    root = Path(getattr(toolkit, "path", ".") or ".")
    _append_insight(root / "insights.md", entry)
    _log_insight_event(toolkit, kind, text, ws_id, phase)

    where = f"attempt {ws_id}" if ws_id else "no open attempt"
    return f"raise-insight recorded ({kind}, {where}) -> insights.md"


def _insight_workspace_id(toolkit):
    handle = getattr(toolkit, "ws", None)
    if handle is None or not handle.is_set():
        return None
    try:
        return getattr(handle.current, "display_id", None)
    except Exception:
        return None


def _insight_phase(toolkit):
    logger = getattr(toolkit, "attempt_logger", None)
    if logger is None:
        return None
    try:
        events = logger.events()
    except Exception:
        return None
    for event in reversed(events):
        if getattr(event, "type", None) == "phase":
            return getattr(event, "phase", None) or None
    return None


def _append_insight(path, entry):
    """Learnings-style append: entries joined by a --- rule, one file per run.

    A true O(1) append — never reads the existing file, so one stray
    non-UTF8 byte can't permanently kill the channel, and a crash mid-write
    can't lose prior entries the way the old whole-file rewrite could.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nonempty = path.exists() and path.stat().st_size > 0
    with open(path, "a", encoding="utf-8") as f:
        if nonempty:
            f.write("\n---\n\n")
        f.write(entry.strip() + "\n")


def _log_insight_event(toolkit, kind, text, ws_id, phase):
    logger = getattr(toolkit, "attempt_logger", None)
    if logger is None or getattr(logger, "path", None) is None:
        return  # no open attempt — the insights.md entry stands on its own
    from groundhog.tools.attempt_logger import LogEvent
    data = {"kind": kind, "text": text}
    if ws_id:
        data["attempt"] = ws_id
    if phase:
        data["phase"] = phase
    try:
        logger.log(LogEvent(type="insight", data=data))
    except Exception:
        pass


def _resolve_parent(ws, history):
    """Best-effort parent Attempt for the gate check.

    Returns (parent, known): ``known`` is False only when the workspace
    carries no parent information at all (e.g. a foreign live workspace),
    in which case the caller should say it assumed fresh.
    """
    parent_id = getattr(ws, "parent", None)
    if parent_id is None:
        attempt = getattr(ws, "attempt", None)  # ReadOnlyWorkspaceView
        if attempt is not None:
            parent_id = getattr(attempt, "parent", None)
        elif not hasattr(ws, "parent"):
            return None, False  # no parent channel at all
    if parent_id in (None, "", "none"):
        return None, True  # genuinely fresh
    if history is None:
        return None, False
    parent = history.get(parent_id)
    if parent is None:
        return None, False
    return parent, True


def _merge_agent_tools(lower: list, higher: list, *, layer: str, log=None) -> list:
    """Merge two tool layers by name — higher wins, every shadow is logged.

    One rule for the whole pipeline (precedence strategy > task > default):
    ``assemble_toolkit`` merges task-hook tools over the framework defaults,
    and the agent strategy merges its per-attempt tools over
    ``toolkit.agent_tools``. Name-keyed, last-write-wins, idempotent.
    """
    by_name = {t.name: t for t in lower}
    for t in higher:
        if t.name in by_name:
            msg = (f"agent tool '{t.name}' from the {layer} layer shadows a "
                   f"lower-precedence tool of the same name")
            if log is not None and hasattr(log, "info"):
                log.info(msg)
            else:
                print(msg)
        by_name[t.name] = t
    return list(by_name.values())


def collect_task_tools(hook, toolkit) -> list:
    """Run a task.py ``agent_tools(toolkit)`` hook with loud validation.

    Failures surface at build time — a TypeError/ValueError here beats a
    swallowed ToolResult(success=False) at agent-invoke time.
    """
    if hook is None:
        return []
    from groundhog.base.agent import AgentTool
    tools = list(hook(toolkit) or [])
    seen = set()
    for t in tools:
        if not isinstance(t, AgentTool):
            raise TypeError(
                f"agent_tools(toolkit) must return AgentTool instances "
                f"(use groundhog.agent_tool(...)), got {t!r}"
            )
        if t.name in seen:
            raise ValueError(
                f"duplicate task tool name {t.name!r} returned by agent_tools()"
            )
        seen.add(t.name)
        # Derived tools with a `toolkit` first parameter get it here, once —
        # invisible to the agent's schema, supplied at invoke time.
        t.bind_toolkit(toolkit)
    return tools


def build_learnings_tool(toolkit):
    """Wrap toolkit.learnings as an agent tool. Returns None if unavailable."""
    if not hasattr(toolkit, 'learnings'):
        return None
    return agent_tool(
        name="get-learnings",
        description=(
            "Read accumulated learnings from previous optimization runs. "
            "Returns notes about what worked, what didn't, dead-ends, "
            "and key thresholds."
        ),
        func=toolkit.learnings.get,
        params={
            "last": {"type": "int", "default": 20,
                     "description": "Number of recent entries to return"},
            "random": {"type": "int", "default": 10,
                       "description": "Number of random older entries to include"},
        },
    )


def _eval_stage_tool(stage):
    """Build a plain eval tool for a single stage. Helper for build_eval_tools
    and promote_best so they share the parameter/description shape."""
    prefix = f"{stage.name}_"
    description = (
        f"{stage.description}. "
        f"Evaluates work/solution.py by default. "
        f"Returns score, metrics, errors/warnings, and artifact paths."
    )

    def fn(path="work/solution.py", s=stage, p=prefix):
        return eval_to_dir(s, path, str(Path(path).parent / "artifacts"), prefix=p)

    return agent_tool(
        name=stage.name,
        description=description,
        func=fn,
        params={
            "path": {
                "type": "path",
                "default": "work/solution.py",
                "description": "Path to .py file to evaluate (default: work/solution.py)",
            },
        },
    )


def promote_best(
    stage,
    dest_path,
    src_relative: str = "work/solution.py",
    compare: Callable[[float, float], bool] = operator.gt,
    parent_solution_path: Optional[Path] = None,
):
    """Build an eval tool for ``stage`` that ALSO snapshots the source file
    to ``dest_path`` whenever the eval's score beats this tool's session-local
    best.

    Use this in place of a plain build_eval_tools entry for the eval tier
    where promotion should happen. Typically: only the highest-fidelity
    stage, since cheaper stages tend to be noisy and can produce
    false-positive snapshots.

    Args:
        stage: ``EvalStage`` to wrap.
        dest_path: where to copy the source file on improvement (typically
            ``ws.path / "solution.py"``).
        src_relative: tool param default for the file to evaluate. Best is
            tracked relative to ``stage``'s scoring; the snapshot copies
            *exactly* the path the agent passed (so re-evals on different
            files don't promote stale work).
        compare: how to compare the new score against the session best.
            Defaults to strict ``>``; pass a different callable for
            multi-objective or "lower is better" cases.
        parent_solution_path: optional parent solution file. When supplied,
            byte-identical candidates are evaluated but not snapshotted.

    Returns an :class:`AgentTool` with the same name and parameter shape as
    a plain eval tool — drop-in replaceable.
    """
    best = [float("-inf")]
    dest = Path(dest_path)
    parent_bytes = None
    if parent_solution_path is not None:
        parent_path = Path(parent_solution_path)
        if parent_path.exists():
            parent_bytes = parent_path.read_bytes()
    prefix = f"{stage.name}_"

    description = (
        f"{stage.description}. "
        f"Evaluates work/solution.py by default. "
        f"Returns score, metrics, errors/warnings, and artifact paths. "
        f"Snapshots solution to {dest.name} on score improvement."
    )

    def fn(path=src_relative, s=stage, p=prefix, dest=dest, best=best, cmp=compare):
        result_str = eval_to_dir(s, path, str(Path(path).parent / "artifacts"), prefix=p)
        # Re-derive score for the comparison. Same shape as the previous
        # promote-best implementation; eval_to_dir already ran the stage,
        # so this is effectively cached for deterministic stages.
        code = Path(path).read_text(encoding="utf-8")
        score = s.score(s.call(code))
        if cmp(score, best[0]):
            src = Path(path)
            if parent_bytes is not None and src.exists() \
                    and src.read_bytes() == parent_bytes:
                return result_str + "\n\nNot promoted: solution.py is identical to the parent."
            best[0] = score
            if src.exists():
                shutil.copy2(str(src), str(dest))
        return result_str

    return agent_tool(
        name=stage.name,
        description=description,
        func=fn,
        params={
            "path": {
                "type": "path",
                "default": src_relative,
                "description": f"Path to .py file to evaluate (default: {src_relative})",
            },
        },
    )


def build_eval_tools(toolkit, ws_path, through=None, promote_dest=None,
                     parent_solution_path=None):
    """Wrap eval stages as agent tools. Called by the strategy per-attempt.

    Args:
        toolkit: the Toolkit (has ``.task`` with evaluator).
        ws_path: workspace Path — for artifact output.
        through: eval stage limit (None = all stages).
        promote_dest: optional Path. When set, the highest-fidelity stage is
            wrapped with :func:`promote_best` so its tool snapshots the
            source file to ``promote_dest`` on score improvement. Cheaper
            stages stay un-wrapped (their scores are noisy).
        parent_solution_path: optional parent solution file used to block
            identical promote-best snapshots.

    Returns list of agent tools, one per eval stage.
    """
    if not hasattr(toolkit, 'task'):
        return []

    effective_through = (
        through
        or getattr(toolkit, 'agent_through', None)
        or getattr(toolkit, 'through', None)
    )
    stages = toolkit.task.evaluator.eval_stages(
        toolkit.task.data, through=effective_through
    )

    tools = []
    final_idx = len(stages) - 1
    for i, stage in enumerate(stages):
        if i == final_idx and promote_dest is not None:
            tools.append(promote_best(
                stage,
                dest_path=promote_dest,
                parent_solution_path=parent_solution_path,
            ))
        else:
            tools.append(_eval_stage_tool(stage))
    return tools


def build_prior_tools(
    prior_attempt,
    history=None,
    scorer=None,
    max_distance: Optional[int] = None,
    scope: str = "lineage",
    exclude_direction: Optional[str] = None,
) -> list:
    """Build tools for browsing prior attempts.

    Three tools, progressive disclosure:
      ``get-priors``    — list metadata on N prior attempts (id, parent,
                          distance, score). Used to discover what's there.
      ``list-prior``    — given a prior id, list its files.
      ``get-prior-file``— given a prior id + file, read its contents.

    Args:
        prior_attempt: the immediate parent of the current workspace. If
            ``None``, no tools are built.
        history: an :class:`AttemptHistory` for walking lineage / siblings.
            Required for multi-prior reach. If ``None``, only the immediate
            parent is reachable.
        scorer: optional ``StageResult -> float`` for the score column in
            ``get-priors`` output. If ``None``, scores are reported as
            ``"?"``.
        max_distance: optional cap on tree distance (1 = parent only, 2 =
            grandparent, etc.). Strategy-level constraint that limits how
            far the agent can reach. ``None`` means no cap.
        scope: ``"lineage"`` (parent chain only, default), ``"tree"``
            (lineage + siblings of each ancestor), ``"family"`` (same core
            direction as the parent), or ``"all"``.
        exclude_direction: optional direction text to hide from ``family`` /
            ``all`` scopes, usually the parent's family when choosing
            cross-pollination inspiration.

    Returns empty list if ``prior_attempt`` is ``None``.
    """
    if prior_attempt is None:
        return []

    def _attempt_direction_key(attempt):
        text = read_direction_from_attempt(attempt)
        key = normalize_direction(text or "")
        return key or None

    parent_direction_key = _attempt_direction_key(prior_attempt)
    excluded_direction_key = normalize_direction(exclude_direction or "") or None

    def _allowed_by_direction(attempt):
        attempt_key = _attempt_direction_key(attempt)
        if excluded_direction_key is not None and attempt_key == excluded_direction_key:
            return False
        if scope == "family":
            return parent_direction_key is not None and attempt_key == parent_direction_key
        return True

    def _allowed_attempt_map():
        return {a.id: a for a, _ in _reachable_attempts()}

    def _resolve(attempt_id: str):
        """Resolve an agent-supplied id to an Attempt. Accepts ``"parent"``,
        an attempt id, an ``"attempt_<id>"`` string, or a unique id prefix
        (e.g. a short commit hash)."""
        allowed = _allowed_attempt_map()
        if history is None:
            if attempt_id in ("parent", prior_attempt.id,
                              f"attempt_{prior_attempt.id}"):
                return prior_attempt if prior_attempt.id in allowed else None
            return None
        if attempt_id == "parent":
            return prior_attempt if prior_attempt.id in allowed else None
        s = attempt_id
        if s.startswith("attempt_"):
            s = s[len("attempt_"):]
        if not s:
            return None
        if s in allowed:
            return allowed[s]
        # Unique-prefix match (git short hashes). Ambiguous prefix → no match.
        matches = [a for k, a in allowed.items() if k.startswith(s)]
        return matches[0] if len(matches) == 1 else None

    def _reachable_attempts():
        """Yield (attempt, distance) pairs the agent is allowed to see."""
        if history is None:
            if _allowed_by_direction(prior_attempt):
                yield prior_attempt, 1
            return

        if scope in ("family", "all"):
            distance = "family" if scope == "family" else "all"
            for a in history.list(only_done=True):
                if _allowed_by_direction(a):
                    yield a, distance
            return

        # Lineage = parent chain from the current workspace's parent up.
        chain = history.lineage(prior_attempt)
        # history.lineage(prior) returns [root, ..., prior].
        # Distance 1 = prior_attempt itself (the immediate parent).
        for offset, a in enumerate(reversed(chain), start=1):
            if max_distance is not None and offset > max_distance:
                break
            if _allowed_by_direction(a):
                yield a, offset

        if scope != "tree":
            return

        # Tree extension: siblings of each ancestor (same parent, not the
        # ancestor itself). Distance follows the ancestor.
        seen_ids = {a.id for a, _ in [(c, 0) for c in chain]}
        for offset, a in enumerate(reversed(chain), start=1):
            if max_distance is not None and offset > max_distance:
                break
            if a.parent is None:
                continue
            for sib in history.list(only_done=False):
                if sib.parent == a.parent and sib.id != a.id \
                        and sib.id not in seen_ids:
                    if _allowed_by_direction(sib):
                        yield sib, offset
                    seen_ids.add(sib.id)

    def _score_of(attempt):
        if scorer is None:
            return None
        try:
            stages = list(attempt.result.stages.values())
            if not stages:
                return None
            return scorer(stages[-1])
        except Exception:
            return None

    def _get_priors(n: int = 5):
        rows = []
        for a, distance in _reachable_attempts():
            score = _score_of(a)
            score_str = f"{score:.4f}" if isinstance(score, float) else "?"
            committed = a.status
            rows.append(
                f"attempt_{a.id}\tparent={a.parent}\tdistance={distance}"
                f"\tscore={score_str}\t{committed}"
            )
            if len(rows) >= n:
                break
        if not rows:
            return "(no priors reachable)"
        return "\n".join(["attempt\tparent\tdistance\tscore\tstatus", *rows])

    def _list_prior(attempt: str = "parent"):
        a = _resolve(attempt)
        if a is None:
            return f"(unknown attempt: {attempt!r})"
        return "\n".join(a.list_files())

    def _get_prior_file(attempt: str, file: str):
        a = _resolve(attempt)
        if a is None:
            return f"(unknown attempt: {attempt!r})"
        text = a.read_file(file)
        if text is None:
            return f"(file not found in attempt_{a.id}: {file!r})"
        return text

    return [
        agent_tool(
            name="get-priors",
            description=(
                "List metadata on prior attempts (attempt id, parent, "
                "distance, score, status). Use to discover what's available "
                f"before reading. Scope: {scope}."
            ),
            func=_get_priors,
            params={
                "n": {
                    "type": "int", "default": 5,
                    "description": "Max number of priors to list",
                },
            },
        ),
        agent_tool(
            name="list-prior",
            description=(
                "List files belonging to a prior attempt. "
                "Pass attempt='parent' (default) or a specific id like "
                "'attempt_42' or '42'."
            ),
            func=_list_prior,
            params={
                "attempt": {
                    "type": "str", "default": "parent",
                    "description": "Prior attempt id ('parent', '42', or 'attempt_42')",
                },
            },
        ),
        agent_tool(
            name="get-prior-file",
            description=(
                "Read a file from a prior attempt. "
                "Pass attempt='parent' or a specific id; file is relative "
                "to the attempt root (e.g. 'work/learnings.md'). "
                "Use list-prior first to see available files."
            ),
            func=_get_prior_file,
            params={
                "attempt": {
                    "type": "str",
                    "description": "Prior attempt id ('parent', '42', or 'attempt_42')",
                },
                "file": {
                    "type": "str",
                    "description": "File path relative to attempt root",
                },
            },
        ),
    ]


_SEARCH_SNIPPET_CHARS = 160
_SEARCH_OUTPUT_CHARS = 8000
_SEARCH_MAX_RESULTS = 100


def search_attempts(toolkit, query: str, scope: str = "all",
                    max_results: int = 20) -> str:
    """Search the run's knowledge base for a keyword or regex: run-root
    learnings.md and insights.md, plus every attempt's core direction,
    attempt log, and work/learnings.md. Returns attempt-stamped matching
    lines (attempt id, file, line, snippet). scope narrows the corpus:
    all | learnings | directions | logs | insights.
    """
    from groundhog.utils.semantic_index import SCOPES, iter_corpus

    if scope not in SCOPES:
        return f"unknown scope {scope!r} (use {'|'.join(SCOPES)})"
    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
    max_results = max(1, min(max_results, _SEARCH_MAX_RESULTS))

    root = getattr(toolkit, "path", None)
    history = getattr(toolkit, "history", None)
    docs = list(iter_corpus(root, history, scope=scope))

    rank = _semantic_rank(root, history, query)
    if rank is not None:
        docs.sort(key=lambda d: rank.get((d.attempt, d.file), len(rank)))

    hits = []
    for doc in docs:
        stamp = f"attempt_{doc.attempt}" if doc.attempt else "run-root"
        for lineno, line in enumerate(doc.text.splitlines(), start=1):
            if pattern.search(line):
                snippet = line.strip()[:_SEARCH_SNIPPET_CHARS]
                hits.append(f"{stamp} {doc.file}:{lineno}: {snippet}")
                if len(hits) >= max_results:
                    break
        if len(hits) >= max_results:
            break

    if not hits:
        return f"(no hits for {query!r} in scope {scope})"
    ranked = ", semantic rank" if rank is not None else ""
    out = "\n".join([f"{len(hits)} hit(s) for {query!r} [scope={scope}{ranked}]",
                     *hits])
    if len(out) > _SEARCH_OUTPUT_CHARS:
        out = out[:_SEARCH_OUTPUT_CHARS] + "\n(truncated)"
    return out


def _semantic_rank(root, history, query):
    """File ranking from the tier-2 semantic index — only when its cache
    already exists (building one is an explicit act, not a search side
    effect). None means lexical corpus order; any index failure falls back
    the same way, so the lexical scan is the always-works path."""
    if root is None:
        return None
    try:
        from groundhog.utils.semantic_index import SemanticIndex

        index = SemanticIndex(root, history)
        if not index.exists() or not index.load():
            return None
        order = {}
        for hit in index.search(query, k=_SEARCH_MAX_RESULTS):
            key = (hit.attempt, hit.file.split("#", 1)[0])
            order.setdefault(key, len(order))
        return order
    except Exception:
        return None
