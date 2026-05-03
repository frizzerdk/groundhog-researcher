"""Agent strategy — delegate work to an autonomous agent backend.

The strategy orchestrates multi-phase agent execution:
    explore (main work) → submit (finalize) → evaluate → fix (retry) → reflect (learnings)

The agent backend (Claude Code, Gemini CLI, etc.) handles the actual
subprocess, tool exposure, and event logging. The strategy owns phasing,
workspace lifecycle, tool filtering, and evaluation.

Tools are provided by toolkit.agent_tools (built by optimizer).
The strategy filters which tools are available per phase.
"""

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from groundhog.base.agent import AgentSpec
from groundhog.base.strategy import Strategy, StrategyConfig, param
from groundhog.tools.conversation_log import conversation_log


# --- Config ---

@dataclass
class AgentConfig(StrategyConfig):
    """Configuration for the Agent strategy."""
    timeout: Optional[int] = param(None, "Total timeout seconds (None=no timeout, use budget instead)")
    budget_usd: Optional[float] = param(1.0, "Max USD budget for explore phase (None=no limit)")
    max_retries: int = param(2, "Max fix attempts on evaluation failure")
    target: str = param("diverse", "Prior selection: 'best' or 'diverse'")
    phase_timeout: int = param(120, "Timeout per non-explore phase")
    phase_max_turns: int = param(5, "Max turns per non-explore phase")
    eval_through: Optional[str] = param(None, "Eval stage limit (None=all)")
    model: Optional[str] = param(None, "Override agent model")
    effort: Optional[str] = param(None, "Override agent effort/reasoning level")
    guidance: str = param("", "Additional guidance appended to agent prompt")
    tier: str = param("default", "Agent backend tier (default/high/budget)")
    core_direction: str = param(
        "",
        "Optional initial core_direction.md text. Canonical name for fresh "
        "direction families; initial_direction is kept as a legacy alias.",
    )
    initial_direction: str = param(
        "",
        "Optional initial core_direction.md text. If set, written to attempt "
        "root before the agent runs (used by PlanApproaches to seed a new "
        "fresh-direction family).",
    )
    force_prior_attempt: Optional[int] = param(
        None,
        "If set, pin this attempt number as the prior, bypassing the "
        "strategy's default selector entirely. Use whenever you want a "
        "specific parent rather than whatever the rating system picks — "
        "common cases: refining a known clean baseline, reproducing a "
        "previous result, deliberately exploring a non-best branch, or "
        "feeding a queue item targeted at a specific attempt.",
    )


# --- Permissions (depth-based, deepest/last wins) ---
#
# Class-level defaults live here so a subclass can opt into a tighter
# sandbox by reassigning ``permissions`` (or ``phase_overrides``) without
# touching module globals. Backend enforcement varies — see
# ``AgentStrategy.permissions`` docstring for the per-backend reality.

BASE_PERMISSIONS = [
    # Reads: workspace tree + sibling/prior tree only. Absolute paths
    # outside the attempt root are denied so an agent can't drift into
    # site-packages, the parent repo, or system files.
    ("allow", "Read(./**)"),
    ("allow", "Read(../**)"),
    ("deny",  "Read(*)"),
    # Writes: only inside work/. Everything else (attempt root, outside,
    # system) is denied.
    ("deny",  "Write(*)"),
    ("allow", "Write(work/*)"),
    ("allow", "Edit(work/*)"),
]

PHASE_OVERRIDES = {
    "explore": [],
    "submit":  [],
    "fix": [],
    "reflect": [],
}


# --- Tool filtering per phase ---

# None = all tools, [] = no tools, list of names = whitelist
PHASE_TOOLS = {
    "explore": None,
    "submit": [],
    "fix": None,
    "reflect": [],
}


# --- Prompt templates ---

LEARNINGS_SEED = """\
# Learnings

Notes from this attempt. Keep high signal-to-noise.
Only add entries that would save time or prevent repeated mistakes.

Good: confirmed dead-ends, key thresholds, techniques with measurable gains.
Bad: speculative ideas, verbose explanations, anything obvious from the code.

Prior attempts' notes are NOT auto-copied here. If you want context from
earlier work, use the get-priors / list-prior / get-prior-file tools to
read them on demand.
"""


# Human-readable sandbox contract injected into every phase prompt.
# Mirrors the rules encoded in BASE_PERMISSIONS. Some backends enforce
# these at the OS or CLI level; others rely on the prompt as the only
# signal. Either way, an explicit statement here keeps the agent from
# wasting turns probing the boundary and documents the contract for the
# transcript reader.
SANDBOX_RULES = """\
## Sandbox & rules

You are running in a sandboxed workspace. The boundaries below are
enforced by the host on backends that support it; on the rest, the
host audits transcripts. **Do not under any circumstances attempt to circumvent them.**

**You may:**
- Read any file inside the attempt directory tree, including
  `work/`, the attempt root, and parent dirs (`../`) for sibling
  attempts and shared materials.
- Write or edit files **inside `work/` only**. This is the area that
  will be submitted at the end of the session.
- Run the registered Groundhog tools (e.g., `evaluate`,
  `get-learnings`, `get-priors`) and your CLI's own Read/Edit tools.

**You may NOT:**
- Read files outside the attempt directory — system paths
  (`C:\\Windows\\...`, `/etc/...`), the parent repository's source,
  user dotfiles, unrelated projects, etc.
- Write or edit anywhere outside `work/` — including the attempt
  root (no `solution.py` rewrites at the root; that's auto-submitted
  from `work/solution.py`), sibling attempts, parent directories, or
  any system path.
- Use shell commands or scripting tricks to circumvent the above —
  no `cp`/`copy` to exfiltrate data, no symlinks pointing outside
  `work/`, no embedding absolute paths in code that gets executed
  later, no spawning sub-processes that bypass the constraints.

If a tool call is blocked, that is the intended behavior. Note it and
move on — do not retry, do not work around. If you believe a rule
prevents legitimate work for this task, say so explicitly in your
response so a human can adjust the policy."""

EXPLORE_PROMPT = """\
{session_header}

You are an expert code optimizer. You work iteratively using tools.

## Task

{task_context}
{scoring_context}
{approach_context}
## Workflow

You are in the **exploration phase**. When your budget runs out you will
automatically move to a submission phase.

1. Read work/solution.py — the current best solution
2. Run `get-learnings` for accumulated knowledge from previous runs
3. Run `{eval_command}` to establish your baseline score
4. **Analyze**: Study the metrics and artifacts — what is the bottleneck?
5. **Plan**: Decide what single change to try and what you expect it to do
6. **Implement**: Edit work/solution.py with your change
7. **Evaluate**: Run `{eval_command}` to measure the impact
8. **Reflect**: Did it help? Why or why not? Iterate from step 4

## Key Rules

- Edit work/solution.py directly — it will be submitted automatically
- Run `{eval_command}` to evaluate (reads work/solution.py by default)
- work/ is your writable area for solution, experiments, and artifacts
- Preserve core_direction.md as the algorithmic backbone when it exists
- Do not fall back to the parent solution; byte-identical children are non-promotable
- Focus on understanding before changing — blind edits waste iterations
{budget_info}{guidance}

{sandbox_rules}

## Files

{file_listing}"""

SUBMIT_PROMPT = """\
Your exploration phase is over. Your work/solution.py will be submitted.
No action needed — it is copied automatically."""

FIX_PROMPT = """\
Your work/solution.py failed evaluation with this error:

{error}

Fix the issue in work/solution.py and run `{eval_command}` to verify.

{sandbox_rules}"""

REFLECT_PROMPT = """\
Update work/learnings.md with what you learned this session:
- What approaches did you try and what scores did they get?
- What worked well? What didn't?
- What dead ends should future attempts avoid?

Do not modify work/solution.py."""

# Per-request explore prompt — agent does everything in one call
EXPLORE_PROMPT_FULL = """\
{session_header}

You are an expert code optimizer. You work iteratively using tools.

## Task

{task_context}
{scoring_context}
{approach_context}
## Workflow

You have one session to improve the solution.

1. Read work/solution.py — the current best solution
2. Run `get-learnings` for accumulated knowledge from previous runs
3. Run `{eval_command}` to establish your baseline score
4. **Analyze**: Study the metrics and artifacts — what is the bottleneck?
5. **Plan**: Decide what single change to try and what you expect it to do
6. **Implement**: Edit work/solution.py with your change
7. **Evaluate**: Run `{eval_command}` to measure the impact
8. **Reflect**: Did it help? Why or why not? Iterate from step 4
9. When done, update work/learnings.md with what you learned

## Key Rules

- Edit work/solution.py directly — it will be submitted automatically
- Run `{eval_command}` to evaluate (reads work/solution.py by default)
- work/ is your writable area for solution, experiments, and artifacts
- Preserve core_direction.md as the algorithmic backbone when it exists
- Do not fall back to the parent solution; byte-identical children are non-promotable
- Focus on understanding before changing — blind edits waste iterations
{budget_info}{guidance}

{sandbox_rules}

## Files

{file_listing}"""


def _normalize_agent_event(event: dict) -> Optional[tuple]:
    """Map a raw backend event to ``(source, kind, summary)`` for AttemptLog.

    Recognised shapes (all five backends emit different keys):

    - copilot:    ``{"type": "tool.execution_start", "data": {"toolName": ...}}``
                  ``{"type": "assistant.message", "data": {"content": ...}}``
    - claude_code:``{"type": "assistant", "message": {"content": [tool_use blocks]}}``
    - opencode:   ``{"type": "tool_use", "part": {"tool": ..., "state": {...}}}``
    - gemini_cli: ``{"type": "tool_use", "tool_name": ..., "parameters": {...}}``
    - codex_cli:  ``{"type": "item.completed", "item": {"type": "command_execution"}}``

    Returns ``None`` for events that aren't worth surfacing (deltas, init
    pings, step boundaries). Anything more nuanced is the renderer's job.
    """
    et = event.get("type", "")

    def _short_path(p: str) -> str:
        """Long absolute paths bury the actual filename in the live tail.
        Reduce ``C:\\repo\\.../work/solution.py`` to ``solution.py``.
        Bare relative names pass through unchanged."""
        if not isinstance(p, str) or not p:
            return ""
        return p.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]

    # ---- copilot ----
    if et == "tool.execution_start":
        data = event.get("data", {}) or {}
        name = data.get("toolName", "")
        if name in ("report_intent", "task_complete"):
            return None
        args = data.get("arguments", {}) or {}
        path_detail = args.get("path") or ""
        if path_detail:
            detail = _short_path(path_detail)
        else:
            cmd = args.get("command", "")
            detail = cmd.split("\n")[0][:60] if isinstance(cmd, str) else ""
        summary = f"{name} {detail}".strip() if detail else name
        return ("agent", "tool_call", summary)

    if et == "assistant.message":
        content = (event.get("data", {}) or {}).get("content", "")
        first = content.strip().split("\n")[0] if content else ""
        return ("agent", "thinking", first) if first else None

    # ---- claude_code (stream-json assistant blocks) ----
    if et == "assistant":
        msg = event.get("message", {}) or {}
        for block in msg.get("content", []) if isinstance(msg.get("content"), list) else []:
            bt = block.get("type")
            if bt == "tool_use":
                name = block.get("name", "tool")
                inp = block.get("input", {}) or {}
                path_detail = inp.get("file_path") or inp.get("path") or ""
                if path_detail:
                    detail = _short_path(path_detail)
                else:
                    cmd = inp.get("command", "")
                    detail = cmd.split("\n")[0][:60] if isinstance(cmd, str) else ""
                summary = f"{name} {detail}".strip() if detail else name
                kind = "edit" if name in ("Edit", "Write") else "tool_call"
                return ("agent", kind, summary)
            if bt == "thinking":
                text = block.get("thinking", "") or ""
                first = text.strip().split("\n")[0]
                if first:
                    return ("agent", "thinking", first[:120])

    # ---- opencode (part-shape) ----
    if et == "tool_use":
        # opencode shape
        part = event.get("part", {}) or {}
        if part.get("type") == "tool":
            tool = part.get("tool", "tool")
            state = part.get("state", {}) or {}
            inp = state.get("input", {}) or {}
            path_detail = inp.get("filePath") or inp.get("path") or ""
            if path_detail:
                detail = _short_path(path_detail)
            else:
                cmd = inp.get("command", "")
                detail = cmd.split("\n")[0][:60] if isinstance(cmd, str) else ""
            summary = f"{tool} {detail}".strip() if detail else tool
            return ("agent", "tool_call", summary)
        # gemini_cli shape (tool_name + parameters at top level)
        tool = event.get("tool_name", "")
        if tool:
            params = event.get("parameters", {}) or {}
            path_detail = params.get("file_path") or ""
            if path_detail:
                detail = _short_path(path_detail)
            else:
                cmd = params.get("command", "")
                detail = cmd.split("\n")[0][:60] if isinstance(cmd, str) else ""
            kind = "edit" if tool in ("write_file", "replace") else "tool_call"
            summary = f"{tool} {detail}".strip() if detail else tool
            return ("agent", kind, summary)

    # ---- codex_cli (item.* events) ----
    if et == "item.completed":
        item = event.get("item", {}) or {}
        itype = item.get("type")
        if itype == "command_execution":
            cmd = (item.get("command", "") or "").split("\n")[0][:80]
            return ("agent", "tool_call", cmd)
        if itype == "agent_message":
            text = (item.get("text", "") or "").strip().split("\n")[0]
            if text:
                return ("agent", "thinking", text[:120])

    return None


class AgentStrategy(Strategy):
    """Delegate optimization work to an autonomous agent.

    Composed method pattern:
        init → select prior → workspace → prepare → explore → submit
        → evaluate → fix loop → reflect → log → commit

    Sandboxing
    ----------
    ``permissions`` and ``phase_overrides`` are overridable rule sets that
    populate ``AgentSpec.allowed_tools`` / ``AgentSpec.denied_tools`` per
    phase. Override them at the subclass or instance level for tighter
    sandboxes — e.g. workspace-only reads:

    .. code-block:: python

        class TightAgent(AgentStrategy):
            permissions = [
                ("allow", "Read(./**)"),       # workspace-relative
                ("allow", "Read(../**)"),      # priors / shared tools
                ("deny",  "Read(*)"),          # everything else
                ("deny",  "Write(*)"),
                ("allow", "Write(work/*)"),
                ("allow", "Edit(work/*)"),
            ]

    Per-backend enforcement reality (the rule list is built the same way
    for all of them; what each backend does with it differs):

    - ``ClaudeCodeAgentBackend``: full enforcement via ``--allowedTools``
      / ``--disallowedTools`` (deny-broad-allow-narrow works).
    - ``GeminiCliAgentBackend``: deny rules are injected into the prompt
      as a "you must not use" instruction (advisory, not enforced).
      ``allowed_tools`` is not consumed.
    - ``CopilotAgentBackend``: tool-name allow via ``--available-tools``;
      path-specific denies via ``--deny-tool``; **blanket** denies like
      ``Read(*)`` / ``Write(*)`` are silently dropped because they would
      override copilot's required ``--allow-all-tools`` flag.
    - ``CodexCliAgentBackend``: deny rules injected into the prompt as
      advisory text (like Gemini). ``allowed_tools`` is not consumed.
      The hard floor is the OS-level ``--sandbox workspace-write``
      (writes confined to attempt-root tree, network blocked) — reads
      remain unrestricted at the sandbox level regardless of what's in
      ``denied_tools``.
    - ``OpenCodeAgentBackend``: maps read/edit/bash rules into a generated
      OpenCode permission config. There is no OS sandbox, so this is
      stronger than prompt-only adapters but weaker than Codex's filesystem
      floor.
    """

    Config = AgentConfig

    # Default permission rules. Override at class or instance level for a
    # tighter sandbox; see class docstring.
    permissions = BASE_PERMISSIONS
    phase_overrides = PHASE_OVERRIDES

    def _resolve_permissions(self, phase):
        """Merge class-level base + phase overrides into allow/deny lists."""
        rules = list(self.permissions) + self.phase_overrides.get(phase, [])
        return (
            [r for a, r in rules if a == "allow"],
            [r for a, r in rules if a == "deny"],
        )

    def __call__(self, toolkit, config=None):
        self._init(toolkit, config)

        if not hasattr(toolkit, 'agent'):
            return {"skipped": "no agent backend available"}

        prior = self._select_prior(toolkit)
        ws = self._start_workspace(toolkit, prior)

        # Open the per-attempt log box so the user sees the run starting
        # before the agent backend produces its first event. Optimizer
        # finalizes with attempt_done/attempt_failed in _log_attempt.
        attempt_log = getattr(toolkit, "attempt_log", None)
        if attempt_log is not None:
            attempt_log.attempt_start(
                num=ws.number,
                prior=prior.number if prior else None,
                queue_label=getattr(toolkit, "_current_queue_label", "") or "",
                budget_total=self.cfg.budget_usd or 0.0,
            )

        try:
            self._prepare_workspace(toolkit, ws, prior)

            backend = toolkit.agent.get(self.cfg.tier)
            if backend.cost_model == "per_request":
                return self._run_per_request(toolkit, ws, prior)
            else:
                return self._run_per_token(toolkit, ws, prior)

        except Exception as e:
            ws.abort()
            return {"skipped": f"agent error: {e}"}

    # --- Execution paths ---

    def _run_per_token(self, toolkit, ws, prior):
        """Multi-phase: explore → submit → evaluate → fix → reflect."""
        session_id = self._explore(toolkit, ws, prior)
        self._update_attempt_log(toolkit, phase="submit")
        self._submit_best(toolkit, ws)
        result = self._evaluate(toolkit, ws)
        result = self._fix_loop(toolkit, ws, session_id, result)
        self._reflect(toolkit, ws, session_id)
        self._log_conversation(ws)

        self._finalize(ws, result, prior)
        attempt = ws.commit(success=result.completed)
        return self._build_log(attempt, prior, result, toolkit)

    def _run_per_request(self, toolkit, ws, prior):
        """Single explore call — agent edits work/solution.py directly."""
        session_id = self._explore_full(toolkit, ws, prior)
        self._log_conversation(ws)

        # Copy work/solution.py to attempt root for evaluation
        self._submit_best(toolkit, ws)

        # Evaluate
        result = self._evaluate(toolkit, ws)

        # Fix if needed
        if not result.completed:
            result = self._fix_loop(toolkit, ws, session_id, result)

        # Promote local learnings to task-level
        self._collect_learnings(toolkit, ws)

        self._finalize(ws, result, prior)
        attempt = ws.commit(success=result.completed)
        return self._build_log(attempt, prior, result, toolkit)

    # --- Init ---

    def _init(self, toolkit, config):
        from groundhog.tools.log import StrategyLog
        self.cfg = self._resolve_config(config)
        self.through = self.cfg.eval_through or getattr(toolkit, 'through', None)
        self.log = toolkit.log if hasattr(toolkit, 'log') else StrategyLog()
        self.cost = 0.0
        self._event_line_len = 0
        self._event_count = 0
        # Stash for use during finalize / subclass hooks (e.g. FreshAgent
        # generates its core direction post-session and needs LLM access).
        self._toolkit = toolkit

    def _on_event(self, event):
        """Live progress callback fired by every agent backend.

        Forwards to ``toolkit.attempt_log`` (the unified per-attempt event
        stream). Each backend's raw event shape is normalized via its own
        ``normalize_event`` helper — see the per-backend module. Backends
        without a normalizer (or events that aren't worth surfacing) are
        silently ignored.
        """
        attempt_log = getattr(self._toolkit, "attempt_log", None) if self._toolkit else None
        if attempt_log is None:
            return
        ev = _normalize_agent_event(event)
        if ev is None:
            return
        source, kind, summary = ev
        attempt_log.event(source=source, kind=kind, summary=summary)
        self._event_count += 1

    def _clear_event_line(self):
        """Legacy hook — the AttemptLog renderer manages its own region.
        Kept as a no-op so subclasses calling it don't break."""
        self._event_line_len = 0

    def _update_attempt_log(self, toolkit, **kwargs):
        """Push a partial state update to the per-attempt log (no-op when
        the optimizer didn't install one)."""
        attempt_log = getattr(toolkit, "attempt_log", None)
        if attempt_log is not None:
            attempt_log.update(**kwargs)

    # --- Selection ---

    def _select_prior(self, toolkit):
        # Explicit pinning via config. Bypasses any auto-selection so queue
        # items (or task code) can target a specific attempt for any
        # reason — known clean baseline, reproducing a result, exploring
        # a non-best branch, etc. Returns None if the named attempt
        # doesn't exist — caller treats that as "no prior available".
        forced = getattr(self.cfg, "force_prior_attempt", None)
        if forced is not None:
            return toolkit.history.get(int(forced))
        if self.cfg.target == "best":
            stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=self.through)
            return toolkit.history.best(stages[-1].score)
        if hasattr(toolkit, 'get_prior'):
            return toolkit.get_prior(toolkit)
        stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=self.through)
        return toolkit.history.best(stages[-1].score)

    # --- Workspace ---

    def _start_workspace(self, toolkit, prior):
        parent = prior.number if prior else None
        return toolkit.history.workspace(parent=parent)

    def _prepare_workspace(self, toolkit, ws, prior):
        # Strategy-managed files in attempt root
        (ws.path / "TASK_CONTEXT.md").write_text(toolkit.task.context.get(), encoding="utf-8")

        # Inherit core direction from prior (read-only for the agent during
        # the session; re-enforced at commit so an agent can't fork the family
        # by rewriting it).
        if prior is not None and hasattr(prior, 'path'):
            from groundhog.utils.direction import inherit_direction
            inherit_direction(prior.path, ws.path)

        # Seed work/solution.py from prior
        if prior:
            (ws.path / "work" / "solution.py").write_text(prior.code, encoding="utf-8")

        # Seed work/learnings.md as an empty notes scratchpad. Prior learnings
        # are NOT copied forward — agents read them on demand via the
        # get-priors / list-prior / get-prior-file tools.
        if not (ws.path / "work" / "learnings.md").exists():
            (ws.path / "work" / "learnings.md").write_text(LEARNINGS_SEED, encoding="utf-8")

    # --- Tool building ---

    def _get_tools(self, toolkit, ws, prior, phase="explore"):
        """Build tools for the agent from toolkit capabilities.

        Assembles: utility tools + learnings + eval tools + prior file access.
        Eval tools are wrapped here (not at optimizer init) so the strategy
        can add policies like promote-best on the final stage.
        """
        allowed = PHASE_TOOLS.get(phase)
        if allowed is not None and not allowed:
            return []

        # General utilities from toolkit (plotting, KB, etc.)
        tools = list(getattr(toolkit, 'agent_tools', []))

        # Learnings tool
        from groundhog.agents.tools import build_learnings_tool
        learnings_tool = build_learnings_tool(toolkit)
        if learnings_tool is not None:
            tools.append(learnings_tool)

        # Eval tools. Promote-best only during explore: it protects against the
        # agent regressing after a good score. In fix, the current root is known
        # to have failed evaluation, so any fix is preferable — we copy work/ to
        # root unconditionally after the fix agent runs.
        if phase in ("explore", "fix"):
            from groundhog.agents.tools import build_eval_tools, build_prior_tools

            promote_dest = (ws.path / "solution.py") if phase == "explore" else None
            parent_solution_path = (
                Path(prior.path) / "solution.py"
                if prior is not None and hasattr(prior, "path")
                else None
            )
            tools += build_eval_tools(
                toolkit,
                ws.path,
                through=self.through,
                promote_dest=promote_dest,
                parent_solution_path=parent_solution_path,
            )

            if prior is not None:
                # Score column comes from the same scorer the optimizer uses.
                stages = toolkit.task.evaluator.eval_stages(
                    toolkit.task.data, through=self.through
                )
                final_scorer = stages[-1].score if stages else None
                tools.extend(
                    build_prior_tools(
                        prior,
                        history=getattr(toolkit, "history", None),
                        scorer=final_scorer,
                        **self._prior_tool_options(toolkit, ws, prior),
                    )
                )

        return tools

    def _prior_tool_options(self, toolkit, ws, prior):
        """Hook for subclasses that want wider/narrower prior visibility."""
        return {}

    # --- Helpers ---

    def _get_eval_command(self, toolkit):
        """Get the first eval stage name for prompt references."""
        through = getattr(toolkit, 'agent_through', None) or self.through
        stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=through)
        return stages[0].name if stages else "evaluate"

    def _build_file_listing(self, ws):
        """List workspace files for agent orientation."""
        files = sorted(
            str(f.relative_to(ws.path)).replace("\\", "/")
            for f in ws.path.rglob("*") if f.is_file()
        )
        return "\n".join(f"  {f}" for f in files) if files else "  (empty)"

    def _build_session_header(self, toolkit, ws, prior):
        """Build session header with prior score and key metrics."""
        if not prior:
            return f"[{toolkit.task.name} #{ws.number}] fresh start"

        prior_score = self._score_result(prior.result, toolkit)
        header = f"[{toolkit.task.name} #{ws.number}] prior=#{prior.number} score={prior_score:.4f}"

        # Append key metrics from the prior's last stage
        prior_metrics = self._get_prior_metrics(prior, toolkit)
        if prior_metrics:
            parts = []
            for k, v in prior_metrics.items():
                if k == "score":
                    continue
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
            if parts:
                header += "\n  " + " ".join(parts)

        return header

    def _get_prior_metrics(self, prior, toolkit):
        """Get metrics dict from the prior's last completed stage."""
        if not prior or not prior.result.completed:
            return {}
        stages = list(prior.result.stages.values())
        return stages[-1].metrics if stages else {}

    def _build_scoring_context(self, toolkit):
        """Build optional scoring section from task context."""
        scoring = toolkit.task.context.get_scoring()
        if scoring:
            return f"\n## Scoring\n\n{scoring}\n"
        return ""

    def _build_approach_context(self, ws):
        """Build the optional 'core direction' section for the explore prompt."""
        from groundhog.utils.direction import read_direction
        text = read_direction(ws.path)
        if text and text.strip():
            return (
                "\n## Core direction (preserve this — the algorithmic invariant "
                "of this family)\n\n"
                f"{text.strip()}\n\n"
                "You may change implementation details, parameters, helpers, "
                "preprocessing, and optimizations. Do not replace the core "
                "direction unless this is a fresh-direction strategy.\n"
            )
        return ""

    # --- Phases ---

    def _build_prompt_vars(self, toolkit, ws, prior):
        """Build common prompt template variables."""
        session_header = self._build_session_header(toolkit, ws, prior)
        eval_command = self._get_eval_command(toolkit)
        scoring_context = self._build_scoring_context(toolkit)
        approach_context = self._build_approach_context(ws)

        budget_info = ""
        if self.cfg.budget_usd:
            budget_info = f"\n- Budget: ${self.cfg.budget_usd:.2f} for this exploration phase."
        if self.cfg.timeout:
            minutes = self.cfg.timeout // 60
            budget_info += f"\n- Time limit: ~{minutes} minutes."
        guidance = f"\n\n## Additional Guidance\n{self.cfg.guidance}" if self.cfg.guidance else ""

        return dict(
            session_header=session_header,
            task_context=toolkit.task.context.get(),
            eval_command=eval_command,
            scoring_context=scoring_context,
            approach_context=approach_context,
            budget_info=budget_info,
            guidance=guidance,
            sandbox_rules=SANDBOX_RULES,
            file_listing=self._build_file_listing(ws),
        )

    def _explore(self, toolkit, ws, prior):
        """Main work phase — agent works in work/."""
        goal = EXPLORE_PROMPT.format(**self._build_prompt_vars(toolkit, ws, prior))
        self._update_attempt_log(toolkit, phase="explore")

        tools = self._get_tools(toolkit, ws, prior, phase="explore")
        allow, deny = self._resolve_permissions("explore")

        spec = AgentSpec(
            goal=goal,
            workspace_path=ws.path,
            tools=tools,
            model=self.cfg.model,
            effort=self.cfg.effort,
            allowed_tools=allow,
            denied_tools=deny,
            timeout=self.cfg.timeout,
            budget_usd=self.cfg.budget_usd,
            on_event=self._on_event,
        )
        result = toolkit.agent.get(self.cfg.tier).run(spec)
        self.cost += result.cost
        self._update_attempt_log(
            toolkit, budget_used=self.cost, turns=result.turns or 0,
        )

        return result.session_id

    def _explore_full(self, toolkit, ws, prior):
        """Per-request explore — agent does everything in one call."""
        goal = EXPLORE_PROMPT_FULL.format(**self._build_prompt_vars(toolkit, ws, prior))
        self._update_attempt_log(toolkit, phase="explore")

        tools = self._get_tools(toolkit, ws, prior, phase="explore")
        allow, deny = self._resolve_permissions("explore")

        spec = AgentSpec(
            goal=goal,
            workspace_path=ws.path,
            tools=tools,
            model=self.cfg.model,
            effort=self.cfg.effort,
            allowed_tools=allow,
            denied_tools=deny,
            timeout=self.cfg.timeout,
            on_event=self._on_event,
        )
        # Per-request explore historically used "high"; preserve that as the
        # default while allowing tier= to override.
        per_request_tier = self.cfg.tier if self.cfg.tier != "default" else "high"
        result = toolkit.agent.get(per_request_tier).run(spec)
        self.cost += result.cost
        self._update_attempt_log(
            toolkit, budget_used=self.cost, turns=result.turns or 0,
        )

        return result.session_id

    def _submit_best(self, toolkit, ws):
        """Fallback: copy work/solution.py to attempt root IF root doesn't exist.

        The promote-best callback on eval tools is the primary mechanism — it
        snapshots the best-scoring version during the session. This function
        only runs for sessions where the agent never successfully ran the
        final eval stage (no promote happened), in which case we fall back to
        whatever is in work/.
        """
        dst = ws.path / "solution.py"
        if dst.exists():
            # Promote-best already handled this — don't overwrite with a
            # potentially regressed final version.
            return
        src = ws.path / "work" / "solution.py"
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    def _collect_learnings(self, toolkit, ws):
        """Promote local learnings to task-level store."""
        learnings_path = ws.path / "work" / "learnings.md"
        if not learnings_path.exists() or not hasattr(toolkit, 'learnings'):
            return
        text = learnings_path.read_text(encoding="utf-8").strip()
        if text and text != LEARNINGS_SEED.strip():
            toolkit.learnings.add(text)

    def _evaluate(self, toolkit, ws):
        """Run the task evaluator on solution.py."""
        self._update_attempt_log(toolkit, phase="evaluate")
        result = toolkit.task.evaluate(ws.path, through=self.through)
        return result

    def _fix_loop(self, toolkit, ws, session_id, result, prior=None):
        """Retry on evaluation failure.

        ``prior`` is threaded through so the fix agent gets the full
        prior-attempt toolset (``get-priors`` / ``list-prior`` /
        ``get-prior-file``). Without it, fix-phase wrappers omit those
        tools, and any agent that tries ``Bash(get-prior-file ...)`` in
        fix mode hits ``command not found`` — defeating the most
        direct path for "diff against parent to debug a regression."
        """
        eval_command = self._get_eval_command(toolkit)
        for retry in range(self.cfg.max_retries):
            if result.completed:
                return result

            error_stage = result.stages[result.failed_stage]
            error_text = f"Stage '{result.failed_stage}': {error_stage.errors}"

            self._update_attempt_log(toolkit, phase=f"fix {retry + 1}")
            allow, deny = self._resolve_permissions("fix")
            goal = FIX_PROMPT.format(
                error=error_text,
                eval_command=eval_command,
                sandbox_rules=SANDBOX_RULES,
            )
            spec = AgentSpec(
                goal=goal,
                workspace_path=ws.path,
                tools=self._get_tools(toolkit, ws, prior=prior, phase="fix"),
                model=self.cfg.model,
                effort=self.cfg.effort,
                allowed_tools=allow,
                denied_tools=deny,
                timeout=self.cfg.phase_timeout,
                session_id=session_id,
            )
            fix_result = toolkit.agent.get(self.cfg.tier).run(spec)
            self.cost += fix_result.cost

            # Root is known-failed here; fix replaces it unconditionally so
            # the re-eval sees the fix even if the agent didn't run the eval
            # tool (promote-best is intentionally disabled for fix phase).
            work_solution = ws.path / "work" / "solution.py"
            if work_solution.exists():
                (ws.path / "solution.py").write_text(
                    work_solution.read_text(encoding="utf-8"), encoding="utf-8")

            result = toolkit.task.evaluate(ws.path, through=self.through)

        return result

    def _reflect(self, toolkit, ws, session_id):
        """Agent writes learnings to work/learnings.md."""
        self._update_attempt_log(toolkit, phase="reflect")
        allow, deny = self._resolve_permissions("reflect")
        spec = AgentSpec(
            goal=REFLECT_PROMPT,
            workspace_path=ws.path,
            tools=self._get_tools(toolkit, ws, prior=None, phase="reflect"),
            model=self.cfg.model,
            effort=self.cfg.effort,
            allowed_tools=allow,
            denied_tools=deny,
            timeout=self.cfg.phase_timeout,
            session_id=session_id,
        )
        result = toolkit.agent.get(self.cfg.tier).run(spec)
        self.cost += result.cost

        # Promote local learnings to task-level store
        self._collect_learnings(toolkit, ws)

    # --- Finalization ---

    def _finalize(self, ws, result, prior):
        """Write result.json. solution.py at root is maintained throughout the
        run by promote-best (explore) and the fix-loop copy — don't overwrite
        it here or we'd regress to the agent's last edit.

        Direction handling:
            - prior is None (fresh-direction strategy): promote any
              ``work/core_direction.md`` the agent wrote to attempt root.
            - prior is not None (inheritance strategy): re-copy parent's
              ``core_direction.md`` to attempt root, overwriting anything
              the agent might have written. This is the soft-gate that
              keeps families from forking mid-session.

        Solution-duplicate guard: if the committed ``solution.py`` is
        byte-identical to the parent's, mark the attempt as non-promotable
        in metadata so selectors skip it. Diversity > a few BT points.
        """
        from groundhog.utils.direction import (
            attempt_number_from_path,
            direction_exists,
            enforce_inherited_direction,
            inherited_direction_changed,
            mark_result_failed,
            promote_workspace_direction,
            read_direction,
        )
        metadata = self._build_metadata(prior)

        if prior is None:
            promote_workspace_direction(ws.path)
            direction = read_direction(ws.path)
            history = getattr(getattr(self, "_toolkit", None), "history", None)
            if not direction:
                reason = "fresh attempt did not create core_direction.md"
                metadata["gate_failure"] = reason
                mark_result_failed(result, "core_direction", reason)
            elif direction_exists(
                history,
                direction,
                exclude=[attempt_number_from_path(ws.path)],
                only_done=False,
            ):
                reason = "fresh attempt duplicated an existing core direction"
                metadata["gate_failure"] = reason
                mark_result_failed(result, "core_direction", reason)
        elif hasattr(prior, "path"):
            if inherited_direction_changed(ws.path, prior.path):
                metadata["direction_restored"] = True
            enforce_inherited_direction(ws.path, prior.path)

        if self._is_solution_duplicate(ws, prior):
            metadata["non_promotable"] = True
            metadata["non_promotable_reason"] = "solution.py is byte-identical to parent"

        from groundhog.utils.results import write_result
        write_result(ws.path, result, metadata=metadata)

    @staticmethod
    def _is_solution_duplicate(ws, prior) -> bool:
        """True iff the committed solution.py equals the parent's byte-for-byte."""
        if prior is None or not hasattr(prior, "path"):
            return False
        ours = ws.path / "solution.py"
        theirs = prior.path / "solution.py"
        if not ours.exists() or not theirs.exists():
            return False
        try:
            return ours.read_bytes() == theirs.read_bytes()
        except OSError:
            return False

    # --- Logging ---

    def _log_conversation(self, ws):
        """Append agent summary events to conversation.json."""
        summary_path = ws.path / "agent_summary.jsonl"
        if not summary_path.exists():
            return
        try:
            lines = summary_path.read_text(encoding="utf-8").strip().split("\n")
            for line in lines:
                if not line.strip():
                    continue
                entry = json.loads(line)
                role = entry.get("role", "Agent")
                content = entry.get("content", entry.get("text", ""))
                if content:
                    conversation_log(ws.path, str(content), role=role, label="agent")
        except (json.JSONDecodeError, IOError):
            pass

    def _build_metadata(self, prior):
        return {
            "strategy": "agent",
            "prior": prior.number if prior else None,
            "cost": round(self.cost, 6),
        }

    def _build_log(self, attempt, prior, result, toolkit):
        stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=self.through)
        final_name = stages[-1].name
        final_result = result.stages.get(final_name)
        score = stages[-1].score(final_result) if final_result else -1.0
        return {
            "attempt": attempt.number,
            "prior": prior.number if prior else None,
            "score": round(score, 4),
            "strategy": "agent",
        }

    def _score_result(self, result, toolkit):
        """Score a result using the current scorer. Falls back through stages."""
        stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=self.through)
        for stage in reversed(stages):
            stage_result = result.stages.get(stage.name)
            if stage_result is not None:
                return stage.score(stage_result)
        return -1.0
