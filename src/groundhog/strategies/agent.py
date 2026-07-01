"""Agent strategy — delegate work to an autonomous agent backend.

The strategy orchestrates multi-phase agent execution:
    explore (main work) → submit (finalize) → evaluate → fix (retry) → reflect (learnings)

The agent backend (Claude Code, Gemini CLI, etc.) handles the actual
subprocess, tool exposure, and event logging. The strategy owns phasing,
workspace lifecycle, tool filtering, and evaluation.

Tools are provided by toolkit.agent_tools (built by optimizer).
The strategy filters which tools are available per phase.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from groundhog.base.agent import AgentSpec
from groundhog.base.strategy import Strategy, StrategyConfig, param
from groundhog.tools.attempt_logger import (
    AssistantEvent, LogEvent, MarkdownAttemptLogger,
    ToolCallEvent, UserEvent, eval_event,
)


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
    name: str = param(
        "",
        "Optional display name (human-readable slug). PlanApproaches sets this "
        "from the proposed direction's name; otherwise it is derived from the "
        "core direction at commit.",
    )
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


def _normalize_agent_event(event: dict) -> Optional[LogEvent]:
    """Map a raw backend event to a typed LogEvent for the attempt log.

    Recognised shapes (all five backends emit different keys):

    - copilot:    ``{"type": "tool.execution_start", "data": {"toolName": ...}}``
                  ``{"type": "assistant.message", "data": {"content": ...}}``
    - claude_code:``{"type": "assistant", "message": {"content": [blocks]}}``
    - opencode:   ``{"type": "tool_use", "part": {"tool": ..., "state": {...}}}``
    - gemini_cli: ``{"type": "tool_use", "tool_name": ..., "parameters": {...}}``
    - codex_cli:  ``{"type": "item.completed", "item": {"type": ...}}``

    Returns ``None`` for events that aren't worth recording (deltas, init
    pings, step boundaries). The full raw stream stays in agent_steps.jsonl;
    this is the structured view. Console rendering derives from the event's
    own ``to_console()``.
    """
    et = event.get("type", "")

    def _tool(name, inp, path_keys=("file_path", "path")):
        args = {}
        for k in path_keys:
            if inp.get(k):
                args["path"] = str(inp[k])
                break
        if not args:
            cmd = inp.get("command", "")
            if isinstance(cmd, str) and cmd:
                args["command"] = cmd.split("\n")[0][:200]
        return ToolCallEvent(name=name or "tool", args=args)

    # ---- copilot ----
    if et == "tool.execution_start":
        data = event.get("data", {}) or {}
        name = data.get("toolName", "")
        if name in ("report_intent", "task_complete"):
            return None
        return _tool(name, data.get("arguments", {}) or {})

    if et == "assistant.message":
        content = (event.get("data", {}) or {}).get("content", "")
        return AssistantEvent(content=content) if content.strip() else None

    # ---- claude_code (stream-json assistant blocks) ----
    if et == "assistant":
        msg = event.get("message", {}) or {}
        blocks = msg.get("content", []) if isinstance(msg.get("content"), list) else []
        for block in blocks:
            bt = block.get("type")
            if bt == "tool_use":
                return _tool(block.get("name", "tool"), block.get("input", {}) or {})
            if bt == "thinking" and (block.get("thinking") or "").strip():
                return AssistantEvent(content=block["thinking"],
                                      data={"channel": "thinking"})
            if bt == "text" and (block.get("text") or "").strip():
                return AssistantEvent(content=block["text"])
        return None

    # ---- opencode (part-shape) / gemini_cli (flat shape) ----
    if et == "tool_use":
        part = event.get("part", {}) or {}
        if part.get("type") == "tool":
            inp = (part.get("state", {}) or {}).get("input", {}) or {}
            return _tool(part.get("tool", "tool"), inp, path_keys=("filePath", "path"))
        tool = event.get("tool_name", "")
        if tool:
            return _tool(tool, event.get("parameters", {}) or {})

    # ---- codex_cli (item.* events) ----
    if et == "item.completed":
        item = event.get("item", {}) or {}
        itype = item.get("type")
        if itype == "command_execution":
            cmd = (item.get("command", "") or "").split("\n")[0][:200]
            return ToolCallEvent(name="shell", args={"command": cmd}) if cmd else None
        if itype == "agent_message" and (item.get("text") or "").strip():
            return AssistantEvent(content=item["text"])

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

        # Point the attempt log at the new workspace and open the console
        # box so the user sees the run starting before the agent backend
        # produces its first event. Optimizer finalizes with
        # attempt_done/attempt_failed in _log_attempt.
        self.logger.attempt_start(
            ws.path,
            num=ws.display_id,
            prior=prior.id if prior else None,
            queue_label=getattr(toolkit, "_current_queue_label", "") or "",
            budget_total=self.cfg.budget_usd or 0.0,
        )

        # Bracket the attempt lifetime on the toolkit's pointer: the setter
        # lives where the workspace is born, and the bracket's finally clears
        # on every exit path. Build-time tools that closed over toolkit.ws
        # read THIS attempt's dir while we're inside. commit() happens inside
        # the bracket and is the last ws.path-touching statement (the folder
        # is renamed at commit).
        from contextlib import nullcontext
        handle = getattr(toolkit, "ws", None)
        bracket = handle.attempt(ws) if handle is not None else nullcontext()

        try:
            with bracket:
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
        result = self._fix_loop(toolkit, ws, session_id, result, prior=prior)
        self._reflect(toolkit, ws, session_id)

        self._finalize(ws, result, prior)
        attempt = ws.commit(success=result.completed)
        return self._build_log(attempt, prior, result, toolkit)

    def _run_per_request(self, toolkit, ws, prior):
        """Single explore call — agent edits work/solution.py directly."""
        session_id = self._explore_full(toolkit, ws, prior)

        # Copy work/solution.py to attempt root for evaluation
        self._submit_best(toolkit, ws)

        # Evaluate
        result = self._evaluate(toolkit, ws)

        # Fix if needed
        if not result.completed:
            result = self._fix_loop(toolkit, ws, session_id, result, prior=prior)

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
        self.logger = getattr(toolkit, 'attempt_logger', None) or MarkdownAttemptLogger()
        # Stash for use during finalize / subclass hooks (e.g. FreshAgent
        # generates its core direction post-session and needs LLM access).
        self._toolkit = toolkit

    def _on_event(self, event):
        """Live progress callback fired by every agent backend.

        Raw backend events are normalized into typed LogEvents and recorded
        in the attempt log, which also feeds the live console renderer.
        Events that aren't worth recording are silently dropped — the raw
        stream stays in agent_steps.jsonl.
        """
        ev = _normalize_agent_event(event)
        if ev is not None:
            self.logger.log(ev)

    def _update_attempt_log(self, toolkit, **kwargs):
        """Push a partial state update to the per-attempt console pane;
        phase transitions are also recorded as events."""
        from groundhog.tools.attempt_logger import PhaseEvent
        if "phase" in kwargs:
            self.logger.log(PhaseEvent(phase=str(kwargs["phase"])))
        self.logger.update(**kwargs)

    # --- Selection ---

    def _select_prior(self, toolkit):
        # Explicit pinning via config. Bypasses any auto-selection so queue
        # items (or task code) can target a specific attempt for any
        # reason — known clean baseline, reproducing a result, exploring
        # a non-best branch, etc. Returns None if the named attempt
        # doesn't exist — caller treats that as "no prior available".
        forced = getattr(self.cfg, "force_prior_attempt", None)
        if forced is not None:
            return toolkit.history.get(str(forced))
        if self.cfg.target == "best":
            stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=self.through)
            return toolkit.history.best(stages[-1].score)
        if hasattr(toolkit, 'get_prior'):
            return toolkit.get_prior(toolkit)
        stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=self.through)
        return toolkit.history.best(stages[-1].score)

    # --- Workspace ---

    def _start_workspace(self, toolkit, prior):
        parent = prior.id if prior else None
        return toolkit.history.workspace(parent=parent)

    def _prepare_workspace(self, toolkit, ws, prior):
        # Strategy-managed files in attempt root
        (ws.path / "TASK_CONTEXT.md").write_text(toolkit.task.context.get(), encoding="utf-8")

        # Inherit core direction from prior (read-only for the agent during
        # the session; re-enforced at commit so an agent can't fork the family
        # by rewriting it).
        if prior is not None:
            from groundhog.utils.direction import inherit_direction_from_attempt
            inherit_direction_from_attempt(prior, ws.path)

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

        # Base layer: framework defaults + task-hook tools, merged by
        # assemble_toolkit. Strategy tools below get merged OVER this base
        # (strategy > task > default) by the same name-keyed rule.
        base = list(getattr(toolkit, 'agent_tools', []))
        tools = []

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
            if prior is None:
                parent_solution_path = None
            elif hasattr(prior, "path"):
                parent_solution_path = Path(prior.path) / "solution.py"
            else:
                # git: materialize the parent solution OUTSIDE the workspace so
                # the eval tool can read it without committing it into the child.
                import tempfile
                _tf = tempfile.NamedTemporaryFile(
                    mode="w", suffix="_parent_solution.py", delete=False,
                    encoding="utf-8")
                _tf.write(prior.code or "")
                _tf.close()
                parent_solution_path = Path(_tf.name)
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

        # One collision rule for the whole pipeline: strategy tools shadow
        # same-named base tools (previously a plain concat — duplicates went
        # to the agent undetected).
        from groundhog.agents.tools import _merge_agent_tools
        return _merge_agent_tools(base, tools, layer="strategy",
                                  log=getattr(toolkit, "log", None))

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
            return f"[{toolkit.task.name} #{ws.display_id}] fresh start"

        prior_score = self._score_result(prior.result, toolkit)
        header = f"[{toolkit.task.name} #{ws.display_id}] prior=#{prior.id} score={prior_score:.4f}"

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
        self.logger.log(UserEvent(content=goal, data={"label": "explore"}))

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
        self.logger.log(LogEvent(type="agent_run", cost=result.cost,
                                 data={"turns": result.turns or 0}))
        self._update_attempt_log(
            toolkit, budget_used=self.logger.total_cost(), turns=result.turns or 0,
        )

        return result.session_id

    def _explore_full(self, toolkit, ws, prior):
        """Per-request explore — agent does everything in one call."""
        goal = EXPLORE_PROMPT_FULL.format(**self._build_prompt_vars(toolkit, ws, prior))
        self._update_attempt_log(toolkit, phase="explore")
        self.logger.log(UserEvent(content=goal, data={"label": "explore"}))

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
        self.logger.log(LogEvent(type="agent_run", cost=result.cost,
                                 data={"turns": result.turns or 0}))
        self._update_attempt_log(
            toolkit, budget_used=self.logger.total_cost(), turns=result.turns or 0,
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
        self.logger.log(eval_event(result, self._score_result(result, toolkit)))
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
            self.logger.log(UserEvent(content=goal, data={"label": f"fix {retry + 1}"}))
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
                on_event=self._on_event,
            )
            fix_result = toolkit.agent.get(self.cfg.tier).run(spec)
            self.logger.log(LogEvent(type="agent_run", cost=fix_result.cost,
                                     data={"turns": fix_result.turns or 0}))

            # Root is known-failed here; fix replaces it unconditionally so
            # the re-eval sees the fix even if the agent didn't run the eval
            # tool (promote-best is intentionally disabled for fix phase).
            work_solution = ws.path / "work" / "solution.py"
            if work_solution.exists():
                (ws.path / "solution.py").write_text(
                    work_solution.read_text(encoding="utf-8"), encoding="utf-8")

            result = toolkit.task.evaluate(ws.path, through=self.through)
            self.logger.log(eval_event(result, self._score_result(result, toolkit)))

        return result

    def _reflect(self, toolkit, ws, session_id):
        """Agent writes learnings to work/learnings.md."""
        self._update_attempt_log(toolkit, phase="reflect")
        self.logger.log(UserEvent(content=REFLECT_PROMPT, data={"label": "reflect"}))
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
            on_event=self._on_event,
        )
        result = toolkit.agent.get(self.cfg.tier).run(spec)
        self.logger.log(LogEvent(type="agent_run", cost=result.cost,
                                 data={"turns": result.turns or 0}))

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
            direction_exists,
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
                exclude=[ws.display_id],
                only_done=False,
            ):
                reason = "fresh attempt duplicated an existing core direction"
                metadata["gate_failure"] = reason
                mark_result_failed(result, "core_direction", reason)
        elif prior is not None:
            from groundhog.utils.direction import (
                inherit_direction_from_attempt,
                inherited_direction_changed_from,
            )
            if inherited_direction_changed_from(ws.path, prior):
                metadata["direction_restored"] = True
            inherit_direction_from_attempt(prior, ws.path)

        if self._is_solution_duplicate(ws, prior):
            metadata["non_promotable"] = True
            metadata["non_promotable_reason"] = "solution.py is byte-identical to parent"

        from groundhog.utils.results import write_result
        write_result(ws.path, result, metadata=metadata)

        # Display name: the planned slug if provided (e.g. PlanApproaches),
        # else a slug of the finalized core direction.
        from groundhog.utils.direction import workspace_name
        ws.name = workspace_name(ws.path, explicit=self.cfg.name)

    @staticmethod
    def _is_solution_duplicate(ws, prior) -> bool:
        """True iff the committed solution.py equals the parent's (backend-agnostic)."""
        from groundhog.utils.direction import solution_matches_attempt
        return solution_matches_attempt(ws.path, prior)

    # --- Logging ---

    def _build_metadata(self, prior):
        return {
            "strategy": "agent",
            "prior": prior.id if prior else None,
            "cost": round(self.logger.total_cost(), 6),
        }

    def _build_log(self, attempt, prior, result, toolkit):
        stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=self.through)
        final_name = stages[-1].name
        final_result = result.stages.get(final_name)
        score = stages[-1].score(final_result) if final_result else -1.0
        return {
            "attempt": attempt.id,
            "prior": prior.id if prior else None,
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
