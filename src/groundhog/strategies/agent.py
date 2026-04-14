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
import sys
from dataclasses import dataclass
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


# --- Permissions (depth-based, deepest/last wins) ---

BASE_PERMISSIONS = [
    ("allow", "Read(*)"),
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


def _resolve_permissions(phase):
    """Merge base + phase overrides into allow/deny lists."""
    rules = BASE_PERMISSIONS + PHASE_OVERRIDES.get(phase, [])
    return (
        [r for a, r in rules if a == "allow"],
        [r for a, r in rules if a == "deny"],
    )


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

Notes from this optimization chain. Keep high signal-to-noise.
Only add entries that would save time or prevent repeated mistakes.

Good: confirmed dead-ends, key thresholds, techniques with measurable gains.
Bad: speculative ideas, verbose explanations, anything obvious from the code.
"""

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
- Focus on understanding before changing — blind edits waste iterations
{budget_info}{guidance}

## Files

{file_listing}"""

SUBMIT_PROMPT = """\
Your exploration phase is over. Your work/solution.py will be submitted.
No action needed — it is copied automatically."""

FIX_PROMPT = """\
Your work/solution.py failed evaluation with this error:

{error}

Fix the issue in work/solution.py and run `{eval_command}` to verify."""

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
- Focus on understanding before changing — blind edits waste iterations
{budget_info}{guidance}

## Files

{file_listing}"""


class AgentStrategy(Strategy):
    """Delegate optimization work to an autonomous agent.

    Composed method pattern:
        init → select prior → workspace → prepare → explore → submit
        → evaluate → fix loop → reflect → log → commit
    """

    Config = AgentConfig

    def __call__(self, toolkit, config=None):
        self._init(toolkit, config)

        if not hasattr(toolkit, 'agent'):
            return {"skipped": "no agent backend available"}

        prior = self._select_prior(toolkit)
        ws = self._start_workspace(toolkit, prior)

        try:
            self._prepare_workspace(toolkit, ws, prior)

            backend = toolkit.agent.get("default")
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
        self._submit_best(toolkit, ws)
        self.log.inline("submit... ")
        self.log.tock()
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

    def _on_event(self, event):
        """Live progress callback — overwrites a single status line with counter."""
        event_type = event.get("type")
        data = event.get("data", {})
        text = None

        if event_type == "assistant.message":
            content = data.get("content", "")
            if content and content.strip():
                first_line = content.strip().split("\n")[0][:60]
                self._event_count += 1
                text = f"> {first_line}"

        elif event_type == "tool.execution_start":
            tool_name = data.get("toolName", "")
            if tool_name in ("report_intent",):
                return
            self._event_count += 1
            args = data.get("arguments", {})
            if tool_name in ("view", "glob", "grep"):
                detail = args.get("path", "").split("\\")[-1].split("/")[-1]
            elif tool_name == "powershell":
                detail = args.get("command", "")[:50].split("\n")[0]
            elif tool_name in ("edit", "create"):
                detail = args.get("path", "").split("\\")[-1].split("/")[-1]
            else:
                detail = ""
            text = f"[{tool_name}] {detail}" if detail else f"[{tool_name}]"

        if text:
            line = f"{self.log.INDENT}  #{self._event_count} {text}"
            pad = max(0, self._event_line_len - len(line))
            sys.stdout.write(f"\r{line}{' ' * pad}")
            sys.stdout.flush()
            self._event_line_len = len(line)

    def _clear_event_line(self):
        """Clear the in-place event status line."""
        if self._event_line_len > 0:
            sys.stdout.write(f"\r{' ' * self._event_line_len}\r")
            sys.stdout.flush()
            self._event_line_len = 0

    # --- Selection ---

    def _select_prior(self, toolkit):
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

        # Copy approach.md from prior (read-only for agent)
        if prior is not None and hasattr(prior, 'path'):
            approach_path = prior.path / "approach.md"
            if approach_path.exists():
                (ws.path / "approach.md").write_text(
                    approach_path.read_text(encoding="utf-8"), encoding="utf-8")

        # Seed work/solution.py from prior
        if prior:
            (ws.path / "work" / "solution.py").write_text(prior.code, encoding="utf-8")

        # Seed work/learnings.md from prior's local learnings (chain knowledge)
        if prior is not None and hasattr(prior, 'path'):
            prior_learnings = prior.path / "work" / "learnings.md"
            if prior_learnings.exists():
                (ws.path / "work" / "learnings.md").write_text(
                    prior_learnings.read_text(encoding="utf-8"), encoding="utf-8")
        if not (ws.path / "work" / "learnings.md").exists():
            (ws.path / "work" / "learnings.md").write_text(LEARNINGS_SEED, encoding="utf-8")

    # --- Tool filtering ---

    def _get_tools(self, toolkit, phase="explore", prior=None):
        """Get tools for the agent: toolkit tools + prior tools."""
        allowed = PHASE_TOOLS.get(phase)
        all_tools = getattr(toolkit, 'agent_tools', [])
        if allowed is not None and not allowed:
            return []
        if allowed is None:
            tools = list(all_tools)
        else:
            tools = [t for t in all_tools if t.name in allowed]

        # Add prior file access tools (per-attempt, only during explore/fix)
        if phase in ("explore", "fix") and prior is not None:
            from groundhog.agents.tools import build_prior_tools
            tools.extend(build_prior_tools(prior))

        return tools

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
        """Build optional approach section from approach.md."""
        approach_path = ws.path / "approach.md"
        if approach_path.exists():
            text = approach_path.read_text(encoding="utf-8").strip()
            if text:
                return f"\n## Approach (preserve this direction)\n\n{text}\n"
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
            file_listing=self._build_file_listing(ws),
        )

    def _explore(self, toolkit, ws, prior):
        """Main work phase — agent works in work/."""
        goal = EXPLORE_PROMPT.format(**self._build_prompt_vars(toolkit, ws, prior))

        budget_str = f"${self.cfg.budget_usd:.2f}" if self.cfg.budget_usd else "unlimited"
        self.log.start(f"--- Agent | {'prior=#' + str(prior.number) if prior else 'fresh'} | budget={budget_str}")

        tools = self._get_tools(toolkit, phase="explore", prior=prior)
        allow, deny = _resolve_permissions("explore")

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
        result = toolkit.agent.get("default").run(spec)
        self._clear_event_line()
        self.cost += result.cost
        self.log.tock()

        return result.session_id

    def _explore_full(self, toolkit, ws, prior):
        """Per-request explore — agent does everything in one call."""
        goal = EXPLORE_PROMPT_FULL.format(**self._build_prompt_vars(toolkit, ws, prior))

        self.log.start(f"--- Agent (per-request) | {'prior=#' + str(prior.number) if prior else 'fresh'}")

        tools = self._get_tools(toolkit, phase="explore", prior=prior)
        allow, deny = _resolve_permissions("explore")

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
        result = toolkit.agent.get("high").run(spec)
        self._clear_event_line()
        self.cost += result.cost
        self.log.tock()

        return result.session_id

    def _submit_best(self, toolkit, ws):
        """Copy work/solution.py to attempt root. Simple — one file, no patching."""
        src = ws.path / "work" / "solution.py"
        if src.exists():
            (ws.path / "solution.py").write_text(
                src.read_text(encoding="utf-8"), encoding="utf-8")

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
        self.log.inline("evaluating... ")
        result = toolkit.task.evaluate(ws.path, through=self.through)
        self.log.tock()
        return result

    def _fix_loop(self, toolkit, ws, session_id, result):
        """Retry on evaluation failure."""
        eval_command = self._get_eval_command(toolkit)
        for retry in range(self.cfg.max_retries):
            if result.completed:
                return result

            error_stage = result.stages[result.failed_stage]
            error_text = f"Stage '{result.failed_stage}': {error_stage.errors}"

            self.log.inline(f"fix {retry + 1}... ")
            allow, deny = _resolve_permissions("fix")
            goal = FIX_PROMPT.format(error=error_text, eval_command=eval_command)
            spec = AgentSpec(
                goal=goal,
                workspace_path=ws.path,
                tools=self._get_tools(toolkit, phase="fix"),
                model=self.cfg.model,
                effort=self.cfg.effort,
                allowed_tools=allow,
                denied_tools=deny,
                timeout=self.cfg.phase_timeout,
                session_id=session_id,
            )
            fix_result = toolkit.agent.get("default").run(spec)
            self.cost += fix_result.cost
            self.log.tock()

            result = toolkit.task.evaluate(ws.path, through=self.through)

        return result

    def _reflect(self, toolkit, ws, session_id):
        """Agent writes learnings to work/learnings.md."""
        allow, deny = _resolve_permissions("reflect")
        spec = AgentSpec(
            goal=REFLECT_PROMPT,
            workspace_path=ws.path,
            tools=self._get_tools(toolkit, phase="reflect"),
            model=self.cfg.model,
            effort=self.cfg.effort,
            allowed_tools=allow,
            denied_tools=deny,
            timeout=self.cfg.phase_timeout,
            session_id=session_id,
        )
        result = toolkit.agent.get("default").run(spec)
        self.cost += result.cost
        self.log.inline("reflect... ")
        self.log.tock()

        # Promote local learnings to task-level store
        self._collect_learnings(toolkit, ws)

    # --- Finalization ---

    def _finalize(self, ws, result, prior):
        """Write result.json and solution.py to attempt root before commit."""
        from groundhog.utils.results import write_result
        write_result(ws.path, result, metadata=self._build_metadata(prior))
        # Copy work/solution.py to root (canonical location for committed attempts)
        work_solution = ws.path / "work" / "solution.py"
        if work_solution.exists():
            (ws.path / "solution.py").write_text(
                work_solution.read_text(encoding="utf-8"), encoding="utf-8")

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
