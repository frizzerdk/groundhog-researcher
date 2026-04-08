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
    ("allow", "Write(workspace/*)"),
    ("allow", "Write(learnings.md)"),
    ("allow", "Edit(workspace/*)"),
    ("allow", "Edit(learnings.md)"),
    ("deny",  "Write(TASK_CONTEXT.md)"),
    ("deny",  "Write(result.json)"),
    ("deny",  "Edit(TASK_CONTEXT.md)"),
    ("deny",  "Edit(result.json)"),
]

PHASE_OVERRIDES = {
    "explore": [
        ("deny", "Write(solution.py)"),
        ("deny", "Edit(solution.py)"),
    ],
    "submit": [
        ("allow", "Write(solution.py)"),
        ("allow", "Edit(solution.py)"),
    ],
    "fix": [
        ("allow", "Write(solution.py)"),
        ("allow", "Edit(solution.py)"),
    ],
    "reflect": [
        ("deny", "Write(solution.py)"),
        ("deny", "Edit(solution.py)"),
    ],
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

Brief, high-value notes that persist across optimization runs.
Keep this document high signal-to-noise. Only add entries that would
save a future optimizer significant time or prevent repeated mistakes.

Good entries: error solutions, confirmed dead-ends, key thresholds,
techniques that produced measurable gains.

Bad entries: speculative ideas, verbose explanations, anything obvious
from reading the code.
"""

EXPLORE_PROMPT = """\
{session_header}

You are an expert code optimizer. You work iteratively using tools.

## Task

{task_context}

## Workflow

You are in the **exploration phase**. When your budget runs out you will
automatically move to a submission phase. Use your budget wisely: understand
WHY the current solution scores what it does before changing anything.

1. Read TASK_CONTEXT.md for the full problem description
2. Read solution.py — the current best approach (READ-ONLY this phase)
3. Run `get-learnings` for accumulated knowledge from previous runs
4. Edit workspace/temp_solution.py with your improvements
5. Run `{eval_command} workspace/temp_solution.py` to evaluate
6. Read the output carefully — understand what improved and why
7. When you beat the current best, copy workspace/temp_solution.py to workspace/best_solution.py
8. Iterate: keep improving workspace/temp_solution.py, save to best_solution.py when you improve

## Key Rules

- solution.py is READ-ONLY — edit workspace/temp_solution.py instead
- When you get a better score, save it: cp workspace/temp_solution.py workspace/best_solution.py
- Your best_solution.py will be submitted in the next phase
- Run `{eval_command} <path>` to evaluate any .py file
- Focus on understanding before changing — blind edits waste iterations
- workspace/ is yours for experiments and analysis scripts
{budget_info}{guidance}

## Files

{file_listing}"""

SUBMIT_PROMPT = """\
Your exploration phase is over. Time to submit your best solution.

1. Review the scores from your evaluation runs during this session
2. Ensure workspace/best_solution.py contains your highest-scoring version
3. If best_solution.py doesn't exist, copy your best from temp_solution.py
4. Copy workspace/best_solution.py to solution.py — you now have write access

Do NOT run any more experiments or modify learnings.md. \
Just make sure solution.py has your best work."""

FIX_PROMPT = """\
Your solution.py failed evaluation with this error:

{error}

Fix the issue in solution.py and run `{eval_command} solution.py` to verify."""

REFLECT_PROMPT = """\
Write a retrospective in learnings.md about this session:
- What approaches did you try and what scores did they get?
- What worked well? What didn't?
- What dead ends should future attempts avoid?
- Any promising directions you didn't have time to explore?

Do not modify solution.py."""

# Per-request explore prompt — agent does everything in one call
EXPLORE_PROMPT_FULL = """\
{session_header}

You are an expert code optimizer. You work iteratively using tools.

## Task

{task_context}

## Workflow

You have one session to improve the solution. Use it well: understand
WHY the current solution scores what it does before changing anything.

1. Read TASK_CONTEXT.md for the full problem description
2. Read solution.py — the current best approach (READ-ONLY)
3. Run `get-learnings` for accumulated knowledge from previous runs
4. Edit workspace/temp_solution.py with your improvements
5. Run `{eval_command} workspace/temp_solution.py` to evaluate
6. Read the output carefully — understand what improved and why
7. When you beat the current best, save it: cp workspace/temp_solution.py workspace/best_solution.py
8. Iterate: keep improving temp_solution.py, save to best_solution.py when you improve
9. When done, write what you learned to workspace/learnings.md — brief, actionable notes

## Key Rules

- solution.py is READ-ONLY — edit workspace/temp_solution.py instead
- When you get a better score, save it: cp workspace/temp_solution.py workspace/best_solution.py
- workspace/best_solution.py will be submitted as your final solution
- workspace/learnings.md will be saved for future runs
- Run `{eval_command} <path>` to evaluate any .py file
- Focus on understanding before changing — blind edits waste iterations
- workspace/ is yours for experiments and analysis scripts
{budget_info}{guidance}

## Files

{file_listing}"""

PATCH_SOLUTION_PROMPT = """\
Look at the agent's work in workspace/ and the evaluation logs.
Find the best-performing version and save it to workspace/best_solution.py.
If workspace/temp_solution.py is the only version, copy it to workspace/best_solution.py."""

PATCH_LEARNINGS_PROMPT = """\
Write a brief retrospective to workspace/learnings.md about the work done so far.
Look at the evaluation results and any changes in workspace/.
- What was tried? What scores?
- What worked? What didn't?
- What to try next?"""


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
        self._submit(toolkit, ws, session_id)
        result = self._evaluate(toolkit, ws)
        result = self._fix_loop(toolkit, ws, session_id, result)
        self._reflect(toolkit, ws, session_id)
        self._log_conversation(ws)

        attempt = ws.commit(result, metadata=self._build_metadata(prior))
        return self._build_log(attempt, prior, result, toolkit)

    def _run_per_request(self, toolkit, ws, prior):
        """Single explore call + verify/patch with cheap models."""
        session_id = self._explore_full(toolkit, ws, prior)
        self._log_conversation(ws)

        # Copy best solution to solution.py (best_solution > temp_solution > patch)
        self._submit_best(toolkit, ws)

        # Evaluate
        result = self._evaluate(toolkit, ws)

        # Fix if needed
        if not result.completed:
            result = self._fix_loop(toolkit, ws, session_id, result)

        # Collect learnings
        self._collect_learnings(toolkit, ws)

        attempt = ws.commit(result, metadata=self._build_metadata(prior))
        return self._build_log(attempt, prior, result, toolkit)

    # --- Init ---

    def _init(self, toolkit, config):
        from groundhog.tools.log import StrategyLog
        self.cfg = self._resolve_config(config)
        self.through = self.cfg.eval_through or getattr(toolkit, 'through', None)
        self.log = toolkit.log if hasattr(toolkit, 'log') else StrategyLog()
        self.cost = 0.0
        self._last_event_text = None

    def _on_event(self, event):
        """Live progress callback for agent events. Prints inline updates."""
        event_type = event.get("type")
        data = event.get("data", {})

        if event_type == "assistant.message":
            content = data.get("content", "")
            if content and content.strip():
                # Show first line of agent's reasoning, truncated
                first_line = content.strip().split("\n")[0][:80]
                if first_line != self._last_event_text:
                    self._last_event_text = first_line
                    self.log.inline(f"\n{self.log.INDENT}  > {first_line}")

        elif event_type == "tool.execution_start":
            tool_name = data.get("toolName", "")
            if tool_name in ("report_intent",):
                return
            args = data.get("arguments", {})
            # Compact tool summary
            if tool_name in ("view", "glob", "grep"):
                detail = args.get("path", "")[:60]
            elif tool_name == "powershell":
                detail = args.get("command", "")[:60]
            elif tool_name == "edit":
                detail = args.get("path", "").split("\\")[-1].split("/")[-1]
            elif tool_name == "create":
                detail = args.get("path", "").split("\\")[-1].split("/")[-1]
            else:
                detail = ""
            summary = f"{tool_name}" + (f" {detail}" if detail else "")
            self.log.inline(f"\n{self.log.INDENT}  [{summary}]")

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
        (ws.path / "TASK_CONTEXT.md").write_text(toolkit.task.context.get(), encoding="utf-8")
        (ws.path / "workspace").mkdir(exist_ok=True)
        if prior:
            (ws.path / "solution.py").write_text(prior.code, encoding="utf-8")
            # Pre-populate temp_solution.py — agent works on this during explore
            (ws.path / "workspace" / "temp_solution.py").write_text(prior.code, encoding="utf-8")
        if prior is not None and hasattr(prior, 'path'):
            approach_path = prior.path / "approach.md"
            if approach_path.exists():
                (ws.path / "approach.md").write_text(
                    approach_path.read_text(encoding="utf-8"), encoding="utf-8")
        # Seed learnings.md if it doesn't exist
        learnings_path = ws.path / "learnings.md"
        if not learnings_path.exists():
            learnings_path.write_text(LEARNINGS_SEED, encoding="utf-8")

    # --- Tool filtering ---

    def _get_tools(self, toolkit, phase="explore"):
        """Filter toolkit.agent_tools by phase."""
        allowed = PHASE_TOOLS.get(phase)
        all_tools = getattr(toolkit, 'agent_tools', [])
        if allowed is not None and not allowed:
            return []
        if allowed is None:
            return list(all_tools)
        return [t for t in all_tools if t.name in allowed]

    # --- Helpers ---

    def _get_eval_command(self, toolkit):
        """Get the first eval stage name for prompt references."""
        stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=self.through)
        return stages[0].name if stages else "evaluate"

    def _build_file_listing(self, ws):
        """List workspace files for agent orientation."""
        files = sorted(
            str(f.relative_to(ws.path))
            for f in ws.path.rglob("*") if f.is_file()
        )
        return "\n".join(f"  {f}" for f in files) if files else "  (empty)"

    # --- Phases ---

    def _explore(self, toolkit, ws, prior):
        """Main work phase — agent explores in workspace/."""
        # Build session header
        if prior:
            prior_score = self._score_result(prior.result, toolkit)
            session_header = f"[{toolkit.task.name} #{ws.number}] prior=#{prior.number} score={prior_score:.4f}"
        else:
            session_header = f"[{toolkit.task.name} #{ws.number}] fresh start"

        eval_command = self._get_eval_command(toolkit)
        budget_info = ""
        if self.cfg.budget_usd:
            budget_info = f"\n- Budget: ${self.cfg.budget_usd:.2f} for this exploration phase."
        if self.cfg.timeout:
            minutes = self.cfg.timeout // 60
            budget_info += f"\n- Time limit: ~{minutes} minutes."
        guidance = f"\n\n## Additional Guidance\n{self.cfg.guidance}" if self.cfg.guidance else ""

        goal = EXPLORE_PROMPT.format(
            session_header=session_header,
            task_context=toolkit.task.context.get(),
            eval_command=eval_command,
            budget_info=budget_info,
            guidance=guidance,
            file_listing=self._build_file_listing(ws),
        )

        budget_str = f"${self.cfg.budget_usd:.2f}" if self.cfg.budget_usd else "unlimited"
        self.log.start(f"--- Agent | {'prior=#' + str(prior.number) if prior else 'fresh'} | budget={budget_str}")

        tools = self._get_tools(toolkit, phase="explore")
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
        self.cost += result.cost
        self.log.tock()

        return result.session_id

    def _explore_full(self, toolkit, ws, prior):
        """Per-request explore — agent does everything in one call."""
        if prior:
            prior_score = self._score_result(prior.result, toolkit)
            session_header = f"[{toolkit.task.name} #{ws.number}] prior=#{prior.number} score={prior_score:.4f}"
        else:
            session_header = f"[{toolkit.task.name} #{ws.number}] fresh start"

        eval_command = self._get_eval_command(toolkit)
        budget_info = ""
        if self.cfg.timeout:
            minutes = self.cfg.timeout // 60
            budget_info = f"\n- Time limit: ~{minutes} minutes."
        guidance = f"\n\n## Additional Guidance\n{self.cfg.guidance}" if self.cfg.guidance else ""

        goal = EXPLORE_PROMPT_FULL.format(
            session_header=session_header,
            task_context=toolkit.task.context.get(),
            eval_command=eval_command,
            budget_info=budget_info,
            guidance=guidance,
            file_listing=self._build_file_listing(ws),
        )

        self.log.start(f"--- Agent (per-request) | {'prior=#' + str(prior.number) if prior else 'fresh'}")

        tools = self._get_tools(toolkit, phase="explore")
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
        self.cost += result.cost
        self.log.tock()

        return result.session_id

    def _patch(self, toolkit, ws, prompt):
        """Cheap agent call to fix a missed step (copy solution, write learnings)."""
        self.log.inline("patch... ")
        allow, deny = _resolve_permissions("explore")
        spec = AgentSpec(
            goal=prompt,
            workspace_path=ws.path,
            tools=[],
            allowed_tools=allow,
            denied_tools=deny,
            timeout=60,
        )
        # Use budget tier for cheap patching
        tier = "budget" if "budget" in toolkit.agent._tiers else "default"
        result = toolkit.agent.get(tier).run(spec)
        self.cost += result.cost
        self.log.tock()

    def _submit_best(self, toolkit, ws):
        """Copy the best workspace solution to solution.py.

        Priority: best_solution.py > temp_solution.py > patch with cheap model.
        """
        best = ws.path / "workspace" / "best_solution.py"
        temp = ws.path / "workspace" / "temp_solution.py"
        prior = ws.path / "solution.py"

        if best.exists():
            (ws.path / "solution.py").write_text(best.read_text(encoding="utf-8"), encoding="utf-8")
        elif temp.exists():
            # Check if temp was actually modified (not just the prior copy)
            temp_code = temp.read_text(encoding="utf-8")
            prior_code = prior.read_text(encoding="utf-8") if prior.exists() else ""
            if temp_code != prior_code:
                (ws.path / "solution.py").write_text(temp_code, encoding="utf-8")
            else:
                # Agent didn't modify anything — patch with cheap model
                self._patch(toolkit, ws, PATCH_SOLUTION_PROMPT)
                if (ws.path / "workspace" / "best_solution.py").exists():
                    (ws.path / "solution.py").write_text(
                        (ws.path / "workspace" / "best_solution.py").read_text(encoding="utf-8"),
                        encoding="utf-8")
        else:
            self._patch(toolkit, ws, PATCH_SOLUTION_PROMPT)
            if (ws.path / "workspace" / "best_solution.py").exists():
                (ws.path / "solution.py").write_text(
                    (ws.path / "workspace" / "best_solution.py").read_text(encoding="utf-8"),
                    encoding="utf-8")

    def _collect_learnings(self, toolkit, ws):
        """Read learnings from workspace, patch with cheap model if missing."""
        learnings_path = ws.path / "workspace" / "learnings.md"
        if learnings_path.exists():
            text = learnings_path.read_text(encoding="utf-8").strip()
            if text and hasattr(toolkit, 'learnings'):
                toolkit.learnings.add(text)
                return

        # Patch: cheap agent to write learnings
        self._patch(toolkit, ws, PATCH_LEARNINGS_PROMPT)
        if learnings_path.exists() and hasattr(toolkit, 'learnings'):
            text = learnings_path.read_text(encoding="utf-8").strip()
            if text:
                toolkit.learnings.add(text)

    def _submit(self, toolkit, ws, session_id):
        """Agent finalizes best solution into solution.py (per-token path)."""
        allow, deny = _resolve_permissions("submit")
        spec = AgentSpec(
            goal=SUBMIT_PROMPT,
            workspace_path=ws.path,
            tools=self._get_tools(toolkit, phase="submit"),
            model=self.cfg.model,
            effort=self.cfg.effort,
            allowed_tools=allow,
            denied_tools=deny,
            timeout=self.cfg.phase_timeout,
            session_id=session_id,
        )
        result = toolkit.agent.get("default").run(spec)
        self.cost += result.cost
        self.log.inline("submit... ")
        self.log.tock()

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
        """Agent writes learnings."""
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

        # Read agent's learnings and add to persistent store
        learnings_path = ws.path / "learnings.md"
        if learnings_path.exists() and hasattr(toolkit, 'learnings'):
            text = learnings_path.read_text(encoding="utf-8").strip()
            if text and text != LEARNINGS_SEED.strip():
                toolkit.learnings.add(text)

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
        stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=self.through)
        final_name = stages[-1].name
        final_result = result.stages.get(final_name)
        return stages[-1].score(final_result) if final_result else -1.0
