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
7. Iterate: keep improving workspace/temp_solution.py

## Key Rules

- solution.py is READ-ONLY — edit workspace/temp_solution.py instead
- Your best version will be submitted to solution.py in the next phase
- Run `{eval_command} <path>` to evaluate any .py file
- Focus on understanding before changing — blind edits waste iterations
- workspace/ is yours for experiments and analysis scripts
{budget_info}{guidance}

## Files

{file_listing}"""

SUBMIT_PROMPT = """\
Your exploration phase is over. Time to submit your best solution.

1. Review the scores from your evaluation runs during this session
2. Ensure workspace/temp_solution.py contains your best version
3. Copy it to solution.py — you now have write access

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

            session_id = self._explore(toolkit, ws, prior)
            self._submit(toolkit, ws, session_id)
            result = self._evaluate(toolkit, ws)
            result = self._fix_loop(toolkit, ws, session_id, result)
            self._reflect(toolkit, ws, session_id)
            self._log_conversation(ws)

            attempt = ws.commit(result, metadata=self._build_metadata(prior))
            return self._build_log(attempt, prior, result, toolkit)

        except Exception as e:
            ws.abort()
            return {"skipped": f"agent error: {e}"}

    # --- Init ---

    def _init(self, toolkit, config):
        from groundhog.tools.log import StrategyLog
        self.cfg = self._resolve_config(config)
        self.through = self.cfg.eval_through or getattr(toolkit, 'through', None)
        self.log = toolkit.log if hasattr(toolkit, 'log') else StrategyLog()
        self.cost = 0.0

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
        )
        result = toolkit.agent.get("default").run(spec)
        self.cost += result.cost
        self.log.tock()

        return result.session_id

    def _submit(self, toolkit, ws, session_id):
        """Agent finalizes best solution into solution.py."""
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
