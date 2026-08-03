"""Challenge strategy — falsify the blocking assumption behind a failure.

Instead of retrying a failed or stalled direction, attack the ASSUMPTION
that blocks it. Pick a challengeable target (a failed attempt, or a family
that stopped improving), have the LLM name the claimed blocker and how well
it was ever verified, then generate a solution that attempts the
falsification plan. A failed challenge is still recorded — its learnings
state whether the assumption survived.
"""

import re
from dataclasses import dataclass
from typing import Optional

from groundhog.base.strategy import Strategy, StrategyConfig, param
from groundhog.tools.attempt_logger import (
    AssistantEvent, LogEvent, MarkdownAttemptLogger, SystemEvent, UserEvent, eval_event,
)
from groundhog.utils.codegen import (
    GenerationFailed, build_prompt, extract_code, generate_text,
)
from groundhog.utils.direction import (
    direction_title, read_direction_from_attempt, write_direction,
)
from groundhog.utils.queries import safe_code, safe_result


@dataclass
class ChallengeConfig(StrategyConfig):
    """Configuration for the Challenge strategy."""
    target: str = param("auto", "auto = pick a failed attempt or stale family; or a specific attempt id")
    staleness_window: int = param(5, "A family with no improvement over its last K attempts counts as stale")
    max_retries: int = param(3, "Max retry attempts when evaluation fails")
    learnings_last: int = param(20, "Most recent learnings to include in prompts")
    learnings_random: int = param(10, "Random older learnings to include for diversity")
    exclude_failures: str = param(
        "gate_failure,SyntaxError,NameError",
        "Comma-separated mechanical-failure markers skipped at target "
        "selection: 'gate_failure' matches gate-rejected attempts, any "
        "other token matches substrings of the failed stage's errors — "
        "such failures carry no assumption worth falsifying")


@dataclass
class _Target:
    """A challengeable item: kind is "attempt" (a failed attempt, challenged
    in-lineage) or "family" (a stale family, challenged fresh via its leader)."""
    kind: str
    attempt: object
    code: str
    reason: Optional[str]
    direction: Optional[str]


class Challenge(Strategy):
    """Falsify the blocking assumption behind a failed or stalled direction.

    Composed method pattern:
        init -> select target -> extract assumption -> attack -> evaluate -> finalize
    """

    Config = ChallengeConfig

    def __call__(self, toolkit, config=None):
        self._init(toolkit, config)
        if not hasattr(toolkit, 'llm'):
            return {"skipped": "no LLM available"}
        target = self._select_target(toolkit)
        if target is None:
            return {"skipped": "no challengeable target"}
        self.log.start(f"--- Challenge | {target.kind}=#{target.attempt.id}"
                       + (" | reason stated" if target.reason else ""))
        ws = self._start_workspace(toolkit, target)
        # Anything raising past this point (LLM error, logger, user
        # evaluator crash outside the retry net) must not leak an
        # in-progress workspace dir.
        try:
            self.logger.attempt_start(ws.path)
            self.log.inline("diagnosing... ")
            diagnosis = self._extract_assumption(toolkit, ws, target)
            self.log.tock()
            self._prepare_workspace(toolkit, ws, diagnosis)
            self.log.inline("attacking... ")
            self._attack(toolkit, ws, target, diagnosis)
            self.log.tock()
            self.log.inline("evaluating... ")
            result = self._evaluate(toolkit, ws)
            self.log.tock()
            self.log.inline("learnings... ")
            self._record_learnings(toolkit, target, diagnosis, result)
            self.log.tock()
            attempt = self._finalize(toolkit, ws, result, target, diagnosis)
        except GenerationFailed as e:
            self.log.inline("generation failed... ")
            attempt = self._finalize_failed(toolkit, ws, str(e))
            return {"attempt": attempt.id, "target": target.attempt.id,
                    "failed": str(e), "strategy": self.name}
        except BaseException:
            ws.abort()
            raise
        return self._build_log(attempt, target, diagnosis, result)

    # --- Init ---

    def _init(self, toolkit, config):
        from groundhog.tools.log import StrategyLog
        self.cfg = self._resolve_config(config)
        self.logger = getattr(toolkit, 'attempt_logger', None) or MarkdownAttemptLogger()
        self.through = getattr(toolkit, 'through', None)
        self.log = toolkit.log if hasattr(toolkit, 'log') else StrategyLog()
        self._stages = toolkit.task.evaluator.eval_stages(
            toolkit.task.data, through=self.through)

    # --- Target selection ---

    def _select_target(self, toolkit):
        history = getattr(toolkit, 'history', None)
        if history is None:
            return None
        if self.cfg.target != "auto":
            return self._explicit_target(toolkit, history)
        challenged = self._challenged_ids(history)
        candidates = (self._failed_targets(toolkit, history, challenged)
                      + self._stale_family_targets(toolkit, history, challenged))
        if not candidates:
            return None
        candidates.sort(key=lambda t: (t.reason is None, -t.attempt.created_at))
        return candidates[0]

    def _explicit_target(self, toolkit, history):
        attempt = history.get(self.cfg.target)
        if attempt is None:
            return None
        code = safe_code(attempt)
        if code is None:
            return None
        return _Target(kind="attempt", attempt=attempt, code=code,
                       reason=self._stated_reason(toolkit, attempt),
                       direction=read_direction_from_attempt(attempt))

    def _failed_targets(self, toolkit, history, challenged):
        targets = []
        for a in history.list(only_done=False):
            if a.status != "fail" or a.id in challenged:
                continue
            if (a.metadata or {}).get("strategy") == self.name:
                continue
            if self._mechanical_failure(a):
                continue
            code = safe_code(a)
            if code is None:
                continue
            targets.append(_Target(kind="attempt", attempt=a, code=code,
                                   reason=self._stated_reason(toolkit, a),
                                   direction=read_direction_from_attempt(a)))
        return targets

    def _mechanical_failure(self, attempt):
        """A failure with no assumption to falsify: gate rejections and plain
        coding errors. Challenging one probes the machinery, not a belief."""
        tokens = [t.strip() for t in self.cfg.exclude_failures.split(",")
                  if t.strip()]
        if not tokens:
            return False
        if "gate_failure" in tokens and (attempt.metadata or {}).get("gate_failure"):
            return True
        result = safe_result(attempt)
        if result is None or result.completed or not result.failed_stage:
            return False
        stage = result.stages.get(result.failed_stage)
        if stage is None or not stage.errors:
            return False
        errors_text = str(stage.errors)
        return any(tok in errors_text for tok in tokens if tok != "gate_failure")

    def _stale_family_targets(self, toolkit, history, challenged):
        window = self.cfg.staleness_window
        targets = []
        for family in history.derive_families():
            if len(family) <= window:
                continue
            scores = [self._score_result(safe_result(a)) for a in family]
            if max(scores[-window:]) > max(scores[:-window]):
                continue
            leader = family[scores.index(max(scores))]
            if leader.id in challenged:
                continue
            code = safe_code(leader)
            if code is None:
                continue
            direction = read_direction_from_attempt(leader)
            targets.append(_Target(kind="family", attempt=leader, code=code,
                                   reason=self._reason_from_learnings(toolkit, leader, direction),
                                   direction=direction))
        return targets

    def _challenged_ids(self, history):
        ids = set()
        for a in history.list(only_done=False):
            target = (a.metadata or {}).get("challenge_target") or {}
            if target.get("attempt") is not None:
                ids.add(target["attempt"])
        return ids

    def _stated_reason(self, toolkit, attempt):
        meta = attempt.metadata or {}
        for key in ("blocker", "parked_reason", "reason", "verdict"):
            if meta.get(key):
                return str(meta[key])
        result = safe_result(attempt)
        if result is not None and not result.completed and result.failed_stage:
            stage = result.stages.get(result.failed_stage)
            if stage is not None and stage.errors:
                return f"failed at '{result.failed_stage}': {stage.errors}"
        return self._reason_from_learnings(toolkit, attempt,
                                           read_direction_from_attempt(attempt))

    def _reason_from_learnings(self, toolkit, attempt, direction):
        if not hasattr(toolkit, 'learnings'):
            return None
        text = toolkit.learnings.get() or ""
        needles = [f"#{attempt.id}"]
        if direction:
            title = direction_title(direction)
            if title and not title.startswith("("):
                needles.append(title)
        for line in text.splitlines():
            if any(needle in line for needle in needles):
                return line.strip()
        return None

    # --- Assumption extraction ---

    def _extract_assumption(self, toolkit, ws, target):
        learnings = toolkit.learnings.get(last=self.cfg.learnings_last, random=self.cfg.learnings_random) if hasattr(toolkit, 'learnings') else None

        parts = [
            "A direction in an iterative optimization run is failed or stalled. "
            "Your job is to name the BLOCKING ASSUMPTION — the claim that, if false, unblocks it.",
            f"## Task\n{toolkit.task.context.get()}",
            f"## Target under challenge ({target.kind} #{target.attempt.id})",
        ]
        if target.direction:
            parts.append(f"Core direction:\n{target.direction.strip()}")
        if target.reason:
            parts.append(f"Stated reason it is blocked/stalled:\n{target.reason}")
        parts.append(f"Recorded results:\n{self._describe_result(target.attempt)}")
        parts.append(f"## Target code\n```python\n{target.code}\n```")
        if learnings:
            parts.append(f"## Learnings\n{learnings}")
            from groundhog.utils.learnings_digest import record_learnings_used
            record_learnings_used(ws.path, learnings)
        parts.append(
            "Answer in exactly this format:\n"
            "BLOCKER: <the claimed blocker, one line>\n"
            "EVIDENCE: <verified|inherited> - <how the blocker was established; was it ever directly tested?>\n"
            "PLAN: <a concrete, minimal implementation that would falsify the blocker>"
        )
        prompt = "\n\n".join(parts)
        system_prompt = (
            "You are a skeptical research auditor. Blockers are often inherited "
            "beliefs, never directly tested. Diagnose precisely; propose the "
            "cheapest experiment that could prove the blocker wrong."
        )

        self.logger.log(UserEvent(content=prompt, data={"label": "Diagnosis"}))
        self.logger.log(SystemEvent(content=system_prompt))
        response = generate_text(toolkit.llm.get("default"), prompt, system_prompt,
                                 retries=self.cfg.max_retries)
        self.logger.log(AssistantEvent(content=response.text, role=response.model,
                                       cost=response.cost, usage=response.usage,
                                       data={"label": "Diagnosis"}))
        return self._parse_diagnosis(response.text)

    @staticmethod
    def _parse_diagnosis(text):
        fields = {}
        current = None
        # Tolerates markdown bolding: **BLOCKER**: and **BLOCKER:** alike.
        pattern = re.compile(
            r"^\s*\*{0,2}(BLOCKER|EVIDENCE|PLAN)\*{0,2}\s*:\s*\*{0,2}\s*(.*)$")
        for line in text.splitlines():
            m = pattern.match(line)
            if m:
                current = m.group(1).lower()
                fields[current] = m.group(2).strip()
            elif current is not None:
                fields[current] = (fields[current] + "\n" + line).strip()
        first_line = next((l.strip() for l in text.splitlines() if l.strip()), "unstated assumption")
        return {
            "blocker": fields.get("blocker") or first_line,
            "evidence": fields.get("evidence", ""),
            "plan": fields.get("plan", ""),
        }

    def _describe_result(self, attempt):
        result = safe_result(attempt)
        if result is None:
            return "(no recorded result)"
        lines = []
        if not result.completed:
            lines.append(f"FAILED at stage '{result.failed_stage}'.")
        for stage_name, stage_result in result.stages.items():
            if stage_result.errors:
                lines.append(f"  {stage_name}: FAILED — {stage_result.errors}")
            else:
                metrics = {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in stage_result.metrics.items()}
                lines.append(f"  {stage_name}: {metrics}")
        return "\n".join(lines) or "(no recorded stages)"

    # --- Workspace ---

    def _start_workspace(self, toolkit, target):
        parent = target.attempt.id if target.kind == "attempt" else None
        return toolkit.history.workspace(parent=parent)

    def _prepare_workspace(self, toolkit, ws, diagnosis):
        (ws.path / "TASK_CONTEXT.md").write_text(toolkit.task.context.get(), encoding="utf-8")
        blocker_line = diagnosis["blocker"].splitlines()[0]
        direction = f"Challenge: {blocker_line}"
        if diagnosis["plan"]:
            direction += f"\n\nFalsification plan:\n{diagnosis['plan']}"
        write_direction(ws.path, direction)

    # --- Attack ---

    def _attack(self, toolkit, ws, target, diagnosis):
        learnings = toolkit.learnings.get(last=self.cfg.learnings_last, random=self.cfg.learnings_random) if hasattr(toolkit, 'learnings') else None

        parts = [
            "A direction was declared blocked by the assumption below. Treat the "
            "assumption as a HYPOTHESIS and write a complete solution that attempts "
            "the falsification plan.",
            f"## Task\n{toolkit.task.context.get()}",
            f"## Claimed blocker (attack this)\n{diagnosis['blocker']}",
        ]
        if diagnosis["evidence"]:
            parts.append(f"## Evidence for the blocker\n{diagnosis['evidence']}")
        if diagnosis["plan"]:
            parts.append(f"## Falsification plan\n{diagnosis['plan']}")
        parts.append(f"## Code that hit the blocker\n```python\n{target.code}\n```")
        if learnings:
            parts.append(f"## Learnings\n{learnings}")
            from groundhog.utils.learnings_digest import record_learnings_used
            record_learnings_used(ws.path, learnings)
        prompt = "\n\n".join(parts)

        system_prompt = (
            "You are an expert programmer running a falsification probe. The "
            "blocker is a claim, not a fact — implement the plan that tests it. "
            "Write complete, runnable code in a ```python block."
        )

        self.logger.log(UserEvent(content=prompt))
        self.logger.log(SystemEvent(content=system_prompt))
        response = generate_text(toolkit.llm.get("high"), prompt, system_prompt,
                                 retries=self.cfg.max_retries)
        self.logger.log(AssistantEvent(content=response.text, role=response.model,
                                       cost=response.cost, usage=response.usage))

        code, _ = extract_code(response.text)
        if code:
            (ws.path / "solution.py").write_text(code, encoding="utf-8")

    # --- Evaluation with retries ---

    def _evaluate(self, toolkit, ws):
        for attempt_num in range(self.cfg.max_retries + 1):
            if not (ws.path / "solution.py").exists():
                (ws.path / "solution.py").write_text("# no code generated", encoding="utf-8")
            result = toolkit.task.evaluate(ws.path, through=self.through)
            self.logger.log(eval_event(result, self._score_result(result)))

            if result.completed:
                return result

            if attempt_num < self.cfg.max_retries:
                error_stage = result.stages[result.failed_stage]
                code = (ws.path / "solution.py").read_text(encoding="utf-8")
                self.log.inline(f"retry {attempt_num + 1}... ")
                self._retry_fix(toolkit, ws, code, error_stage, attempt_num + 1)

        return result

    def _retry_fix(self, toolkit, ws, broken_code, error_stage, retry_num):
        error_context = f"Attempt {retry_num} failed with errors: {error_stage.errors}"
        prompt = build_prompt(
            context=toolkit.task.context.get(),
            prior_code=broken_code,
            mode="diff",
        )
        prompt += f"\n\nERROR — PLEASE FIX:\n{error_context}"
        system_prompt = "The code has errors. Fix them using SEARCH/REPLACE blocks."

        self.logger.log(UserEvent(content=prompt, data={"label": f"Retry {retry_num}"}))
        self.logger.log(SystemEvent(content=system_prompt))

        try:
            response = generate_text(toolkit.llm.get("default"), prompt, system_prompt)
        except GenerationFailed:
            return  # leave the broken code; the eval loop records the failure
        self.logger.log(AssistantEvent(content=response.text, role=response.model,
                                       cost=response.cost, usage=response.usage, data={"label": f"Retry {retry_num}"}))

        fixed_code, _ = extract_code(response.text, broken_code)
        if fixed_code:
            (ws.path / "solution.py").write_text(fixed_code, encoding="utf-8")

    # --- Learnings ---

    def _record_learnings(self, toolkit, target, diagnosis, result):
        if not hasattr(toolkit, 'learnings'):
            return

        if result.completed:
            outcome = f"completed with score {self._score_result(result):.4f}"
        else:
            stage = result.stages.get(result.failed_stage)
            outcome = f"failed at '{result.failed_stage}': {stage.errors if stage else 'unknown'}"

        prompt = (
            "A falsification challenge was run against a claimed blocker.\n\n"
            f"CLAIMED BLOCKER: {diagnosis['blocker']}\n"
            f"EVIDENCE CLASS: {diagnosis['evidence'] or 'unstated'}\n"
            f"FALSIFICATION PLAN: {diagnosis['plan'] or 'unstated'}\n\n"
            f"CHALLENGE OUTCOME: the probe {outcome} "
            f"(challenged {target.kind} #{target.attempt.id}).\n\n"
            "Write 1-2 bullet points about what was learned. State EXPLICITLY "
            "whether the assumption SURVIVED the challenge or FELL, and why. "
            "Keep it short — these notes guide future attempts."
        )
        system_prompt = "You are a concise research assistant. Write brief, actionable observations."

        self.logger.log(UserEvent(content=prompt, data={"label": "Learnings"}))
        try:
            response = generate_text(toolkit.llm.get("default"), prompt, system_prompt)
        except GenerationFailed:
            return  # a missing learning must not sink an already-evaluated attempt
        self.logger.log(AssistantEvent(content=response.text, role=response.model,
                                       cost=response.cost, usage=response.usage, data={"label": "Learnings"}))

        toolkit.learnings.add(response.text)

    # --- Finalization ---

    def _finalize(self, toolkit, ws, result, target, diagnosis):
        """The standard finish: direction gates -> record -> commit -> score note."""
        from groundhog.utils.finalize import finalize_attempt
        metadata = {
            "strategy": self.name,
            "prior": target.attempt.id if target.kind == "attempt" else None,
            "challenge_target": {"kind": target.kind, "attempt": target.attempt.id,
                                 "reason": target.reason},
            "challenge_assumption": diagnosis,
            "cost": round(self.logger.total_cost(), 6),
        }
        # prior=None on purpose: a challenge mints a NEW direction even when the
        # workspace is parented on the target — inheriting would clobber the
        # challenge direction, and the fresh gates (direction present, not a
        # duplicate) are the right legitimacy checks here.
        return finalize_attempt(toolkit, ws, result, None, metadata=metadata)

    def _finalize_failed(self, toolkit, ws, reason):
        """Generation died — record the attempt as a failure, never orphan it."""
        from groundhog.utils.finalize import finalize_failed
        self.logger.log(LogEvent(type="error", data={"error": reason}))
        metadata = {"strategy": self.name,
                    "cost": round(self.logger.total_cost(), 6),
                    "generation_failed": reason}
        return finalize_failed(toolkit, ws, reason, None, metadata=metadata)

    # --- Scoring ---

    def _score_result(self, result):
        if result is None or not result.completed:
            return -1.0
        final_stage = self._stages[-1]
        final_result = result.stages.get(final_stage.name)
        return final_stage.score(final_result) if final_result else -1.0

    # --- Logging ---

    def _build_log(self, attempt, target, diagnosis, result):
        return {
            "attempt": attempt.id,
            "target": target.attempt.id,
            "kind": target.kind,
            "assumption": diagnosis["blocker"],
            "score": round(self._score_result(result), 4),
            "strategy": self.name,
        }
