r"""CodexCliAgentBackend — runs OpenAI's Codex CLI as a subprocess with tool access.

Uses ``codex exec --json`` for one-shot non-interactive runs. Streams
JSONL events; captures the session id from the first ``thread.started``
event for resume; sums token usage from ``turn.completed`` events.

Sandboxing
----------
``--sandbox workspace-write`` blocks network and blocks writes to
absolute system paths (verified on Windows: ``C:\\tmp\\...`` returns
access-denied). **However**, the writable area is broader than the
``cwd`` we set — empirically, writes to ``../sibling/`` succeed even
with ``sandbox_workspace_write.writable_roots=[]`` and even when no
``.git/`` exists at any level. There's no observed config knob on
Windows that tightens the writable area to ``cwd``-only short of
switching to ``-s read-only`` (which then blocks legitimate writes
inside ``cwd`` too). Implication for parallel optimization: an agent
in ``attempts/NNN_M/`` can write to ``attempts/<sibling>/`` — caller
must isolate each attempt's parent directory if that matters.
``-c approval_policy=never`` makes the run non-interactive.

**Codex does not restrict reads** in workspace-write mode — the agent
can `cat` any file the user has read access to. There is no native
allow/deny pattern equivalent to claude's ``--allowedTools`` /
``--disallowedTools``. Like the Gemini adapter,
``spec.allowed_tools`` / ``spec.denied_tools`` are surfaced as
**advisory prompt text** only. The hard floor is the sandbox flag.

Tool exposure
-------------
The HTTP tool server + multi-format wrappers (``.ps1`` / ``.cmd`` /
``.py`` / extensionless bash) are written to a ``%TEMP%`` bin dir,
matching the convention the other backends use. Two flags make this
work inside codex's sandbox on Windows:

- ``--add-dir <bin_dir>`` grants the sandbox visibility into the
  wrapper directory so PowerShell can resolve ``.cmd`` / ``.ps1``
  files via PATHEXT lookup.
- ``-c shell_environment_policy.inherit=all`` propagates our parent
  ``PATH`` (with ``bin_dir`` prepended) into the spawned shell.

The HTTP tool server (port in ``TOOL_SERVER_PORT`` env) also works
as a fallback that the agent can call via ``Invoke-RestMethod``.

Cost
----
Codex emits token usage in ``turn.completed.usage`` but does **not**
report USD. ``AgentResult.cost`` is left at ``0.0``; per-token rates
are model-specific and not bundled. Track tokens externally if needed.
``budget_usd`` is accepted for API parity but **not enforced** — codex
has no equivalent of ``--max-budget-usd``.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional



def _resolve_codex_bin() -> str:
    """Locate the codex executable.

    On Windows, npm installs ``codex.cmd`` / ``codex.ps1`` rather than
    ``codex.exe``; ``subprocess.Popen(["codex", ...])`` then fails with
    WinError 2 because Popen doesn't auto-walk PATHEXT in all shell
    environments. Resolve to a concrete path up front.
    """
    return shutil.which("codex") or "codex"

from groundhog.base.agent import AgentBackend, AgentResult, AgentSpec
from groundhog.agents.tool_server import (
    ToolServer,
    build_tool_docs,
    cleanup_wrappers,
    generate_wrappers,
)


class CodexCliAgentBackend(AgentBackend):
    """Agent backend that runs the codex CLI as a subprocess.

    Tools exposed via HTTP tool server + bash wrappers on PATH.
    Resume via ``codex exec resume <session_id>``.
    """
    cost_model = "per_token"

    def __init__(self,
                 model: Optional[str] = None,
                 effort: str = "medium",
                 sandbox: str = "workspace-write",
                 max_budget_usd: Optional[float] = None):
        self.model = model
        self.effort = effort
        self.sandbox = sandbox
        # Codex doesn't enforce USD budgets; kept for API parity.
        self.max_budget_usd = max_budget_usd

    def run(self, spec: AgentSpec) -> AgentResult:
        server = None
        bin_dir = None
        try:
            # Wrappers live in a %TEMP% bin dir, matching the other backends.
            # Codex's sandbox normally hides %TEMP% from the spawned shell,
            # so we pass ``--add-dir <bin_dir>`` (see _build_command) to
            # grant the sandbox visibility into the wrapper directory.
            bin_dir = Path(tempfile.mkdtemp(prefix="codex_tools_"))
            server = self._start_tool_server(spec)
            port = server.port if server else None
            if spec.tools and port is not None:
                generate_wrappers(spec.tools, bin_dir, port)
            env = self._build_env(spec, bin_dir, port)
            cmd = self._build_command(spec, bin_dir=bin_dir)
            events = self._run_subprocess(cmd, env, spec)
            return self._parse_result(events)
        except TimeoutError as e:
            return AgentResult(success=False, output="", error=str(e))
        except Exception as e:
            return AgentResult(success=False, output="", error=str(e))
        finally:
            if server:
                server.stop()
            if bin_dir is not None:
                cleanup_wrappers(bin_dir)

    def _start_tool_server(self, spec: AgentSpec) -> Optional[ToolServer]:
        if not spec.tools:
            return None
        server = ToolServer(spec.tools)
        server.start()
        return server

    def _build_env(self, spec: AgentSpec, bin_dir: Path, port: Optional[int]) -> dict:
        env = os.environ.copy()
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
        if port is not None:
            env["TOOL_SERVER_PORT"] = str(port)
        if os.name == "nt":
            env.setdefault("PYTHONIOENCODING", "utf-8")
        env.update(spec.env)
        return env

    def _build_prompt(self, spec: AgentSpec) -> str:
        """Goal + tool docs + the resolved permission policy.

        Codex has no native allow/deny tool flags, so the policy is
        advisory text. Include BOTH allow and deny lists (with the
        narrow-allow-overrides-broad-deny explanation) — without the
        allow side, models see ``Write(*)`` deny and refuse all writes
        including the ``Write(work/*)`` ones the strategy permits.
        """
        from groundhog.agents.gemini_cli import _format_permission_policy

        prompt = spec.goal
        docs = build_tool_docs(spec.tools)
        if docs:
            prompt += "\n\n" + docs
        if spec.allowed_tools or spec.denied_tools:
            prompt += "\n\n" + _format_permission_policy(
                spec.allowed_tools, spec.denied_tools
            )
        return prompt

    def _resolve_prompt(self, spec: AgentSpec) -> str:
        """Pick the prompt text for the current run. On resume the goal
        is the follow-up; on first turn it's the full prompt with tool
        docs + denies attached."""
        if spec.session_id:
            return spec.goal
        return self._build_prompt(spec)

    def _build_command(self, spec: AgentSpec, bin_dir: Optional[Path] = None) -> list:
        model = spec.model or self.model
        effort = spec.effort or self.effort

        # Common flags for both new-session and resume paths. ``--add-dir``
        # exposes the %TEMP% wrapper bin to the sandbox; without it,
        # ``pwsh -Command <wrapper>`` reports "not recognized" because the
        # sandbox hides paths outside cwd from the spawned shell.
        # ``shell_environment_policy.inherit=all`` propagates our PATH
        # (which has bin_dir prepended) into the spawned shell's env.
        # ``project_root_markers=[]`` disables codex's walk-up search for
        # ``.git`` (etc.) when computing the sandbox workspace boundary —
        # without this, codex uses the enclosing repo as the project root
        # and writes to ``../sibling/`` succeed because they're inside the
        # repo. Anchoring the boundary at cwd is what we want for parallel
        # attempts that share a parent directory.
        common = [
            _resolve_codex_bin(), "exec",
            "--json",
            "--skip-git-repo-check",
            "-s", self.sandbox,
            "-c", "approval_policy=never",
            "-c", "shell_environment_policy.inherit=all",
            "-c", "project_root_markers=[]",
        ]
        if bin_dir is not None:
            common += ["--add-dir", str(bin_dir)]
        if model:
            common += ["-m", model]
        if effort:
            common += ["-c", f"model_reasoning_effort={effort}"]

        # Prompt is fed via stdin (handled in _run_subprocess) — passing it
        # as a positional argv on Windows truncates at the first newline
        # because the codex.cmd npm wrapper splits on \n. Using ``-`` reads
        # from stdin per the codex-exec docs.
        if spec.session_id:
            return common + ["resume", spec.session_id, "-"]
        return common + ["-"]

    def _run_subprocess(self, cmd: list, env: dict, spec: AgentSpec) -> List[dict]:
        """Run subprocess with --json, writing events live to workspace."""
        jsonl_path = spec.workspace_path / "agent_steps.jsonl"
        summary_path = spec.workspace_path / "agent_summary.jsonl"
        deadline = time.monotonic() + spec.timeout if spec.timeout else None

        # encoding="utf-8" + errors="replace" — same Windows fix as the
        # copilot adapter. Default cp1252 locale chokes on JSON bytes
        # codex emits (observed: 0x9d at position 139 → UnicodeDecodeError).
        # stdin=PIPE so we can feed the prompt (avoids argv newline
        # truncation on Windows via the codex.cmd npm wrapper).
        prompt = self._resolve_prompt(spec)
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(spec.workspace_path),
            env=env,
        )
        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
        except (OSError, BrokenPipeError):
            pass

        events: List[dict] = []
        try:
            with open(jsonl_path, "a", encoding="utf-8") as raw_file, \
                 open(summary_path, "a", encoding="utf-8") as summary_file:
                # Mirror the initial prompt — codex doesn't echo it back.
                initial = {
                    "type": "user",
                    "subtype": "initial_prompt",
                    "message": {"role": "user", "content": spec.goal},
                }
                raw_file.write(json.dumps(initial) + "\n")
                raw_file.flush()
                summary_file.write(json.dumps({"role": "user", "content": spec.goal}) + "\n")
                summary_file.flush()

                for line in proc.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    events.append(event)
                    raw_file.write(json.dumps(event) + "\n")
                    raw_file.flush()

                    for summary_line in _summarize_event(event):
                        summary_file.write(json.dumps(summary_line) + "\n")
                    summary_file.flush()

                    if spec.on_event:
                        try:
                            spec.on_event(event)
                        except Exception:
                            pass

                    if deadline and time.monotonic() > deadline:
                        proc.kill()
                        raise TimeoutError(f"Agent timed out after {spec.timeout}s")

            proc.wait()
        except TimeoutError:
            raise
        except Exception:
            proc.kill()
            proc.wait()
            raise

        if proc.returncode != 0 and not events:
            raise RuntimeError(f"codex exited with code {proc.returncode}")

        return events

    def _parse_result(self, events: List[dict]) -> AgentResult:
        """Reduce JSONL stream into a single AgentResult.

        Field map:
          session_id  ← first ``thread.started.thread_id``
          output      ← text of last ``item.completed`` of type ``agent_message``
          turns       ← count of ``turn.completed`` events
          success     ← no ``turn.failed`` and no top-level ``error`` event
          cost        ← 0.0 (codex doesn't report USD; track tokens externally)
        """
        session_id = None
        output_text = ""
        turns = 0
        failed_msg: Optional[str] = None

        for ev in events:
            etype = ev.get("type")
            if etype == "thread.started" and session_id is None:
                session_id = ev.get("thread_id")
            elif etype == "turn.completed":
                turns += 1
            elif etype == "turn.failed":
                err = ev.get("error", {})
                failed_msg = err.get("message") if isinstance(err, dict) else str(err)
            elif etype == "error":
                failed_msg = ev.get("message", "codex error")
            elif etype == "item.completed":
                item = ev.get("item", {})
                if item.get("type") == "agent_message":
                    text = item.get("text", "")
                    if text:
                        output_text = text  # last one wins

        success = failed_msg is None
        return AgentResult(
            success=success,
            output=output_text,
            session_id=session_id,
            cost=0.0,
            turns=turns,
            duration_ms=0,
            error=failed_msg,
            steps=_extract_steps(events),
        )


# --- Event parsing (module-level) ---

def _summarize_event(event: dict) -> List[dict]:
    """Convert a Codex JSONL event into clean summary lines.

    Codex event shape:
      thread.started     → {thread_id}
      turn.started       → {}
      turn.completed     → {usage: {...token counts}}
      turn.failed        → {error: {message}}
      item.started/updated/completed → {item: {id, type, ...details}}
      error              → {message}
    """
    etype = event.get("type")

    if etype == "thread.started":
        return [{"role": "system", "type": "thread_started",
                 "thread_id": event.get("thread_id")}]

    if etype == "turn.completed":
        usage = event.get("usage", {}) or {}
        return [{
            "role": "result",
            "type": "turn_completed",
            "input_tokens": usage.get("input_tokens"),
            "cached_input_tokens": usage.get("cached_input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
        }]

    if etype == "turn.failed":
        err = event.get("error", {}) or {}
        return [{"role": "result", "type": "turn_failed",
                 "error": err.get("message", "")}]

    if etype == "error":
        return [{"role": "result", "type": "error",
                 "error": event.get("message", "")}]

    if etype == "item.completed":
        item = event.get("item", {}) or {}
        itype = item.get("type")
        if itype == "agent_message":
            text = item.get("text", "")
            if text.strip():
                return [{"role": "assistant", "type": "text", "content": text}]
        elif itype == "reasoning":
            text = item.get("text", "")
            if text.strip():
                return [{"role": "assistant", "type": "thinking", "content": text}]
        elif itype == "command_execution":
            return [{
                "role": "assistant",
                "type": "tool_use",
                "tool": "command_execution",
                "input": {"command": item.get("command", "")},
                "output": (item.get("aggregated_output", "") or "")[:500],
                "exit_code": item.get("exit_code"),
                "status": item.get("status"),
            }]
        elif itype == "file_change":
            return [{
                "role": "assistant",
                "type": "tool_use",
                "tool": "file_change",
                "input": {"changes": item.get("changes", [])},
                "status": item.get("status"),
            }]
        elif itype == "mcp_tool_call":
            return [{
                "role": "assistant",
                "type": "tool_use",
                "tool": f"mcp:{item.get('server','?')}/{item.get('tool','?')}",
                "input": item.get("arguments", {}),
                "output": str(item.get("result", ""))[:500],
                "status": item.get("status"),
            }]
        elif itype == "error":
            return [{"role": "assistant", "type": "error",
                     "content": item.get("message", "")}]

    return []


def _extract_steps(events: List[dict]) -> List[dict]:
    """Compact step summaries derived from Codex item.completed events."""
    MAX_TEXT = 500
    steps: List[dict] = []

    for ev in events:
        if ev.get("type") != "item.completed":
            continue
        item = ev.get("item", {}) or {}
        itype = item.get("type")

        if itype == "agent_message":
            text = item.get("text", "")
            if text.strip():
                steps.append({
                    "type": "text",
                    "text": text[:MAX_TEXT] + ("..." if len(text) > MAX_TEXT else ""),
                })

        elif itype == "command_execution":
            cmd = item.get("command", "")
            out = item.get("aggregated_output", "") or ""
            steps.append({
                "type": "tool_use",
                "tool": "shell",
                "input": {"command": cmd[:MAX_TEXT] + ("..." if len(cmd) > MAX_TEXT else "")},
                "output": out[:MAX_TEXT] + ("..." if len(out) > MAX_TEXT else ""),
                "exit_code": item.get("exit_code"),
                "status": item.get("status"),
            })

        elif itype == "file_change":
            steps.append({
                "type": "tool_use",
                "tool": "file_change",
                "input": {"changes": item.get("changes", [])},
                "status": item.get("status"),
            })

        elif itype == "mcp_tool_call":
            args = item.get("arguments", {})
            res = item.get("result", "")
            steps.append({
                "type": "tool_use",
                "tool": f"mcp:{item.get('server','?')}/{item.get('tool','?')}",
                "input": args,
                "output": str(res)[:MAX_TEXT],
                "status": item.get("status"),
            })

    return steps
