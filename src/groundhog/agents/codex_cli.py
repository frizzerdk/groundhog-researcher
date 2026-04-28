"""CodexCliAgentBackend — runs OpenAI's Codex CLI as a subprocess with tool access.

Uses ``codex exec --json`` for one-shot non-interactive runs. Streams
JSONL events; captures the session id from the first ``thread.started``
event for resume; sums token usage from ``turn.completed`` events.

Sandboxing
----------
``--sandbox workspace-write`` confines **writes** to the agent's cwd
(filesystem floor enforced by the OS — verified on Windows: writes to
absolute system paths return "access denied"; writes inside the
attempt directory tree succeed) and blocks network by default.
``-c approval_policy=never`` makes the run non-interactive.

**Codex does not restrict reads** in workspace-write mode — the agent
can `cat` any file the user has read access to. There is no native
allow/deny pattern equivalent to claude's ``--allowedTools`` /
``--disallowedTools``. Like the Gemini adapter,
``spec.allowed_tools`` / ``spec.denied_tools`` are surfaced as
**advisory prompt text** only. The hard floor is the sandbox flag.

Tool exposure on Windows
-------------------------
The same HTTP tool-server + PATH-injected bash/cmd wrappers used by the
Copilot adapter are wired up here, plus
``-c shell_environment_policy.inherit=all`` to forward our PATH into
the codex subprocess. **Caveat**: in practice, codex's spawned
PowerShell (the default shell on Windows) does not always resolve
unqualified wrapper names like ``smoke`` — observed in e2e: half of
``pwsh -Command <wrapper>`` invocations report "not recognized" even
with PATH inheritance enabled. Native binaries (``cmd``, ``Get-*``)
work fine. Tracking; works around itself today by the agent reading
the source files directly when wrappers fail.

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
            bin_dir = Path(tempfile.mkdtemp(prefix="codex_tools_"))
            server = self._start_tool_server(spec)
            port = server.port if server else None
            if spec.tools and port is not None:
                generate_wrappers(spec.tools, bin_dir, port)
            env = self._build_env(spec, bin_dir, port)
            cmd = self._build_command(spec)
            events = self._run_subprocess(cmd, env, spec)
            return self._parse_result(events)
        except TimeoutError as e:
            return AgentResult(success=False, output="", error=str(e))
        except Exception as e:
            return AgentResult(success=False, output="", error=str(e))
        finally:
            if server:
                server.stop()
            if bin_dir:
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
        """Goal + tool docs + (advisory) deny rules. Codex has no native
        deny-pattern flag, so the prompt is the only place denies show up."""
        prompt = spec.goal
        docs = build_tool_docs(spec.tools)
        if docs:
            prompt += "\n\n" + docs
        if spec.denied_tools:
            lines = ["\n\n## Restrictions",
                     "You MUST NOT use the following tools or actions:"]
            for rule in spec.denied_tools:
                lines.append(f"- {rule}")
            prompt += "\n".join(lines)
        return prompt

    def _build_command(self, spec: AgentSpec) -> list:
        model = spec.model or self.model
        effort = spec.effort or self.effort

        if spec.session_id:
            # Resume an existing thread. The follow-up prompt is short — no
            # need to re-attach tool docs (the tools are still on PATH).
            cmd = [
                _resolve_codex_bin(), "exec",
                "--json",
                "--skip-git-repo-check",
                "-s", self.sandbox,
                "-c", "approval_policy=never",
                # Inherit parent PATH (etc.) so the agent's shell can find
                # bash wrappers we put on PATH for tool exposure. Without
                # this, codex's shell_environment_policy strips PATH and
                # tools become invisible.
                "-c", "shell_environment_policy.inherit=all",
            ]
            if model:
                cmd += ["-m", model]
            if effort:
                cmd += ["-c", f"model_reasoning_effort={effort}"]
            cmd += ["resume", spec.session_id, spec.goal]
        else:
            prompt = self._build_prompt(spec)
            cmd = [
                _resolve_codex_bin(), "exec",
                "--json",
                "--skip-git-repo-check",
                "-s", self.sandbox,
                "-c", "approval_policy=never",
                # Inherit parent PATH (etc.) so the agent's shell can find
                # bash wrappers we put on PATH for tool exposure. Without
                # this, codex's shell_environment_policy strips PATH and
                # tools become invisible.
                "-c", "shell_environment_policy.inherit=all",
            ]
            if model:
                cmd += ["-m", model]
            if effort:
                cmd += ["-c", f"model_reasoning_effort={effort}"]
            cmd += [prompt]
        return cmd

    def _run_subprocess(self, cmd: list, env: dict, spec: AgentSpec) -> List[dict]:
        """Run subprocess with --json, writing events live to workspace."""
        jsonl_path = spec.workspace_path / "agent_steps.jsonl"
        summary_path = spec.workspace_path / "agent_summary.jsonl"
        deadline = time.monotonic() + spec.timeout if spec.timeout else None

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            cwd=str(spec.workspace_path),
            env=env,
        )

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
