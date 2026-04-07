"""CopilotAgentBackend — runs GitHub Copilot CLI as a subprocess with tool access.

Copilot charges per "premium request" (fixed cost per call), not per token.
The strategy adapts: one big explore call instead of multi-phase.

Key differences from ClaudeCodeAgentBackend:
- Permission flags: --allow-tool/--deny-tool (not --allowedTools/--disallowedTools)
- Permission syntax: shell(cmd:*) (not Bash(cmd:*)), write (not Write(*))
- Output format: --output-format json (not stream-json)
- Effort: --reasoning-effort (not --effort)
- No --max-budget-usd (fixed cost per request)
- Session ID from result event (can't pre-assign)
- Event types: assistant.message, tool.execution_complete, etc.
"""

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from groundhog.base.agent import AgentBackend, AgentResult, AgentSpec
from groundhog.agents.tool_server import (
    ToolServer,
    build_tool_docs,
    cleanup_wrappers,
    generate_wrappers,
)


def _translate_permission(rule: str) -> str:
    """Translate Claude-style permission to Copilot-style.

    Bash(cmd:*) → shell(cmd:*)
    Bash(cmd)   → shell(cmd)
    Write(*)    → write
    Write(path) → write(path)
    Edit(*)     → write
    Edit(path)  → write(path)
    Read(*)     → read
    Read(path)  → read(path)
    """
    # Bash(...) → shell(...)
    m = re.match(r'Bash\((.+)\)', rule)
    if m:
        return f"shell({m.group(1)})"

    # Write/Edit(*) → write, Write/Edit(path) → write(path)
    m = re.match(r'(?:Write|Edit)\((.+)\)', rule)
    if m:
        inner = m.group(1)
        return "write" if inner == "*" else f"write({inner})"

    # Read(*) → read, Read(path) → read(path)
    m = re.match(r'Read\((.+)\)', rule)
    if m:
        inner = m.group(1)
        return "read" if inner == "*" else f"read({inner})"

    # Pass through unknown rules
    return rule


class CopilotAgentBackend(AgentBackend):
    """Agent backend that runs the copilot CLI as a subprocess.

    Tools are exposed via HTTP tool server + bash wrapper scripts.
    Session resume via --resume=<session_id> (ID from result event).
    """
    cost_model = "per_request"

    def __init__(self, model: str = "claude-sonnet-4.6", effort: str = "high"):
        self.model = model
        self.effort = effort

    def run(self, spec: AgentSpec) -> AgentResult:
        server = None
        bin_dir = None
        try:
            bin_dir = Path(tempfile.mkdtemp(prefix="copilot_tools_"))
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
        # Ensure tool wrapper scripts use UTF-8 on Windows (default cp1252 crashes
        # on unicode characters like em dashes or degree symbols in output)
        if os.name == "nt":
            env.setdefault("PYTHONIOENCODING", "utf-8")
        env.update(spec.env)
        return env

    def _build_prompt(self, spec: AgentSpec) -> str:
        """Augment goal with tool documentation."""
        docs = build_tool_docs(spec.tools)
        if docs:
            return spec.goal + "\n\n" + docs
        return spec.goal

    # Copilot builtin tools that the agent may need
    BUILTIN_TOOLS = [
        "view", "edit", "create", "glob", "grep", "powershell",
        "report_intent", "task_complete",
    ]

    def _build_command(self, spec: AgentSpec) -> list:
        model = spec.model or self.model
        effort = spec.effort or self.effort
        prompt = self._build_prompt(spec)

        cmd = [
            "copilot", "-p", prompt,
            "--output-format", "json",
            "--model", model,
            "--reasoning-effort", effort,
            "--autopilot",
        ]

        if spec.session_id:
            cmd += ["--resume", spec.session_id]

        # Whitelist available tools: copilot builtins + tool-server wrappers.
        # This controls what tools the model can SEE.
        available = list(self.BUILTIN_TOOLS)
        for tool in spec.tools:
            available.append(tool.name)
        cmd += ["--available-tools", ",".join(available)]

        # --allow-all-tools is required for non-interactive (pipe/json) mode.
        # Individual --allow-tool flags only suppress the confirmation prompt
        # in interactive mode; in pipe mode tools silently don't execute.
        cmd.append("--allow-all-tools")

        # Deny rules for specific files (e.g. solution.py during explore phase).
        # Only translate path-specific denies — blanket denies like Write(*)
        # would block all writes since copilot doesn't support "deny broad,
        # allow specific" patterns.
        for rule in spec.denied_tools:
            translated = _translate_permission(rule)
            # Skip blanket denies — they'd override --allow-all-tools
            if translated in ("write", "read", "shell"):
                continue
            cmd += ["--deny-tool", translated]

        return cmd

    def _run_subprocess(self, cmd: list, env: dict, spec: AgentSpec) -> List[dict]:
        """Run subprocess with JSON output, writing events live to workspace."""
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

        events = []
        try:
            with open(jsonl_path, "a") as raw_file, open(summary_path, "a") as summary_file:
                # Don't write initial prompt — copilot emits user.message with it

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
            raise RuntimeError(f"copilot exited with code {proc.returncode}")

        return events

    def _parse_result(self, events: List[dict]) -> AgentResult:
        """Parse copilot NDJSON events into AgentResult."""
        data = None
        for event in reversed(events):
            if event.get("type") == "result":
                data = event
                break

        if data is None:
            # Collect text from assistant.message events
            text_parts = []
            for event in events:
                if event.get("type") == "assistant.message":
                    content = event.get("data", {}).get("content", "")
                    if content:
                        text_parts.append(content)
            combined = " ".join(text_parts)
            return AgentResult(success=True, output=combined.strip(),
                               steps=_extract_steps(events))

        usage = data.get("usage", {})
        return AgentResult(
            success=data.get("exitCode", 0) == 0,
            output="",  # copilot result event doesn't have output text
            session_id=data.get("sessionId"),
            cost=usage.get("premiumRequests", 0) * 0.0333,  # $10/300 requests
            turns=0,  # count from events
            duration_ms=usage.get("sessionDurationMs", 0),
            error=None if data.get("exitCode", 0) == 0 else f"exit code {data.get('exitCode')}",
            steps=_extract_steps(events),
        )


# --- Event parsing ---

def _summarize_event(event: dict) -> List[dict]:
    """Convert copilot NDJSON event to clean summary lines.

    Copilot event schema (from actual CLI output):
    - user.message:              data.content (the prompt)
    - assistant.message:         data.content + data.toolRequests[]
      toolRequests[]:            .name, .arguments, .toolCallId, .type
    - tool.execution_start:      data.toolCallId, data.toolName, data.arguments
    - tool.execution_complete:   data.toolCallId, data.result.content
    - assistant.message_delta:   ephemeral streaming
    - assistant.reasoning:       ephemeral thinking
    - assistant.turn_start/end:  turn boundaries
    - session.*:                 session management (ephemeral)
    - result:                    sessionId, exitCode, usage{}
    """
    event_type = event.get("type")
    data = event.get("data", {})

    if event_type == "user.message":
        content = data.get("content", data.get("transformedContent", ""))
        if content:
            return [{"role": "user", "content": content}]
        return []

    if event_type == "assistant.message":
        lines = []
        content = data.get("content", "")
        if content and content.strip():
            lines.append({"role": "assistant", "type": "text", "content": content})
        for req in data.get("toolRequests", []):
            lines.append({
                "role": "assistant",
                "type": "tool_use",
                "tool": req.get("name", "unknown"),
                "input": req.get("arguments", {}),
            })
        return lines

    if event_type == "tool.execution_complete":
        result = data.get("result", {})
        content = result.get("content", "") if isinstance(result, dict) else str(result)
        return [{
            "role": "tool_result",
            "tool_use_id": data.get("toolCallId"),
            "content": content,
        }]

    if event_type == "result":
        usage = event.get("usage", {})
        return [{
            "role": "result",
            "session_id": event.get("sessionId"),
            "duration_ms": usage.get("sessionDurationMs"),
            "premium_requests": usage.get("premiumRequests"),
        }]

    # Skip ephemeral events: deltas, reasoning, session.*, turn boundaries
    return []


def _extract_steps(events: List[dict]) -> List[dict]:
    """Extract compact step summaries from copilot NDJSON events."""
    MAX_TEXT = 500
    steps = []
    pending_tools: Dict[str, int] = {}

    for event in events:
        event_type = event.get("type")
        data = event.get("data", {})

        if event_type == "assistant.message":
            content = data.get("content", "")
            if content and content.strip():
                steps.append({
                    "type": "text",
                    "text": content[:MAX_TEXT] + ("..." if len(content) > MAX_TEXT else ""),
                })
            for req in data.get("toolRequests", []):
                tool_input = req.get("arguments", {})
                compact_input = {
                    k: (v[:MAX_TEXT] + "..." if isinstance(v, str) and len(v) > MAX_TEXT else v)
                    for k, v in tool_input.items()
                } if isinstance(tool_input, dict) else {}
                step = {
                    "type": "tool_use",
                    "tool": req.get("name", "unknown"),
                    "input": compact_input,
                }
                steps.append(step)
                req_id = req.get("toolCallId")
                if req_id:
                    pending_tools[req_id] = len(steps) - 1

        elif event_type == "tool.execution_complete":
            req_id = data.get("toolCallId")
            result = data.get("result", {})
            output = result.get("content", "") if isinstance(result, dict) else str(result)
            truncated = (output[:MAX_TEXT] + "...") if isinstance(output, str) and len(output) > MAX_TEXT else str(output)[:MAX_TEXT]

            if req_id and req_id in pending_tools:
                idx = pending_tools.pop(req_id)
                steps[idx]["output"] = truncated
            else:
                steps.append({"type": "tool_result", "output": truncated})

    return steps
