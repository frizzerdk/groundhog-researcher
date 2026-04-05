"""GeminiCliAgentBackend — runs Gemini CLI as a subprocess with tool access.

Mirrors ClaudeCodeAgentBackend's structure, adapted for Gemini CLI's flags
and NDJSON event schema.

Key differences from Claude Code:
- Deny rules encoded in prompt (no --disallowedTools flag)
- Turn limits via temp .gemini/settings.json
- Session ID returned in result event (can't pre-assign)
- Different NDJSON event types: message, tool_use, tool_result (standalone)

Ported from EvaluatableExperiments/src/agents/implementations/gemini_cli.py.
"""

import json
import os
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


class GeminiCliAgentBackend(AgentBackend):
    """Agent backend that runs the gemini CLI as a subprocess.

    Tools are exposed via HTTP tool server + bash wrapper scripts.
    Session resume: session_id returned in result event, passed back via
    spec.session_id on subsequent calls.
    """

    def __init__(self, model: str = "gemini-2.5-flash",
                 max_turns: Optional[int] = None,
                 sandbox: bool = False,
                 approval_mode: str = "yolo"):
        self.model = model
        self.max_turns = max_turns
        self.sandbox = sandbox
        self.approval_mode = approval_mode
        self._temp_config_dir = None

    def run(self, spec: AgentSpec) -> AgentResult:
        server = None
        bin_dir = None
        try:
            bin_dir = Path(tempfile.mkdtemp(prefix="gemini_tools_"))
            server = self._start_tool_server(spec)
            port = server.port if server else None
            if spec.tools and port is not None:
                generate_wrappers(spec.tools, bin_dir, port)
            self._write_gemini_config(spec)
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
            self._cleanup_config()

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
        env.update(spec.env)
        return env

    def _write_gemini_config(self, spec: AgentSpec) -> None:
        """Write temporary .gemini/settings.json for turn limits."""
        max_turns = self.max_turns
        if not max_turns:
            return

        config = {"model": {"maxSessionTurns": max_turns}}
        config_dir = spec.workspace_path / ".gemini"
        config_dir.mkdir(exist_ok=True)
        (config_dir / "settings.json").write_text(json.dumps(config, indent=2))
        self._temp_config_dir = config_dir

    def _build_prompt(self, spec: AgentSpec) -> str:
        """Augment goal with tool docs and deny rules (encoded in prompt)."""
        prompt = spec.goal
        docs = build_tool_docs(spec.tools)
        if docs:
            prompt += "\n\n" + docs

        # Gemini has no --disallowedTools flag — encode in prompt
        if spec.denied_tools:
            lines = ["\n\n## Restrictions\n"]
            lines.append("You MUST NOT use the following tools or actions:")
            for rule in spec.denied_tools:
                lines.append(f"- {rule}")
            prompt += "\n".join(lines)

        return prompt

    def _build_command(self, spec: AgentSpec) -> list:
        model = spec.model or self.model

        if spec.session_id:
            # Resuming — still need tool docs and deny rules in prompt
            prompt = self._build_prompt(spec)
            cmd = [
                "gemini", "-p", prompt,
                "--output-format", "stream-json",
                "-m", model,
                "--resume", spec.session_id,
            ]
        else:
            prompt = self._build_prompt(spec)
            cmd = [
                "gemini", "-p", prompt,
                "--output-format", "stream-json",
                "-m", model,
            ]

        if self.approval_mode != "default":
            cmd += ["--approval-mode", self.approval_mode]

        if self.sandbox:
            cmd.append("--sandbox")

        return cmd

    def _run_subprocess(self, cmd: list, env: dict, spec: AgentSpec) -> List[dict]:
        """Run subprocess with stream-json, writing events live to workspace."""
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
                actual_prompt = cmd[cmd.index("-p") + 1] if "-p" in cmd else spec.goal
                prompt_event = {
                    "type": "user",
                    "subtype": "initial_prompt",
                    "message": {"role": "user", "content": actual_prompt},
                }
                raw_file.write(json.dumps(prompt_event) + "\n")
                raw_file.flush()
                summary_file.write(json.dumps({"role": "user", "content": actual_prompt}) + "\n")
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
            raise RuntimeError(f"gemini exited with code {proc.returncode}")

        return events

    def _parse_result(self, events: List[dict]) -> AgentResult:
        """Parse Gemini stream-json events into AgentResult."""
        data = None
        for event in reversed(events):
            if event.get("type") == "result":
                data = event
                break

        if data is None:
            text_parts = []
            for event in events:
                if event.get("type") == "message":
                    content = event.get("content", "")
                    if isinstance(content, str):
                        text_parts.append(content)
                elif event.get("type") == "assistant":
                    for block in event.get("message", {}).get("content", []):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
            combined = " ".join(str(p) for p in text_parts if p)
            return AgentResult(success=True, output=combined.strip(), steps=_extract_steps(events))

        is_error = data.get("is_error", False)
        result_text = data.get("result", data.get("response", ""))
        cost = data.get("total_cost_usd", data.get("cost_usd", 0.0))
        turns = data.get("num_turns", data.get("turns", 0))

        return AgentResult(
            success=not is_error,
            output=result_text,
            session_id=data.get("session_id"),
            cost=cost or 0.0,
            turns=turns or 0,
            duration_ms=data.get("duration_ms", 0),
            error=result_text if is_error else None,
            steps=_extract_steps(events),
        )

    def _cleanup_config(self):
        """Remove temp .gemini/settings.json if we created it."""
        if self._temp_config_dir:
            settings = self._temp_config_dir / "settings.json"
            if settings.exists():
                settings.unlink()
            try:
                self._temp_config_dir.rmdir()
            except OSError:
                pass
            self._temp_config_dir = None


# --- Event parsing (module-level) ---

def _summarize_event(event: dict) -> List[dict]:
    """Convert a raw Gemini stream-json event into clean summary lines.

    Handles both Gemini-native event types (message, tool_use, tool_result)
    and Claude-style events (assistant, user) for compatibility.
    """
    event_type = event.get("type")

    if event_type == "message":
        lines = []
        content = event.get("content", event.get("message", {}).get("content", ""))
        if isinstance(content, str) and content.strip():
            lines.append({"role": "assistant", "type": "text", "content": content})
        elif isinstance(content, list):
            for block in content:
                block_type = block.get("type", "text")
                if block_type == "thinking":
                    lines.append({"role": "assistant", "type": "thinking",
                                  "content": block.get("thinking", block.get("content", ""))})
                elif block_type == "text":
                    text = block.get("text", block.get("content", ""))
                    if text.strip():
                        lines.append({"role": "assistant", "type": "text", "content": text})
                elif block_type == "tool_use":
                    lines.append({
                        "role": "assistant", "type": "tool_use",
                        "tool": block.get("name", "unknown"),
                        "input": block.get("input", block.get("args", {})),
                    })
        return lines

    if event_type == "assistant":
        lines = []
        for block in event.get("message", {}).get("content", []):
            block_type = block.get("type")
            if block_type == "thinking":
                lines.append({"role": "assistant", "type": "thinking", "content": block.get("thinking", "")})
            elif block_type == "text":
                text = block.get("text", "")
                if text.strip():
                    lines.append({"role": "assistant", "type": "text", "content": text})
            elif block_type == "tool_use":
                lines.append({
                    "role": "assistant", "type": "tool_use",
                    "tool": block.get("name", "unknown"),
                    "input": block.get("input", {}),
                })
        return lines

    if event_type == "tool_use":
        return [{
            "role": "assistant", "type": "tool_use",
            "tool": event.get("name", event.get("tool", "unknown")),
            "input": event.get("input", event.get("args", {})),
        }]

    if event_type == "tool_result":
        return [{
            "role": "tool_result",
            "tool_use_id": event.get("tool_use_id", event.get("id")),
            "content": event.get("content", event.get("output", "")),
        }]

    if event_type == "user":
        lines = []
        content = event.get("message", {}).get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    lines.append({
                        "role": "tool_result",
                        "tool_use_id": block.get("tool_use_id"),
                        "content": block.get("content", ""),
                    })
        elif isinstance(content, str) and content.strip():
            lines.append({"role": "tool_result", "content": content})
        return lines

    if event_type == "result":
        return [{
            "role": "result",
            "subtype": event.get("subtype", ""),
            "duration_ms": event.get("duration_ms"),
            "cost_usd": event.get("total_cost_usd"),
            "turns": event.get("num_turns"),
            "session_id": event.get("session_id"),
        }]

    return []


def _extract_steps(events: List[dict]) -> List[dict]:
    """Extract compact step summaries from Gemini stream-json events."""
    MAX_TEXT = 500
    steps = []
    pending_tools: Dict[str, int] = {}

    for event in events:
        event_type = event.get("type")

        if event_type == "message":
            content = event.get("content", event.get("message", {}).get("content", ""))
            if isinstance(content, str) and content.strip():
                steps.append({
                    "type": "text",
                    "text": content[:MAX_TEXT] + ("..." if len(content) > MAX_TEXT else ""),
                })
            elif isinstance(content, list):
                for block in content:
                    _extract_block(block, steps, pending_tools, MAX_TEXT)

        elif event_type == "assistant":
            for block in event.get("message", {}).get("content", []):
                _extract_block(block, steps, pending_tools, MAX_TEXT)

        elif event_type == "tool_use":
            tool_input = event.get("input", event.get("args", {}))
            compact_input = _truncate_input(tool_input, MAX_TEXT) if isinstance(tool_input, dict) else {}
            step = {
                "type": "tool_use",
                "tool": event.get("name", event.get("tool", "unknown")),
                "input": compact_input,
            }
            steps.append(step)
            tool_use_id = event.get("id", event.get("tool_use_id"))
            if tool_use_id:
                pending_tools[tool_use_id] = len(steps) - 1

        elif event_type in ("tool_result", "tool"):
            tool_use_id = event.get("tool_use_id", event.get("id"))
            content = event.get("content", event.get("output", ""))
            truncated = (content[:MAX_TEXT] + "...") if isinstance(content, str) and len(content) > MAX_TEXT else str(content)[:MAX_TEXT]

            if tool_use_id and tool_use_id in pending_tools:
                idx = pending_tools.pop(tool_use_id)
                steps[idx]["output"] = truncated
            else:
                steps.append({"type": "tool_result", "output": truncated})

    return steps


def _extract_block(block: dict, steps: list, pending_tools: dict, max_text: int) -> None:
    """Extract a single content block into steps list."""
    block_type = block.get("type", "text")
    if block_type == "text":
        text = block.get("text", block.get("content", ""))
        if text.strip():
            steps.append({
                "type": "text",
                "text": text[:max_text] + ("..." if len(text) > max_text else ""),
            })
    elif block_type == "tool_use":
        tool_input = block.get("input", block.get("args", {}))
        compact_input = _truncate_input(tool_input, max_text) if isinstance(tool_input, dict) else {}
        step = {
            "type": "tool_use",
            "tool": block.get("name", "unknown"),
            "input": compact_input,
        }
        steps.append(step)
        tool_use_id = block.get("id")
        if tool_use_id:
            pending_tools[tool_use_id] = len(steps) - 1


def _truncate_input(input_dict: dict, max_len: int) -> dict:
    """Truncate string values in tool input for compact representation."""
    return {
        k: (v[:max_len] + "..." if isinstance(v, str) and len(v) > max_len else v)
        for k, v in input_dict.items()
    }
