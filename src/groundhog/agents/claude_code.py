"""ClaudeCodeAgentBackend — runs Claude Code CLI as a subprocess with tool access.

Tools are exposed via a localhost HTTP server with bash wrapper scripts
prepended to PATH so the CLI can call them via Bash.

Ported from EvaluatableExperiments/src/agents/implementations/claude_code.py.
Simplified: flat constructor params, unified AgentSpec, centralized tool_server.
"""

import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from groundhog.base.agent import AgentBackend, AgentResult, AgentSpec
from groundhog.agents.tool_server import (
    ToolServer,
    build_tool_docs,
    cleanup_wrappers,
    generate_wrappers,
)


class ClaudeCodeAgentBackend(AgentBackend):
    """Agent backend that runs the claude CLI as a subprocess.

    Tools are exposed via HTTP tool server + bash wrapper scripts.
    Supports session resume via --session-id / --resume.
    """

    def __init__(self, model: str = "sonnet", effort: str = "high",
                 max_turns: Optional[int] = None,
                 max_budget_usd: Optional[float] = None):
        self.model = model
        self.effort = effort
        self.max_turns = max_turns
        self.max_budget_usd = max_budget_usd

    def run(self, spec: AgentSpec) -> AgentResult:
        server = None
        bin_dir = None
        try:
            bin_dir = Path(tempfile.mkdtemp(prefix="claude_tools_"))
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
        """Augment the goal with tool documentation."""
        docs = build_tool_docs(spec.tools)
        if docs:
            return spec.goal + "\n\n" + docs
        return spec.goal

    def _build_command(self, spec: AgentSpec) -> list:
        model = spec.model or self.model
        effort = spec.effort or self.effort

        if spec.session_id:
            # Resuming an existing session
            prompt = spec.goal  # no tool docs needed on resume
            cmd = [
                "claude", "-p", prompt,
                "--output-format", "stream-json",
                "--verbose",
                "--model", model,
                "--effort", effort,
                "--resume", spec.session_id,
            ]
        else:
            # New session — assign a session ID for potential future resume
            prompt = self._build_prompt(spec)
            session_id = str(uuid.uuid4())
            cmd = [
                "claude", "-p", prompt,
                "--output-format", "stream-json",
                "--verbose",
                "--model", model,
                "--effort", effort,
                "--session-id", session_id,
            ]

        max_turns = self.max_turns
        if max_turns:
            cmd += ["--max-turns", str(max_turns)]

        # Budget: spec overrides constructor default
        max_budget = spec.budget_usd if spec.budget_usd is not None else self.max_budget_usd
        if max_budget:
            cmd += ["--max-budget-usd", str(max_budget)]

        # Permission rules
        allow_rules = list(spec.allowed_tools)
        for tool in spec.tools:
            allow_rules.append(f"Bash({tool.name}:*)")
            allow_rules.append(f"Bash({tool.name})")
        if allow_rules:
            cmd += ["--allowedTools"] + allow_rules

        # Claude resolves rule conflicts as ``deny > allow``: a broad
        # ``Read(*)`` deny disables the Read tool entirely even when
        # specific ``Read(./**)`` allows are present (same for Write/Edit).
        # When the spec has narrow allows for a tool, drop the broad deny —
        # the allow patterns themselves act as the boundary in claude.
        deny_rules = _filter_redundant_broad_denies(
            list(spec.denied_tools), allow_rules
        )
        if deny_rules:
            cmd += ["--disallowedTools"] + deny_rules

        return cmd

    def _run_subprocess(self, cmd: list, env: dict, spec: AgentSpec) -> List[dict]:
        """Run subprocess with stream-json, writing events live to workspace.

        Writes:
          agent_steps.jsonl   — raw events (full metadata, for debugging)
          agent_summary.jsonl — clean summary (content only, no metadata noise)
        """
        jsonl_path = spec.workspace_path / "agent_steps.jsonl"
        summary_path = spec.workspace_path / "agent_summary.jsonl"
        deadline = time.monotonic() + spec.timeout if spec.timeout else None

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit — streams to terminal
            text=True,
            cwd=str(spec.workspace_path),
            env=env,
        )

        events = []
        try:
            with open(jsonl_path, "a", encoding="utf-8") as raw_file, \
                 open(summary_path, "a", encoding="utf-8") as summary_file:
                # Write initial prompt — the CLI doesn't emit it
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
            raise RuntimeError(f"claude exited with code {proc.returncode}")

        return events

    def _parse_result(self, events: List[dict]) -> AgentResult:
        """Parse stream-json events into AgentResult."""
        # Find the result event
        data = None
        for event in reversed(events):
            if event.get("type") == "result":
                data = event
                break

        if data is None:
            combined = " ".join(
                event.get("result", event.get("content", ""))
                for event in events if isinstance(event, dict)
            )
            return AgentResult(
                success=True,
                output=combined.strip(),
                steps=_extract_steps(events),
            )

        is_error = data.get("is_error", False)
        result_text = data.get("result", "")

        return AgentResult(
            success=not is_error,
            output=result_text,
            session_id=data.get("session_id"),
            cost=data.get("total_cost_usd", 0.0),
            turns=data.get("num_turns", 0),
            duration_ms=data.get("duration_ms", 0),
            error=result_text if is_error else None,
            steps=_extract_steps(events),
        )


def _filter_redundant_broad_denies(deny_rules: List[str], allow_rules: List[str]) -> List[str]:
    """Strip broad ``Tool(*)`` denies when narrow ``Tool(<pattern>)`` allows
    exist for the same tool. Claude treats deny as higher precedence than
    allow, so a broad deny would shadow the narrow allow and disable the
    tool entirely. The narrow allow is itself the boundary — paths not
    matching it are already denied by default.
    """
    def _tool_name(rule: str) -> Optional[str]:
        if "(" in rule and rule.endswith(")"):
            return rule.split("(", 1)[0]
        return None

    def _is_broad(rule: str) -> bool:
        return rule.endswith("(*)")

    narrow_allow_tools = {
        _tool_name(r) for r in allow_rules
        if _tool_name(r) and not _is_broad(r)
    }
    return [r for r in deny_rules
            if not (_is_broad(r) and _tool_name(r) in narrow_allow_tools)]


# --- Event parsing (module-level, reusable) ---

def _summarize_event(event: dict) -> List[dict]:
    """Convert a raw stream-json event into clean summary lines.

    Strips token counts, UUIDs, signatures, cache info — keeps only
    the content the model produced or received.
    """
    event_type = event.get("type")

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
                    "role": "assistant",
                    "type": "tool_use",
                    "tool": block.get("name", "unknown"),
                    "input": block.get("input", {}),
                })
        return lines

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
    """Extract compact step summaries from stream-json events.

    Walks events correlating tool_use blocks with their results.
    """
    MAX_TEXT = 500
    steps = []
    pending_tools: Dict[str, int] = {}

    for event in events:
        event_type = event.get("type")

        if event_type == "assistant":
            for block in event.get("message", {}).get("content", []):
                block_type = block.get("type")

                if block_type == "text":
                    text = block.get("text", "")
                    if text.strip():
                        steps.append({
                            "type": "text",
                            "text": text[:MAX_TEXT] + ("..." if len(text) > MAX_TEXT else ""),
                        })

                elif block_type == "tool_use":
                    tool_input = block.get("input", {})
                    compact_input = {
                        k: (v[:MAX_TEXT] + "..." if isinstance(v, str) and len(v) > MAX_TEXT else v)
                        for k, v in tool_input.items()
                    }
                    step = {
                        "type": "tool_use",
                        "tool": block.get("name", "unknown"),
                        "input": compact_input,
                    }
                    steps.append(step)
                    tool_use_id = block.get("id")
                    if tool_use_id:
                        pending_tools[tool_use_id] = len(steps) - 1

        elif event_type == "tool":
            tool_use_id = event.get("tool_use_id")
            content = event.get("content", "")
            truncated = (content[:MAX_TEXT] + "...") if isinstance(content, str) and len(content) > MAX_TEXT else str(content)[:MAX_TEXT]

            if tool_use_id and tool_use_id in pending_tools:
                idx = pending_tools.pop(tool_use_id)
                steps[idx]["output"] = truncated
            else:
                steps.append({"type": "tool_result", "output": truncated})

    return steps
