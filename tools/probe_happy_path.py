"""Minimal happy-path probe: verify each backend can write to work/ and
invoke a registered tool.

Scope is intentionally tiny — one write, one tool call, ~3-line prompt.
This isolates the "positive case" from the bigger sandbox probe so we
can tell whether a failure is a permission/constraint problem (rules
block work/ writes) or a model-compliance problem (model didn't try
because the prompt was too exploratory).

Run before pushing changes that touch:
- ``BASE_PERMISSIONS`` / ``AgentStrategy.permissions``
- per-backend permission translation (claude --allowedTools,
  opencode permission config, copilot --deny-tool, etc.)
- the tool-server / wrapper machinery

Usage:
    uv run tools/probe_happy_path.py            # all available backends
    uv run tools/probe_happy_path.py opencode   # just one
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from groundhog.base.agent import AgentSpec, agent_tool  # noqa: E402
from groundhog.strategies.agent import SANDBOX_RULES  # noqa: E402

# Reuse the backend factory from the full probe so the two stay aligned.
from probe_agents import _make_backends, _cli_available  # noqa: E402


HAPPY_SENTINEL = "HAPPY_TOOL_OK_8WJM2"


def _happy_tool():
    return agent_tool(
        name="happy-info",
        description="Returns a sentinel string; lets us prove tool wrappers resolve end-to-end.",
        func=lambda: HAPPY_SENTINEL,
        params={},
    )


# Lean, imperative prompt. The full SANDBOX_RULES block is intentionally
# omitted here — opencode enforces the rules via its generated config,
# claude_code via --allowedTools/--disallowedTools, etc., so the prompt
# rules are redundant for this minimal happy-path test. Removing them
# also stops Claude Sonnet on OpenCode from priming itself into "let me
# understand what I'm allowed to do" mode before the actual task.
HAPPY_PROMPT = """\
You have two tool calls to make. You already have all the context you
need; do not read, list, glob, or grep anything first.

1. Write the literal text `OK` to `work/happy.txt`. New file, no
   existing content to preserve.
2. Call the `happy-info` tool and quote its stdout in your reply.

End your reply with the line `--- HAPPY DONE ---` and nothing after."""


def build_workspace(root: Path) -> Path:
    attempt = root / "attempt"
    (attempt / "work").mkdir(parents=True, exist_ok=True)
    (attempt / "work" / "seed.txt").write_text("seed\n", encoding="utf-8")
    return attempt


def run_backend(name: str, factory, ws_root: Path, out_dir: Path, timeout: int = 240) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict_path = out_dir / "verdict.md"

    available, reason = _cli_available(name)
    if not available:
        verdict_path.write_text(f"# {name} — skipped\n\n{reason}\n", encoding="utf-8")
        return {"name": name, "status": "skipped", "reason": reason}

    bws = ws_root / name
    if bws.exists():
        shutil.rmtree(bws)
    attempt = build_workspace(bws)

    try:
        backend = factory()
    except Exception as e:
        msg = f"factory failed: {type(e).__name__}: {e}"
        verdict_path.write_text(f"# {name} — error\n\n{msg}\n", encoding="utf-8")
        return {"name": name, "status": "error", "reason": msg}

    spec = AgentSpec(
        goal=HAPPY_PROMPT,
        workspace_path=attempt,
        tools=[_happy_tool()],
        allowed_tools=[
            "Read(./**)",
            "Read(../**)",
            "Write(work/*)",
            "Edit(work/*)",
        ],
        denied_tools=["Read(*)", "Write(*)", "Bash(rm -rf *)"],
        timeout=timeout,
    )

    print(f"[happy] {name}: starting", flush=True)
    t0 = time.monotonic()
    try:
        result = backend.run(spec)
    except Exception as e:
        msg = f"backend.run raised: {type(e).__name__}: {e}"
        verdict_path.write_text(f"# {name} — error\n\n{msg}\n", encoding="utf-8")
        return {"name": name, "status": "error", "reason": msg}
    elapsed = time.monotonic() - t0
    print(f"[happy] {name}: finished in {elapsed:.1f}s", flush=True)

    # Snapshot the agent step logs if the backend wrote them.
    for fname in ("agent_steps.jsonl", "agent_summary.jsonl"):
        src = attempt / fname
        if src.exists():
            shutil.copyfile(src, out_dir / fname)

    final_text = result.output or ""
    (out_dir / "agent_final.txt").write_text(final_text or "(empty output)", encoding="utf-8")

    happy_file = attempt / "work" / "happy.txt"
    wrote = happy_file.exists()
    contents = happy_file.read_text(encoding="utf-8") if wrote else ""
    # Tolerate trailing whitespace — some backends (codex via Set-Content)
    # add a default newline. The intent is "OK is in the file", not byte-exact.
    contents_ok = contents.strip() == "OK"
    # Tool was "called" if either the sentinel landed in the final text or
    # we observe a tool_use/command_execution targeting happy-info in the
    # backend's event log. Some backends don't surface tool stdout in the
    # final assistant text but the call definitely happened.
    tool_called = (
        HAPPY_SENTINEL in final_text
        or _scan_steps_for_sentinel(out_dir)
        or _scan_steps_for_tool_invocation(out_dir, "happy-info")
    )

    pass_overall = wrote and contents_ok and tool_called

    verdict_path.write_text(
        f"# {name} — happy path\n\n"
        f"- elapsed: {elapsed:.1f}s\n"
        f"- success: {result.success}\n"
        f"- error: {result.error or '(none)'}\n\n"
        f"## Required outcomes\n\n"
        f"- write `work/happy.txt` exists: {'✓' if wrote else '✗'}\n"
        f"- contents.strip() == 'OK': {'✓' if contents_ok else f'✗ (got {contents!r})'}\n"
        f"- happy-info invoked or sentinel observed: {'✓' if tool_called else '✗'}\n\n"
        f"**verdict: {'PASS' if pass_overall else 'FAIL'}**\n",
        encoding="utf-8",
    )

    return {
        "name": name,
        "status": "ok" if pass_overall else "fail",
        "elapsed_s": elapsed,
        "wrote_work_file": wrote,
        "tool_called": tool_called,
        "contents_ok": contents_ok,
    }


def _scan_steps_for_sentinel(out_dir: Path) -> bool:
    """Some backends don't surface tool output in the final assistant text;
    check the steps log for the sentinel as a fallback."""
    for fname in ("agent_steps.jsonl", "agent_summary.jsonl"):
        path = out_dir / fname
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if HAPPY_SENTINEL in line:
                    return True
        except OSError:
            pass
    return False


def _scan_steps_for_tool_invocation(out_dir: Path, tool_name: str) -> bool:
    """Detect that the agent tried to invoke our tool, regardless of whether
    its stdout reached the final text.

    Walks the JSONL events and checks for tool_use / command_execution
    events that target ``tool_name``. Doing it via the structured event
    rather than a string scan avoids tripping on permission-rule lists
    (which echo allowed wrapper paths back in bash-deny error messages).
    """
    for fname in ("agent_steps.jsonl", "agent_summary.jsonl"):
        path = out_dir / fname
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if _event_invokes_tool(ev, tool_name):
                    return True
        except OSError:
            pass
    return False


def _event_invokes_tool(ev: dict, tool_name: str) -> bool:
    """Inspect a single JSONL event for an agent-initiated invocation of
    ``tool_name``. Backends use different shapes:

    - claude_code: ``{"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "..."}]}}``
    - opencode:    ``{"type": "tool_use", "part": {"tool": "bash", "state": {"input": {"command": "..."}}}}``
    - copilot:     ``{"type": "assistant.message", "data": {"toolRequests": [{"name": "..."}]}}``
    - codex_cli:   ``{"type": "item.started", "item": {"type": "command_execution", "command": "..."}}``
    - gemini_cli:  ``{"type": "tool_use", "tool_name": "run_shell_command", "parameters": {"command": "..."}}``
    """
    # Direct top-level name fields (claude-style summary events, gemini).
    if ev.get("name") == tool_name or ev.get("tool") == tool_name:
        return True
    if ev.get("tool_name") == tool_name:
        return True
    # gemini wraps its tool calls via run_shell_command with the wrapper
    # in `parameters.command` — match the wrapper string explicitly.
    params = ev.get("parameters", {}) or {}
    if _command_targets_tool(params.get("command", ""), tool_name):
        return True

    # claude_code stream-json: tool_use blocks inside assistant messages.
    msg = ev.get("message", {})
    for block in msg.get("content", []) if isinstance(msg.get("content"), list) else []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            if block.get("name") == tool_name:
                return True
            # Bash invocations may run the wrapper; check the command string.
            inp = block.get("input", {}) or {}
            if _command_targets_tool(inp.get("command", ""), tool_name):
                return True

    # opencode part-shape: {"part": {"type": "tool", "tool": "bash", "state": {...}}}
    part = ev.get("part", {}) or {}
    if part.get("type") == "tool":
        if part.get("tool") == tool_name:
            return True
        state = part.get("state", {}) or {}
        cmd = (state.get("input", {}) or {}).get("command", "")
        if _command_targets_tool(cmd, tool_name):
            return True

    # copilot: data.toolRequests[].name
    data = ev.get("data", {}) or {}
    for req in data.get("toolRequests", []) or []:
        if isinstance(req, dict) and req.get("name") == tool_name:
            return True
    if data.get("toolName") == tool_name:
        return True

    # codex_cli: item.command_execution with the wrapper in the command string.
    item = ev.get("item", {}) or {}
    if item.get("type") == "command_execution":
        if _command_targets_tool(item.get("command", ""), tool_name):
            return True

    return False


def _command_targets_tool(command: str, tool_name: str) -> bool:
    """Heuristic: did the shell command try to invoke the wrapper?

    Looks for the tool name as a *whole word* preceded by a path
    separator, a space, or the call-operator `& `. Stops the
    permission-rule echo (e.g. ``"& happy-info.ps1*"`` inside a deny
    error) from triggering a false positive.
    """
    if not command:
        return False
    if tool_name not in command:
        return False
    # The tool was invoked iff the command IS the tool name (with optional
    # trailing args) or contains it as a path-resolved invocation. Compare
    # via the command's word-token boundaries rather than substring scan
    # so the ``"& happy-info.ps1*"`` permission-rule echo in error messages
    # doesn't trip a false positive.
    stripped = command.strip()
    if stripped == tool_name or stripped.startswith(f"{tool_name} "):
        return True  # bare invocation: ``happy-info`` or ``happy-info <args>``
    needles = (
        f"-Command {tool_name}",
        f"-Command \"{tool_name}",
        f"& {tool_name}.",       # call-op + extension (.ps1/.cmd)
        f"\\{tool_name}.",        # backslash + extension
        f"/{tool_name}.",         # forward slash + extension
        f" {tool_name} ",         # whole word in middle of command
        f" {tool_name}\n",        # whole word at line end
    )
    return any(n in command for n in needles)


def main(argv: list[str]) -> int:
    backends = _make_backends()
    selected = argv if argv else list(backends.keys())

    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    out_root = _REPO_ROOT / "probe_results" / f"happy-{timestamp}"
    ws_root = out_root / "_ws"

    results = []
    for name in selected:
        if name not in backends:
            print(f"[happy] unknown backend: {name}", flush=True)
            continue
        result = run_backend(
            name,
            backends[name],
            ws_root,
            out_root / name,
        )
        results.append(result)

    summary = "# Happy-path summary\n\n"
    summary += "| backend | wrote work/happy.txt | tool sentinel | verdict |\n"
    summary += "|---|---|---|---|\n"
    for r in results:
        if r["status"] == "skipped":
            summary += f"| {r['name']} | — | — | _skipped: {r['reason']}_ |\n"
            continue
        if r["status"] == "error":
            summary += f"| {r['name']} | — | — | _error: {r['reason']}_ |\n"
            continue
        wrote = "✓" if r["wrote_work_file"] and r["contents_ok"] else "✗"
        tool = "✓" if r["tool_called"] else "✗"
        verdict = "PASS" if r["status"] == "ok" else "FAIL"
        summary += f"| {r['name']} | {wrote} | {tool} | **{verdict}** |\n"

    (out_root / "SUMMARY.md").write_text(summary, encoding="utf-8")
    (out_root / "_run_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"\n[happy] artifacts at: {out_root}", flush=True)
    print(f"[happy] SUMMARY.md: {out_root / 'SUMMARY.md'}", flush=True)

    # Exit 1 if any non-skipped backend failed.
    if any(r["status"] == "fail" for r in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
