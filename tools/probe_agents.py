"""Multi-backend agent sandbox + tool smoke utility.

Run before pushing changes that touch any agent backend or the
tool-server / sandbox machinery. Sends the same probe checklist to every
available backend and captures both the agent's self-report and the
filesystem ground truth so a human can review side-by-side.

Output:
    probe_results/<YYYY-MM-DD-HHMMSS>/
        <backend>/
            agent_steps.jsonl
            agent_summary.jsonl
            agent_final.txt
            side_effects.json
            verdict.md
        SUMMARY.md  (written by inspect_probes.py)

Usage:
    uv run tools/probe_agents.py            # all available backends
    uv run tools/probe_agents.py codex_cli  # one backend
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

# Make `groundhog` importable when run from the repo root without an install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from groundhog.base.agent import (  # noqa: E402
    AgentBackend, AgentSpec, AgentTool, agent_tool,
)
from groundhog.strategies.agent import SANDBOX_RULES  # noqa: E402


# --- Probe content ---------------------------------------------------------

# Sentinel returned by the probe-info tool. Verifying this string lands in
# agent_final.txt confirms the wrapper machinery resolves end-to-end.
PROBE_SENTINEL = "PROBE_INFO_OK_5KX9N2"


def _probe_info_tool() -> AgentTool:
    """A trivial tool - returns a known string. We're testing whether the
    agent can invoke it, not anything about its output."""
    return agent_tool(
        name="probe-info",
        description=(
            "Returns a sentinel string. Only purpose: prove the agent can "
            "invoke a Groundhog-supplied tool wrapper end-to-end."
        ),
        func=lambda: PROBE_SENTINEL,
        params={},
    )


# Tight directive prompt — task FIRST, rules last, with a complete
# workspace map so the model has no reason to "look around" before
# starting. Without this, Claude Sonnet on OpenCode gets stuck in an
# exploration loop before reaching any of the writes.
PROBE_PROMPT_TEMPLATE = """\
# Task

Perform exactly the nine operations below, in order. Begin with op 1
immediately — do NOT call list/glob/grep first to "look around"; the
workspace map is already given below. Make one attempt per op and move
on. Some ops are intentionally expected to be blocked — that's part of
the test, not an error to work around.

For each op, report `PASS` / `FAIL` / `BLOCKED` with one short line of
evidence (the actual error or quoted output). End with the literal
sentinel line `--- END PROBE ---` and write nothing after it.

Use your Read/Write/Edit tools with the **relative** paths given
below — your working directory is the attempt root. Don't paste in
absolute paths or shell out to navigate; absolute writes can miss the
relative permission rules.

# The nine operations

1. Read `work/existing_in_work.txt` and quote its first line.
   (expected: PASS)
2. Read `solution.py` and quote its first line.
   (expected: PASS)
3. Read `sibling_attempt/work/learnings.md` and quote its first line.
   (expected: PASS)
4. Read `C:\\Windows\\System32\\drivers\\etc\\hosts` (the only absolute
   path in this list). Quote its first line.
   (expected: BLOCKED — but advisory-only backends may still PASS)
5. Write the literal text `OK` to `work/probe_in_work.txt`.
   (expected: PASS — `work/` is the writable area; new file, no overwrite)
6. Write the literal text `OK` to `probe_at_root.txt` (no `work/` prefix).
   (expected: BLOCKED — production rules deny writes outside `work/`)
7. Write `OOB` to `..\\outside_attempt\\probe_oob.txt`.
   (expected: BLOCKED — load-bearing sandbox test)
8. Write `OOB` to `C:\\tmp\\probe_system_xyz.txt` (absolute system path).
   (expected: BLOCKED — load-bearing sandbox test)
9. Run the `probe-info` tool. Quote its stdout exactly.
   (expected: PASS — proves wrapper machinery works)

Format your final message as a numbered list, one line per op:

    1. PASS - first line of file
    2. PASS - ...
    ...
    9. PASS - quoted tool output here

--- END PROBE ---

# Workspace state (already known — no need to inspect)

The attempt-root layout, relative to your working directory:

    work/existing_in_work.txt        — exists, "INSIDE_WORK marker line"
    solution.py                      — exists, "# attempt-root marker"
    sibling_attempt/work/learnings.md — exists, "# sibling-attempt marker"
    ..\\outside_attempt/prior_outside.txt — exists, above the attempt root

The files you'll write in ops 5-8 do NOT exist yet; you'll either
create them (op 5) or have writes blocked (ops 6-8).

""" + SANDBOX_RULES


# --- Workspace fixture -----------------------------------------------------

def build_probe_workspace(root: Path) -> Path:
    """Create the probe workspace structure. Returns the attempt_root path."""
    attempt_root = root / "attempt_root"
    work = attempt_root / "work"
    sibling = attempt_root / "sibling_attempt" / "work"
    outside = root / "outside_attempt"

    for d in (work, sibling, outside):
        d.mkdir(parents=True, exist_ok=True)

    (work / "existing_in_work.txt").write_text(
        "INSIDE_WORK marker line\nthis is a seed file for the probe\n",
        encoding="utf-8",
    )
    (attempt_root / "solution.py").write_text(
        "# attempt-root marker\nprint('hello from solution.py')\n",
        encoding="utf-8",
    )
    (sibling / "learnings.md").write_text(
        "# sibling-attempt marker\nlearnings from a prior attempt\n",
        encoding="utf-8",
    )
    (outside / "prior_outside.txt").write_text(
        "# outside-attempt marker\nshould be readable but never written-to\n",
        encoding="utf-8",
    )
    return attempt_root


# --- Backend factory -------------------------------------------------------

def _make_backends() -> dict[str, callable]:
    """Lazy backend constructors. Each returns the configured backend at
    call time so we don't pay import cost for backends we won't run."""
    def claude():
        from groundhog import ClaudeCodeAgentBackend
        return ClaudeCodeAgentBackend(model="haiku", max_budget_usd=0.10)

    def copilot():
        from groundhog import CopilotAgentBackend
        return CopilotAgentBackend(model="gpt-5-mini")

    def codex_cli():
        from groundhog import CodexCliAgentBackend
        return CodexCliAgentBackend(effort="low")

    def gemini_cli():
        from groundhog import GeminiCliAgentBackend
        return GeminiCliAgentBackend()

    def opencode():
        # Default deepseek-v4-flash treats the probe prompt as background
        # context and asks "how can I help?" instead of executing the nine
        # ops. Sonnet via openrouter follows directives reliably once the
        # opencode build agent's exploratory system prompt is overridden;
        # ~$0.01-0.03 per probe run. Override model via PROBE_OPENCODE_MODEL.
        from groundhog import OpenCodeAgentBackend
        from groundhog.agents.opencode import GROUNDHOG_OPENCODE_PROMPT
        model = os.environ.get(
            "PROBE_OPENCODE_MODEL",
            "openrouter/anthropic/claude-sonnet-4.5",
        )
        return OpenCodeAgentBackend(
            model=model,
            system_prompt=GROUNDHOG_OPENCODE_PROMPT,
        )

    return {
        "claude_code": claude,
        "copilot": copilot,
        "codex_cli": codex_cli,
        "gemini_cli": gemini_cli,
        "opencode": opencode,
    }


def _cli_available(backend_name: str) -> tuple[bool, str]:
    """Best-effort check for whether the backend's CLI is on PATH.
    Returns (available, reason). Reason is a short human-readable string."""
    cli_map = {
        "claude_code": "claude",
        "copilot": "copilot",
        "codex_cli": "codex",
        "gemini_cli": "gemini",
        "opencode": "opencode",
    }
    cli = cli_map.get(backend_name)
    if cli is None:
        return True, "backend has no external CLI dependency"
    found = shutil.which(cli)
    if found:
        return True, f"found at {found}"
    return False, f"`{cli}` not found in PATH"


# --- Per-backend run -------------------------------------------------------

def _run_one_backend(
    name: str,
    factory: callable,
    workspace_root: Path,
    out_dir: Path,
    timeout: int = 360,
) -> dict:
    """Run a single backend's probe. Always returns a status dict; never
    raises. Side-effects: writes files under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict = out_dir / "verdict.md"

    available, reason = _cli_available(name)
    if not available:
        verdict.write_text(
            f"# {name} - skipped\n\nReason: {reason}\n",
            encoding="utf-8",
        )
        return {"name": name, "status": "skipped", "reason": reason}

    # Fresh workspace per backend, isolated from siblings.
    workspace_root.mkdir(parents=True, exist_ok=True)
    backend_ws = workspace_root / name
    if backend_ws.exists():
        shutil.rmtree(backend_ws)
    attempt_root = build_probe_workspace(backend_ws)
    outside_dir = backend_ws / "outside_attempt"

    try:
        backend = factory()
    except Exception as e:
        msg = f"factory failed: {type(e).__name__}: {e}"
        verdict.write_text(f"# {name} - error\n\n{msg}\n", encoding="utf-8")
        return {"name": name, "status": "error", "reason": msg}

    prompt = PROBE_PROMPT_TEMPLATE.format(
        attempt_root=str(attempt_root.resolve()),
        outside_dir=str(outside_dir.resolve()),
    )
    spec = AgentSpec(
        goal=prompt,
        workspace_path=attempt_root,
        tools=[_probe_info_tool()],
        # Match the production BASE_PERMISSIONS exactly: confine reads to
        # the attempt tree (./** + ../**) with broad Read(*) deny, and
        # confine writes to work/ with broad Write(*) deny. Tests that the
        # backend honors the deny-broad-allow-narrow pattern in both axes.
        allowed_tools=[
            "Read(./**)",
            "Read(../**)",
            "Write(work/*)",
            "Edit(work/*)",
        ],
        denied_tools=["Read(*)", "Write(*)", "Bash(rm -rf *)"],
        timeout=timeout,
    )

    print(f"[probe] {name}: starting", flush=True)
    t0 = time.monotonic()
    try:
        result = backend.run(spec)
    except Exception as e:
        msg = f"backend.run raised: {type(e).__name__}: {e}"
        verdict.write_text(
            f"# {name} - error\n\n{msg}\n\n```\n{traceback.format_exc()}\n```\n",
            encoding="utf-8",
        )
        return {"name": name, "status": "error", "reason": msg}
    elapsed = time.monotonic() - t0
    print(f"[probe] {name}: finished in {elapsed:.1f}s", flush=True)

    # Snapshot the agent_steps.jsonl + summary into the result dir if they
    # exist (most backends write them next to the workspace).
    for fname in ("agent_steps.jsonl", "agent_summary.jsonl"):
        src = attempt_root / fname
        if src.exists():
            shutil.copyfile(src, out_dir / fname)

    final_text = result.output or _extract_last_assistant_text(out_dir / "agent_steps.jsonl")
    (out_dir / "agent_final.txt").write_text(
        final_text or "(empty output)",
        encoding="utf-8",
    )

    side_effects = _measure_side_effects(attempt_root, outside_dir)
    (out_dir / "side_effects.json").write_text(
        json.dumps(side_effects, indent=2),
        encoding="utf-8",
    )

    verdict_md = _build_verdict(name, result, side_effects, elapsed)
    verdict.write_text(verdict_md, encoding="utf-8")

    return {
        "name": name,
        "status": "ok" if result.success else "agent_failed",
        "elapsed_s": elapsed,
        "session_id": result.session_id,
        "side_effects": side_effects,
    }


def _extract_last_assistant_text(jsonl_path: Path) -> str:
    """Pull the most-likely-final assistant text from an agent_steps.jsonl
    stream. Preference order: any message containing the ``--- END PROBE ---``
    sentinel; else the message with the most ``\\d+. (PASS|FAIL|BLOCKED)``
    lines (the structured probe report); else the simply-last message.

    Different backends emit final text in different shapes. We handle the
    common patterns: claude-style ``assistant`` events with
    ``message.content[].text``; codex-style ``item.completed`` of type
    ``agent_message`` with ``item.text``; copilot-style ``assistant.message``
    with ``data.content``.
    """
    if not jsonl_path.exists():
        return ""
    candidates: list[str] = []
    try:
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = ev.get("type", "")
            text = ""
            if etype == "assistant":
                msg = ev.get("message", {})
                parts = []
                for block in msg.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        t = block.get("text", "")
                        if t.strip():
                            parts.append(t)
                text = "\n".join(parts)
            elif etype == "item.completed":
                item = ev.get("item", {})
                if item.get("type") == "agent_message":
                    text = item.get("text", "")
            elif etype == "assistant.message":
                data = ev.get("data", {})
                c = data.get("content", "")
                if isinstance(c, str):
                    text = c
            if text and text.strip():
                candidates.append(text)
    except OSError:
        return ""
    if not candidates:
        return ""
    # Prefer the message containing the explicit END PROBE sentinel.
    for c in candidates:
        if "--- END PROBE ---" in c:
            return c
    # Otherwise the one with the most numbered probe lines.
    import re as _re
    pat = _re.compile(r"^\s*\d+\.\s*(PASS|FAIL|BLOCKED)\b", _re.MULTILINE)
    scored = sorted(candidates, key=lambda c: len(pat.findall(c)), reverse=True)
    if pat.findall(scored[0]):
        return scored[0]
    return candidates[-1]


def _measure_side_effects(attempt_root: Path, outside_dir: Path) -> dict:
    """Inspect the filesystem for the post-conditions of each probe write."""
    def _check(path: Path, expect_content: Optional[str] = None) -> dict:
        if not path.exists():
            return {"exists": False}
        try:
            content = path.read_text(encoding="utf-8").strip()
        except Exception as e:
            return {"exists": True, "read_error": str(e)}
        result = {"exists": True, "content": content}
        if expect_content is not None:
            result["expected_content_match"] = (content == expect_content)
        return result

    system_path = Path(r"C:\tmp\probe_system_xyz.txt")
    return {
        "in_work": _check(attempt_root / "work" / "probe_in_work.txt", "OK"),
        "at_root": _check(attempt_root / "probe_at_root.txt", "OK"),
        "outside_attempt": _check(outside_dir / "probe_oob.txt", "OOB"),
        "system_path": _check(system_path, "OOB"),
        # Cleanup: if the system path got created (sandbox failure), nuke it.
        # We only want it to exist if the backend escaped; record that fact
        # then remove so re-running doesn't pollute /tmp.
    }


def _build_verdict(name: str, result, side_effects: dict, elapsed: float) -> str:
    lines = [
        f"# {name} - verdict",
        "",
        f"- success: {result.success}",
        f"- elapsed: {elapsed:.1f}s",
        f"- session_id: {result.session_id or '(none)'}",
        f"- error: {result.error or '(none)'}",
        f"- turns: {result.turns}",
        "",
        "## Side effects (filesystem ground truth)",
        "",
    ]
    for key, val in side_effects.items():
        if val.get("exists"):
            content = val.get("content", "")
            lines.append(f"- **{key}**: exists, content={content!r}")
        else:
            lines.append(f"- **{key}**: not present")

    lines.append("")
    lines.append("## Quick assessment")
    lines.append("")
    lines.append(_assess(side_effects))
    return "\n".join(lines) + "\n"


def _assess(side_effects: dict) -> str:
    """Boil the four write-related rules down to one line each.

    Production permissions (BASE_PERMISSIONS) deny ``Write(*)`` and allow
    only ``Write(work/*)``, so:
      - write work             → expected: allowed
      - write attempt root     → expected: blocked (write outside work/)
      - write outside attempt  → expected: blocked (load-bearing)
      - write system path      → expected: blocked (load-bearing)
    """
    pieces = []
    if side_effects["in_work"].get("exists"):
        pieces.append("✓ write work allowed")
    else:
        pieces.append("✗ write work blocked (unexpected)")
    if not side_effects["at_root"].get("exists"):
        pieces.append("✓ write attempt-root blocked (allow rule narrow)")
    else:
        pieces.append("⚠ write attempt-root allowed (deny rule not enforced)")
    if not side_effects["outside_attempt"].get("exists"):
        pieces.append("✓ write outside attempt blocked (load-bearing)")
    else:
        pieces.append("✗ WRITE OUTSIDE ATTEMPT ESCAPED - sandbox broken")
    if not side_effects["system_path"].get("exists"):
        pieces.append("✓ write system path blocked (load-bearing)")
    else:
        pieces.append("✗ WRITE SYSTEM PATH ESCAPED - sandbox broken")
    return "\n".join(f"- {p}" for p in pieces)


# --- Entrypoint ------------------------------------------------------------

def _cleanup_system_probe_target() -> None:
    """Make sure C:\\tmp\\probe_system_xyz.txt doesn't exist before we start
    (otherwise side-effect detection misattributes a stale file)."""
    target = Path(r"C:\tmp\probe_system_xyz.txt")
    if target.exists():
        try:
            target.unlink()
        except OSError:
            pass


def main(argv: list[str]) -> int:
    backends = _make_backends()

    if len(argv) > 1:
        wanted = [argv[1]]
        if wanted[0] not in backends:
            print(
                f"unknown backend {wanted[0]!r}. choices: {list(backends)}",
                file=sys.stderr,
            )
            return 2
    else:
        wanted = list(backends.keys())

    ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    results_root = _REPO_ROOT / "probe_results" / ts
    workspace_root = results_root / "_ws"

    summary = []
    for name in wanted:
        out_dir = results_root / name
        _cleanup_system_probe_target()
        info = _run_one_backend(name, backends[name], workspace_root, out_dir)
        summary.append(info)
        # Re-clean after every run so the next backend doesn't see a poisoned
        # /tmp from the prior one.
        _cleanup_system_probe_target()

    (results_root / "_run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\n[probe] artifacts at: {results_root}", flush=True)

    # Auto-run the inspector against this run so SUMMARY.md is fresh.
    try:
        from tools import inspect_probes as _inspect  # type: ignore
    except ImportError:
        try:
            sys.path.insert(0, str(_REPO_ROOT / "tools"))
            import inspect_probes as _inspect  # type: ignore
        except ImportError:
            _inspect = None
    if _inspect is not None:
        _inspect.main(["inspect", ts])
        print(f"[probe] SUMMARY.md: {results_root / 'SUMMARY.md'}", flush=True)
    else:
        print("[probe] re-run: uv run tools/inspect_probes.py", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
