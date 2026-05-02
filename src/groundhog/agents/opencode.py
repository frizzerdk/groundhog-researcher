"""OpenCodeAgentBackend - runs OpenCode CLI as an autonomous agent.

OpenCode is useful for Groundhog because it can route through many providers,
including OpenRouter, while still acting as a coding agent with file and shell
tools. This adapter uses ``opencode run --format json`` and exposes Groundhog's
evaluation tools through the existing HTTP tool server + wrapper scripts.

Security model
--------------
OpenCode does not have an OS sandbox flag like Codex. Instead, we inject a
temporary ``opencode.json`` into the attempt workspace and point
``OPENCODE_CONFIG`` at it:

* read/search/list/edit are allowed according to Groundhog's attempt rules
* external directory access is denied
* webfetch/websearch/task are denied
* bash is denied by default, with allow rules for the ``%TEMP%`` wrapper bin
  (path is allocated per-run and threaded through the permission config)

That lets the agent edit the attempt workspace and call Groundhog eval tools
without granting broad shell access. Provider credentials can come from
OpenCode auth storage or from environment variables such as
``OPENROUTER_API_KEY``. The CLI is run with ``--pure`` and
``--dangerously-skip-permissions`` by default; the latter auto-approves only
actions not explicitly denied by the generated permission config.

Cost
----
OpenCode can show token/cost stats via ``opencode stats``, but raw JSON events
are provider/model dependent. This adapter parses cost/token fields when they
appear and otherwise leaves cost at ``0.0``.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from groundhog.base.agent import AgentBackend, AgentResult, AgentSpec
from groundhog.agents.tool_server import (
    ToolServer,
    build_tool_docs,
    cleanup_wrappers,
    generate_wrappers,
)


DEFAULT_MODEL = "openrouter/deepseek/deepseek-v4-flash"


# OpenCode's stock ``build`` agent ships with a system prompt that biases
# the model toward "understand the workspace before acting" — useful for
# ad-hoc human-in-the-loop coding, but it makes both Claude Sonnet and
# Deepseek wander into describe-mode when given a directive task. We can
# override that persona by setting ``agent.<name>.prompt`` in the
# generated opencode.json. This constant is the recommended override for
# Groundhog: keep all tools enabled so the agent CAN read/list/grep when
# the task says to, but stop priming it to do so before reading the task.
#
# Kept terse on purpose — opencode also ships its own boilerplate around
# this prompt, and verbose extra instructions get drowned. A short,
# imperative override lands with more weight than a long one.
GROUNDHOG_OPENCODE_PROMPT = (
    "You are running inside an automated Groundhog session. The user "
    "message is the complete task — execute it now using your tools. "
    "Do not call list/glob/grep/read tools to explore unless the task "
    "explicitly tells you to. Do not ask the user clarifying questions. "
    "Do not summarize what you would do — do it. When a permission rule "
    "blocks a tool call, that's intended; note it and move on without "
    "retrying or working around."
)


def _resolve_opencode_bin() -> str:
    """Locate the opencode executable as a concrete path.

    On Windows, npm installs an ``opencode.cmd`` shim that forwards argv
    to the real ``opencode.exe`` via ``%*``. cmd.exe's argument parser
    treats newlines as command terminators, so a multi-line prompt
    passed as a positional argv gets truncated at the first ``\\n`` —
    sonnet then sees only the first line of the prompt and politely
    asks "what do you want?". Resolving directly to the bundled .exe
    bypasses cmd.exe entirely and preserves multi-line argv.
    """
    if os.name == "nt":
        # Prefer the bundled .exe under node_modules, which the .cmd
        # wrapper would otherwise launch with mangled argv.
        npm_bin = os.environ.get("APPDATA")
        if npm_bin:
            for variant in ("opencode-windows-x64", "opencode-windows-x64-baseline"):
                candidate = (
                    Path(npm_bin) / "npm" / "node_modules" / "opencode-ai"
                    / "node_modules" / variant / "bin" / "opencode.exe"
                )
                try:
                    if candidate.exists():
                        return str(candidate)
                except OSError:
                    continue
            # Fallback: the .cmd shim. Multi-line prompts will be
            # truncated, but at least the binary is found.
            cmd_candidate = Path(npm_bin) / "npm" / "opencode.cmd"
            try:
                if cmd_candidate.exists():
                    return str(cmd_candidate)
            except OSError:
                return str(cmd_candidate)
    found = shutil.which("opencode")
    if found:
        return found
    return "opencode"


class OpenCodeAgentBackend(AgentBackend):
    """Agent backend that runs the opencode CLI as a subprocess."""

    cost_model = "per_token"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        agent: str = "build",
        small_model: Optional[str] = None,
        auto_approve: bool = True,
        pure: bool = True,
        system_prompt: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        system_prompt
            Optional override for the agent's system prompt. When set,
            generated as ``agent.<name>.prompt`` in opencode.json so it
            replaces opencode's stock ``build`` persona without taking
            away any tools. See ``GROUNDHOG_OPENCODE_PROMPT`` for the
            recommended Groundhog default — pass it explicitly to opt in.
        """
        self.model = model
        self.agent = agent
        self.small_model = small_model
        self.auto_approve = auto_approve
        self.pure = pure
        self.system_prompt = system_prompt

    def run(self, spec: AgentSpec) -> AgentResult:
        server = None
        config_snapshot = None
        bin_dir: Optional[Path] = None
        try:
            # Allocate the wrapper bin under %TEMP% to match the convention
            # used by claude_code/copilot/codex_cli — keeps the attempt
            # workspace clean of generated scaffolding. Path is fed into
            # _build_config so opencode's permission allow-rules whitelist
            # bash invocations against that path.
            bin_dir = Path(tempfile.mkdtemp(prefix="opencode_tools_"))
            config_snapshot = self._write_workspace_config(spec, bin_dir)
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
            if config_snapshot is not None:
                self._restore_workspace_config(config_snapshot)
            if bin_dir is not None:
                cleanup_wrappers(bin_dir)

    def _start_tool_server(self, spec: AgentSpec) -> Optional[ToolServer]:
        if not spec.tools:
            return None
        server = ToolServer(spec.tools)
        server.start()
        return server

    def _write_workspace_config(
        self,
        spec: AgentSpec,
        bin_dir: Optional[Path] = None,
    ) -> tuple[Path, Optional[bytes]]:
        """Install a temporary per-attempt OpenCode config and remember prior state."""
        config_path = spec.workspace_path / "opencode.json"
        previous = config_path.read_bytes() if config_path.exists() else None
        config_path.write_text(
            json.dumps(self._build_config(spec, bin_dir=bin_dir), indent=2),
            encoding="utf-8",
        )
        return config_path, previous

    def _restore_workspace_config(self, snapshot: tuple[Path, Optional[bytes]]) -> None:
        config_path, previous = snapshot
        try:
            if previous is None:
                config_path.unlink(missing_ok=True)
            else:
                config_path.write_bytes(previous)
        except OSError:
            pass

    def _build_env(self, spec: AgentSpec, bin_dir: Path, port: Optional[int]) -> dict:
        env = os.environ.copy()
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
        if port is not None:
            env["TOOL_SERVER_PORT"] = str(port)
        if os.name == "nt":
            env.setdefault("PYTHONIOENCODING", "utf-8")

        env["OPENCODE_CONFIG"] = str((spec.workspace_path / "opencode.json").resolve())
        env.update(spec.env)
        return env

    def _build_config(self, spec: AgentSpec, bin_dir: Optional[Path] = None) -> dict:
        model = spec.model or self.model
        config = {
            "$schema": "https://opencode.ai/config.json",
            "model": model,
            "small_model": self.small_model or model,
            "provider": self._provider_config(model, spec),
            "permission": self._permission_config(spec, bin_dir=bin_dir),
        }

        # Emit a custom agent block when:
        #   (a) the user asked for a non-default agent name, OR
        #   (b) we want to override opencode's stock system prompt.
        # Opencode supports overriding the built-in ``build`` agent's
        # config by defining ``agent.build`` in opencode.json, so a
        # system_prompt override works for both default and custom names.
        needs_agent_block = self.agent != "build" or self.system_prompt
        if needs_agent_block:
            agent_entry = {
                "description": "Run a Groundhog research attempt with local evaluation tools.",
                "mode": "primary",
                "model": model,
                "permission": self._permission_config(spec, bin_dir=bin_dir),
            }
            if self.system_prompt:
                agent_entry["prompt"] = self.system_prompt
            if self.agent != "build":
                config["default_agent"] = self.agent
            config["agent"] = {self.agent: agent_entry}

        return config

    def _provider_config(self, model: str, spec: AgentSpec) -> dict:
        """Add OpenRouter model/api-key hints without requiring global setup."""
        if not model.startswith("openrouter/"):
            return {}

        model_id = model.split("/", 1)[1]
        provider = {
            "models": {
                model_id: {},
            },
        }
        if os.environ.get("OPENROUTER_API_KEY") or "OPENROUTER_API_KEY" in spec.env:
            provider["options"] = {"apiKey": "{env:OPENROUTER_API_KEY}"}
        return {"openrouter": provider}

    def _permission_config(self, spec: AgentSpec, bin_dir: Optional[Path] = None) -> dict:
        bash_rules = self._bash_permission_config(spec)
        # Match invocations against the actual bin_dir path (allocated by run()
        # under %TEMP%). Both forward- and back-slash variants land in the
        # rules so we cover bash- and PowerShell-style invocations alike.
        bin_paths: list[str] = []
        if bin_dir is not None:
            forward = str(bin_dir).replace("\\", "/")
            backward = str(bin_dir).replace("/", "\\")
            bin_paths.extend([forward, backward])
        for tool in spec.tools:
            name = tool.name
            for prefix in bin_paths:
                fwd = prefix.replace("\\", "/")
                bwd = prefix.replace("/", "\\")
                bash_rules[f"{fwd}/{name}*"] = "allow"
                bash_rules[f"{bwd}\\{name}*"] = "allow"
                bash_rules[f"& {bwd}\\{name}.ps1*"] = "allow"
                bash_rules[f"python {fwd}/{name}.py*"] = "allow"
                bash_rules[f"python {bwd}\\{name}.py*"] = "allow"
                bash_rules[f"py {bwd}\\{name}.py*"] = "allow"
            # Also keep PATH-resolved bare-name invocations working — bin_dir
            # is on the spawned shell's PATH so the agent may call wrappers
            # without a path prefix.
            bash_rules[f"{name}*"] = "allow"
            bash_rules[f"{name}.ps1*"] = "allow"
            bash_rules[f"{name}.cmd*"] = "allow"
            bash_rules[f"& {name}.ps1*"] = "allow"

        # OpenCode read/external_directory/list patterns match against the
        # ABSOLUTE file path (per opencode docs), and `*` in patterns is
        # `.*` in regex with no globstar support — so relative-style rules
        # like "./**" never match anything. Pass absolute_only=True for the
        # path-based permissions so we emit only resolved absolute patterns.
        # Edit uses workspace-relative paths (workspace_scoped=True).
        read_rules = self._path_permission_config(
            spec, ("Read",), default="allow", absolute_only=True,
        )
        return {
            "read": read_rules,
            "grep": "allow",
            "glob": "allow",
            "list": read_rules,
            "edit": self._path_permission_config(
                spec,
                ("Write", "Edit"),
                default="allow",
                workspace_scoped=True,
            ),
            "bash": bash_rules,
            "external_directory": read_rules,
            "webfetch": "deny",
            "websearch": "deny",
            "task": "deny",
            "skill": "deny",
            "question": "deny",
        }

    def _path_permission_config(
        self,
        spec: AgentSpec,
        prefixes: tuple,
        default: object,
        absolute_only: bool = False,
        workspace_scoped: bool = False,
    ):
        rules = {}
        denied = list(spec.denied_tools)
        allowed = list(spec.allowed_tools)

        broad_deny = any(_rule_inner(rule, prefixes) == "*" for rule in denied)
        broad_allow = any(_rule_inner(rule, prefixes) == "*" for rule in allowed)
        if broad_deny:
            rules["*"] = "deny"
        elif broad_allow:
            rules["*"] = "allow"

        for rule in allowed:
            inner = _rule_inner(rule, prefixes)
            if inner and inner != "*":
                for variant in _path_rule_variants(
                    inner,
                    spec.workspace_path,
                    absolute_only,
                    workspace_scoped,
                ):
                    rules[variant] = "allow"

        for rule in denied:
            inner = _rule_inner(rule, prefixes)
            if inner and inner != "*":
                for variant in _path_rule_variants(
                    inner,
                    spec.workspace_path,
                    absolute_only,
                    workspace_scoped,
                ):
                    rules[variant] = "deny"

        if not rules:
            return default
        if set(rules.keys()) == {"*"}:
            return rules["*"]
        return rules

    def _bash_permission_config(self, spec: AgentSpec) -> dict:
        rules = {"*": "deny"}

        for rule in spec.allowed_tools:
            inner = _rule_inner(rule, ("Bash",))
            if inner:
                rules[inner] = "allow"

        for rule in spec.denied_tools:
            inner = _rule_inner(rule, ("Bash",))
            if inner:
                rules[inner] = "deny"

        return rules

    def _build_prompt(self, spec: AgentSpec) -> str:
        workspace = spec.workspace_path.resolve()
        prompt = (
            "## Workspace\n"
            f"The workspace root for this run is: {workspace}\n"
            f"Use only paths under {workspace} for write/edit operations.\n"
            f"The solution work file is exactly: {workspace / 'work' / 'solution.py'}\n\n"
            + spec.goal
        )
        docs = build_tool_docs(spec.tools)
        if docs:
            prompt += "\n\n" + docs
            prompt += (
                "\nUse these bash tools for Groundhog evaluation and progress checks. "
                "On Windows, prefer PowerShell paths like "
                "`& .\\.groundhog_tools\\evaluate.ps1 --path solution.py`."
            )

        if spec.denied_tools:
            lines = ["\n\n## Restrictions",
                     "You MUST NOT use the following tools or actions:"]
            for rule in spec.denied_tools:
                lines.append(f"- {rule}")
            prompt += "\n".join(lines)

        return prompt

    def _build_command(self, spec: AgentSpec) -> list:
        model = spec.model or self.model
        prompt = self._build_prompt(spec)

        cmd = [
            _resolve_opencode_bin(), "run",
            "--format", "json",
            "--model", model,
            "--agent", self.agent,
            "--dir", str(spec.workspace_path),
        ]

        if self.pure:
            cmd.append("--pure")

        if spec.session_id:
            cmd += ["--session", spec.session_id]

        if self.auto_approve:
            cmd.append("--dangerously-skip-permissions")

        cmd.append(prompt)
        return cmd

    def _run_subprocess(self, cmd: list, env: dict, spec: AgentSpec) -> List[dict]:
        """Run opencode JSONL output, writing raw and summarized logs.

        OpenCode's JSON mode may emit only coarse step events, sometimes only
        after the model run completes. Use subprocess.run(timeout=...) instead
        of blocking on stdout iteration so Groundhog can enforce timeouts.
        """
        jsonl_path = spec.workspace_path / "agent_steps.jsonl"
        summary_path = spec.workspace_path / "agent_summary.jsonl"

        events: List[dict] = []
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                cwd=str(spec.workspace_path),
                env=env,
                encoding="utf-8",
                errors="replace",
            )
            try:
                stdout, _ = proc.communicate(timeout=spec.timeout)
            except subprocess.TimeoutExpired:
                _stop_process_tree(proc)
                raise TimeoutError(f"Agent timed out after {spec.timeout}s")

            with open(jsonl_path, "a", encoding="utf-8") as raw_file, \
                 open(summary_path, "a", encoding="utf-8") as summary_file:
                prompt_event = {
                    "type": "user",
                    "subtype": "initial_prompt",
                    "message": {"role": "user", "content": spec.goal},
                }
                raw_file.write(json.dumps(prompt_event) + "\n")
                raw_file.flush()
                summary_file.write(json.dumps({"role": "user", "content": spec.goal}) + "\n")
                summary_file.flush()

                for line in stdout.splitlines():
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

            if proc.returncode != 0 and not events:
                raise RuntimeError(f"opencode exited with code {proc.returncode}")
        except Exception:
            raise

        return events

    def _parse_result(self, events: List[dict]) -> AgentResult:
        session_id = None
        output_parts: List[str] = []
        steps = _extract_steps(events)
        failed_msg: Optional[str] = None
        turns = 0
        cost = 0.0

        for ev in events:
            session_id = session_id or _event_session_id(ev)
            etype = ev.get("type", "")
            if etype in ("step_start", "turn.started"):
                turns += 1
            if etype in ("error", "turn.failed"):
                failed_msg = _event_error(ev) or "opencode error"

            text = _event_text(ev)
            if text:
                output_parts.append(text)

            usage = _event_usage(ev)
            if usage:
                cost += float(usage.get("cost", usage.get("cost_usd", 0.0)) or 0.0)

        output = "\n".join(p for p in output_parts if p).strip()
        return AgentResult(
            success=failed_msg is None,
            output=output,
            session_id=session_id,
            cost=cost,
            turns=turns,
            duration_ms=0,
            error=failed_msg,
            steps=steps,
        )


def _rule_inner(rule: str, prefixes: tuple) -> Optional[str]:
    """Return the parenthesized payload for Claude-style permission rules."""
    for prefix in prefixes:
        marker = f"{prefix}("
        if rule.startswith(marker) and rule.endswith(")"):
            return rule[len(marker):-1]
    return None


def _path_rule_variants(
    inner: str,
    workspace_path: Path,
    absolute_only: bool = False,
    workspace_scoped: bool = False,
) -> List[str]:
    """Generate OpenCode permission patterns from a Claude-style rule inner.

    OpenCode's pattern matcher treats `*` as `.*` regex with no globstar
    support, and read/list/external_directory match against absolute paths
    while edit matches against workspace-relative paths. We translate the
    Claude-style inner into a set of variants that cover both shapes:

    - ``absolute_only=True`` (read/list): emit absolute paths only, with
      the ``./`` / ``../`` prefix resolved against the workspace.
    - ``workspace_scoped=True`` (edit): emit paths relative to the git
      worktree root or the workspace, plus a few common decorations.
    - default: emit the original inner plus relative + absolute variants.

    For ``./<x>`` and bare relative ``<x>`` we resolve to the
    workspace-anchored absolute path. For ``../<x>`` we resolve through
    ``os.path.normpath`` so the literal ``..`` doesn't survive into the
    pattern (opencode would not normalize it during match).
    """
    import os

    workspace_abs = workspace_path.resolve()

    def _both_slashes(p: str) -> List[str]:
        bwd = p.replace("/", "\\")
        fwd = p.replace("\\", "/")
        return [bwd, fwd]

    def _abs_from_relative(rel: str) -> str:
        """Resolve workspace-relative ``rel`` (no leading ``./``) to an
        absolute path string. ``rel`` may be empty (workspace root)."""
        joined = workspace_abs if not rel else workspace_abs / rel
        # Use normpath rather than .resolve() — the path may not exist
        # yet, and we only need the textual form for pattern matching.
        return os.path.normpath(str(joined))

    # Classify the input.
    if ":" in inner or inner.startswith(("/", "\\")):
        # Already absolute — normalize and pass through.
        abs_form = os.path.normpath(inner)
        rel_form = None
    elif inner in (".", "./", ".\\"):
        abs_form = str(workspace_abs)
        rel_form = ""
    elif inner.startswith(("./", ".\\")):
        rel = inner[2:]
        abs_form = _abs_from_relative(rel)
        rel_form = rel
    elif inner in ("..", "../", "..\\"):
        abs_form = os.path.normpath(str(workspace_abs / ".."))
        rel_form = None  # outside workspace
    elif inner.startswith(("../", "..\\")):
        abs_form = os.path.normpath(str(workspace_abs / inner))
        rel_form = None
    else:
        # Bare relative ("work/*", "**") — treat as workspace-relative.
        abs_form = _abs_from_relative(inner)
        rel_form = inner

    variants: List[str] = []
    if absolute_only:
        # Read / external_directory: absolute paths only. Strip any
        # trailing ``/**`` or ``/*`` from the resolved absolute path —
        # opencode's ``*`` already acts as a regex ``.*`` (matches across
        # path separators), so the suffixes are redundant and would not
        # match the directory itself otherwise. Emit:
        #   - <base>            → matches the dir itself
        #   - <base>\* / <base>/*  → matches direct + transitive descendants
        #   - <base>*           → matches dir + prefix collisions (broader)
        base = abs_form
        while True:
            if base.endswith(("\\**", "/**")):
                base = base[:-3]
            elif base.endswith(("\\*", "/*")):
                base = base[:-2]
            else:
                break
        variants.extend(_both_slashes(base))
        variants.extend(_both_slashes(base + os.sep + "*"))
        variants.extend(_both_slashes(base + "*"))
    elif workspace_scoped:
        # Edit: workspace-relative paths. If the workspace is nested under
        # a git root, opencode anchors edit at the git root — adjust by
        # prepending the path from git root to the workspace.
        if rel_form is not None:
            outer_relative = _relative_to_outer_project(
                Path(abs_form), workspace_path
            )
            if outer_relative:
                variants.extend(_both_slashes(outer_relative))
                variants.extend(_both_slashes(f"./{outer_relative}"))
            elif rel_form:
                variants.extend(_both_slashes(rel_form))
                variants.extend(_both_slashes(f"./{rel_form}"))
            else:
                variants.extend([".", "./"])
        else:
            # Outside-workspace edit rules — fall back to absolute.
            variants.extend(_both_slashes(abs_form))
    else:
        # Default (legacy): emit raw + relative + absolute forms.
        variants.append(inner)
        if rel_form is not None:
            variants.extend(_both_slashes(f"./{rel_form}" if rel_form else "."))
        variants.extend(_both_slashes(abs_form))

    return list(dict.fromkeys(v for v in variants if v))


def _relative_to_outer_project(abs_path: Path, workspace_path: Path) -> Optional[str]:
    """Return path relative to an enclosing project root, if one exists.

    OpenCode may anchor tools at a parent git repository even when ``--dir`` is
    an attempt workspace. In that case ``work/*`` would target repo-root
    ``work/*``. Emit ``attempt_x/work/*`` instead.
    """
    outer = _find_enclosing_git_root(workspace_path)
    if not outer:
        return None
    try:
        return str(abs_path.resolve().relative_to(outer.resolve()))
    except ValueError:
        return None


def _find_enclosing_git_root(workspace_path: Path) -> Optional[Path]:
    current = workspace_path.resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return None


def _stop_process_tree(proc: subprocess.Popen) -> None:
    """Terminate only the OpenCode process tree started by this backend."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            return
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass


def _event_session_id(event: dict) -> Optional[str]:
    for key in ("sessionID", "session_id", "sessionId"):
        if event.get(key):
            return event[key]
    data = event.get("data", {}) or {}
    for key in ("sessionID", "session_id", "sessionId"):
        if data.get(key):
            return data[key]
    part = event.get("part", {}) or {}
    for key in ("sessionID", "session_id", "sessionId"):
        if part.get(key):
            return part[key]
    return None


def _event_text(event: dict) -> str:
    etype = event.get("type")
    part = event.get("part", {}) or {}
    if part.get("type") in ("text", "message", "assistant-message"):
        return str(part.get("text", part.get("content", "")))

    if etype in ("text", "message"):
        return str(event.get("text", event.get("content", "")))
    if etype == "assistant.message":
        return str((event.get("data", {}) or {}).get("content", ""))

    item = event.get("item", {}) or {}
    if item.get("type") in ("text", "agent_message", "assistant-message"):
        return str(item.get("text", item.get("content", "")))

    return ""


def _event_error(event: dict) -> str:
    err = event.get("error") or event.get("data", {}).get("error")
    if isinstance(err, dict):
        return str(err.get("message", err))
    if err:
        return str(err)
    return str(event.get("message", ""))


def _event_usage(event: dict) -> dict:
    for key in ("usage", "stats"):
        value = event.get(key)
        if isinstance(value, dict):
            return value
    data = event.get("data", {}) or {}
    for key in ("usage", "stats"):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    part = event.get("part", {}) or {}
    tokens = part.get("tokens")
    if isinstance(tokens, dict) or part.get("cost") is not None:
        usage = {}
        if isinstance(tokens, dict):
            usage.update({
                "input_tokens": tokens.get("input"),
                "output_tokens": tokens.get("output"),
                "reasoning_output_tokens": tokens.get("reasoning"),
                "cached_input_tokens": (tokens.get("cache") or {}).get("read")
                if isinstance(tokens.get("cache"), dict) else None,
            })
        if part.get("cost") is not None:
            usage["cost"] = part.get("cost")
        return usage
    for key in ("usage", "stats"):
        value = part.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _summarize_event(event: dict) -> List[dict]:
    etype = event.get("type")
    session_id = _event_session_id(event)

    if etype == "step_start":
        return [{"role": "system", "type": "step_start", "session_id": session_id}]

    if etype in ("text", "message", "assistant.message"):
        text = _event_text(event)
        return [{"role": "assistant", "type": "text", "content": text}] if text.strip() else []

    if etype == "tool_use":
        part = event.get("part", {}) or {}
        if part.get("type") == "tool":
            state = part.get("state", {}) or {}
            return [{
                "role": "assistant",
                "type": "tool_use",
                "tool": part.get("tool", "unknown"),
                "input": state.get("input", {}),
                "output": str(state.get("output", ""))[:500],
                "status": state.get("status"),
            }]
        return [{
            "role": "assistant",
            "type": "tool_use",
            "tool": event.get("tool", event.get("name", "unknown")),
            "input": event.get("input", event.get("args", {})),
        }]

    if etype in ("tool_result", "tool"):
        return [{
            "role": "tool_result",
            "tool_use_id": event.get("tool_use_id", event.get("id")),
            "content": event.get("content", event.get("output", "")),
        }]

    if etype in ("step_finish", "turn.completed", "result"):
        usage = _event_usage(event)
        return [{
            "role": "result",
            "type": etype,
            "session_id": session_id,
            "cost_usd": usage.get("cost", usage.get("cost_usd")),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
        }]

    if etype in ("error", "turn.failed"):
        return [{"role": "result", "type": "error", "error": _event_error(event)}]

    part = event.get("part", {}) or {}
    if part.get("type") == "text":
        text = _event_text(event)
        return [{"role": "assistant", "type": "text", "content": text}] if text.strip() else []

    return []


def _extract_steps(events: List[dict]) -> List[dict]:
    max_text = 500
    steps: List[dict] = []

    for event in events:
        etype = event.get("type")
        text = _event_text(event)
        if text:
            steps.append({
                "type": "text",
                "text": text[:max_text] + ("..." if len(text) > max_text else ""),
            })
            continue

        if etype == "tool_use":
            part = event.get("part", {}) or {}
            if part.get("type") == "tool":
                state = part.get("state", {}) or {}
                output = str(state.get("output", ""))
                steps.append({
                    "type": "tool_use",
                    "tool": part.get("tool", "unknown"),
                    "input": _truncate_input(state.get("input", {}), max_text),
                    "output": output[:max_text] + ("..." if len(output) > max_text else ""),
                    "status": state.get("status"),
                })
                continue
            tool_input = event.get("input", event.get("args", {}))
            steps.append({
                "type": "tool_use",
                "tool": event.get("tool", event.get("name", "unknown")),
                "input": _truncate_input(tool_input, max_text),
            })
            continue

        if etype in ("tool_result", "tool"):
            output = str(event.get("content", event.get("output", "")))
            steps.append({
                "type": "tool_result",
                "output": output[:max_text] + ("..." if len(output) > max_text else ""),
            })

    return steps


def _truncate_input(input_dict: object, max_len: int) -> dict:
    if not isinstance(input_dict, dict):
        return {}
    return {
        k: (v[:max_len] + "..." if isinstance(v, str) and len(v) > max_len else v)
        for k, v in input_dict.items()
    }
