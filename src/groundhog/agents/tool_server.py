"""HTTP tool server and bash wrapper generation for agent tool access.

Starts a localhost HTTP server exposing AgentTool instances as POST endpoints.
Generates bash wrapper scripts that agents call via their shell tool.

Bash wrappers support both positional and --kwargs modes:
    get-learnings 5 3                    # positional
    get-learnings --last 5 --random 3    # named (preferred by LLM agents)

Ported from EvaluatableExperiments/src/agents/implementations/tool_server.py
with --kwargs support added.
"""

import json
import shutil
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

from groundhog.base.agent import AgentTool


class ToolServer:
    """Localhost HTTP server exposing tools as POST endpoints.

    Runs in a daemon thread. Each tool is accessible at POST /{tool_name}.
    """

    def __init__(self, tools: List[AgentTool]):
        self._tools = {tool.name: tool for tool in tools}
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self.port: Optional[int] = None

    def start(self) -> int:
        handler_class = _make_handler(self._tools)
        self._server = HTTPServer(("127.0.0.1", 0), handler_class)
        self.port = self._server.server_address[1]

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._ready.set()
        return self.port

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None


def _make_handler(tools: Dict[str, AgentTool]) -> type:
    """Build a request handler class with tools baked in."""

    class ToolHandler(BaseHTTPRequestHandler):
        _tools = tools

        def log_message(self, format, *args):
            pass  # Suppress default stderr logging

        def do_POST(self):
            tool_name = self.path.lstrip("/")

            if tool_name not in self._tools:
                self._send_error(404, f"Unknown tool: {tool_name}")
                return

            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                kwargs = json.loads(body) if body else {}
            except (json.JSONDecodeError, ValueError) as e:
                self._send_error(400, f"Invalid JSON: {e}")
                return

            if not isinstance(kwargs, dict):
                self._send_error(400, "Request body must be a JSON object")
                return

            tool = self._tools[tool_name]
            try:
                result = tool.execute(**kwargs)
                response = {
                    "success": result.success,
                    "output": result.output,
                    "error": result.error,
                }
            except Exception as e:
                response = {
                    "success": False,
                    "output": "",
                    "error": str(e),
                }

            self._send_json(200, response)

        def _send_json(self, status: int, data: dict):
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, status: int, message: str):
            self._send_json(status, {"success": False, "output": "", "error": message})

    return ToolHandler


# --- Parameter ordering ---

def _get_ordered_params(tool: AgentTool) -> tuple:
    """Get parameters ordered: required first, optional last.

    Returns (ordered_names, required_count, defaults_dict, path_params).
    path_params is a set of param names with type "path" — wrappers resolve
    these to absolute paths from the agent's cwd.
    """
    params = tool.get_parameters()
    required = []
    optional = []
    defaults = {}
    path_params = set()

    for name, schema in params.items():
        if "default" in schema:
            optional.append(name)
            defaults[name] = schema["default"]
        else:
            required.append(name)
        if schema.get("type") == "path":
            path_params.add(name)

    ordered = required + optional
    return ordered, len(required), defaults, path_params


# --- Bash wrapper generation ---

def _warn_if_store_python(python_path: Path) -> None:
    """Microsoft Store Python breaks sandboxed agents' tool wrappers.

    A venv based on the Store interpreter redirects through WindowsApps,
    which AppContainer-sandboxed CLIs (codex) cannot traverse — every tool
    call then dies with "No Python at ...". Detect it up front and say so,
    because inside the agent the failure looks like a broken tool, not a
    broken interpreter.
    """
    probe = [python_path]
    cfg = python_path.parent.parent / "pyvenv.cfg"
    if cfg.is_file():
        try:
            for line in cfg.read_text(encoding="utf-8").splitlines():
                if line.strip().lower().startswith("home"):
                    probe.append(Path(line.split("=", 1)[1].strip()))
                    break
        except OSError:
            pass
    if any("windowsapps" in str(p).lower() for p in probe):
        print(
            "WARNING: agent tools run on a Microsoft Store Python "
            f"({python_path}); sandboxed agents (codex) cannot launch it. "
            "Rebase the environment on a regular interpreter, e.g.: "
            "uv python install && uv python pin && uv sync",
            file=sys.stderr,
        )


def generate_wrappers(tools: List[AgentTool], bin_dir: Path, port: int) -> None:
    """Generate wrapper scripts for each tool in bin_dir.

    Creates cross-platform Python scripts (no bash/curl dependency).
    On Unix: executable scripts with #!/usr/bin/env python3 shebang.
    On Windows: .cmd wrappers that call the Python script.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    python_path = sys.executable
    is_windows = sys.platform == "win32"

    if is_windows:
        _warn_if_store_python(Path(python_path))

    for tool in tools:
        ordered_names, required_count, defaults, path_params = _get_ordered_params(tool)
        py_script = _build_python_wrapper(
            tool.name, ordered_names, required_count, defaults, path_params, port,
        )

        if is_windows:
            # Write .py script + .cmd launcher (for cmd.exe)
            py_path = bin_dir / f"{tool.name}.py"
            py_path.write_text(py_script, encoding="utf-8")
            cmd_path = bin_dir / f"{tool.name}.cmd"
            cmd_path.write_text(
                f'@set PYTHONIOENCODING=utf-8\n@"{python_path}" "%~dp0{tool.name}.py" %*\n',
                encoding="utf-8",
            )
            # PowerShell launcher (for codex/copilot CLIs on Windows that use
            # pwsh as their shell — they don't auto-resolve .cmd via PATHEXT
            # when invoked as `pwsh -Command <name>`). Native PowerShell HTTP
            # client, no Python: sandboxed CLIs (codex AppContainer) cannot
            # launch WindowsApps- or uv-based interpreters at all.
            ps1_path = bin_dir / f"{tool.name}.ps1"
            ps1_path.write_text(
                _build_ps1_wrapper(
                    tool.name, ordered_names, required_count, defaults,
                    path_params, port,
                ),
                encoding="utf-8",
            )
            # Extensionless bash wrapper for Git Bash (Claude Code on Windows)
            bash_path = bin_dir / tool.name
            bash_python = python_path.replace("\\", "/")
            bash_path.write_text(
                f'#!/bin/bash\nexport PYTHONIOENCODING=utf-8\n'
                f'exec "{bash_python}" "$(dirname "$0")/{tool.name}.py" "$@"\n',
                encoding="utf-8",
            )
        else:
            # Write executable Python script with shebang
            script_path = bin_dir / tool.name
            script_path.write_text(f"#!/usr/bin/env python3\n{py_script}", encoding="utf-8")
            script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


_PS1_BODY = r'''
$argList = @($args | ForEach-Object { "$_" })
$kwMode = [bool]($argList | Where-Object { $_.StartsWith('--') })

if (-not $kwMode -and $argList.Count -lt $REQUIRED) {
    $usagePos = (@($NAMES[0..([Math]::Max($REQUIRED - 1, 0))] | ForEach-Object { "<$_>" }) -join ' ')
    [Console]::Error.WriteLine("Usage: $TOOL $usagePos")
    [Console]::Error.WriteLine("       $TOOL " + (@($NAMES | ForEach-Object { "--$_ <val>" }) -join ' '))
    exit 1
}

$params = @{}
if ($kwMode) {
    $i = 0
    while ($i -lt $argList.Count) {
        if ($argList[$i].StartsWith('--') -and ($i + 1) -lt $argList.Count) {
            $params[$argList[$i].Substring(2)] = $argList[$i + 1]
            $i += 2
        } else { $i += 1 }
    }
    foreach ($n in $NAMES) {
        if (-not $params.ContainsKey($n) -and $DEFAULTS.ContainsKey($n)) { $params[$n] = $DEFAULTS[$n] }
    }
} else {
    for ($i = 0; $i -lt $NAMES.Count; $i++) {
        if ($i -lt $argList.Count) { $params[$NAMES[$i]] = $argList[$i] }
        elseif ($DEFAULTS.ContainsKey($NAMES[$i])) { $params[$NAMES[$i]] = $DEFAULTS[$NAMES[$i]] }
    }
}

foreach ($n in $PATH_PARAMS) {
    if ($params.ContainsKey($n) -and $params[$n] -is [string]) {
        # Resolves relative to the PS current location and exists on 5.1
        # (the 2-arg [IO.Path]::GetFullPath overload does not).
        try { $params[$n] = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($params[$n]) }
        catch {}
    }
}

$body = if ($params.Count -gt 0) { ConvertTo-Json $params -Compress } else { '{}' }
try {
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$PORT/$TOOL" -Method Post `
        -ContentType 'application/json' -Body $body
} catch {
    [Console]::Error.WriteLine("Error calling ${TOOL}: $($_.Exception.Message)")
    exit 1
}
if ($r.success) { Write-Output $r.output } else { [Console]::Error.WriteLine("$($r.error)"); exit 1 }
'''


def _ps_literal(v) -> str:
    if isinstance(v, bool):
        return "$true" if v else "$false"
    if v is None:
        return "$null"
    if isinstance(v, (int, float)):
        return str(v)
    return "'" + str(v).replace("'", "''") + "'"


def _build_ps1_wrapper(
    tool_name: str,
    param_names: List[str],
    required_count: int,
    defaults: Dict[str, Any],
    path_params: set,
    port: int,
) -> str:
    """Native PowerShell tool client — same protocol as the Python wrapper.

    Exists because sandboxed Windows CLIs (codex AppContainer) cannot launch
    Python interpreters that live behind WindowsApps or uv-managed dirs;
    Invoke-RestMethod needs nothing outside pwsh itself.
    """
    names = "@(" + ", ".join(_ps_literal(n) for n in param_names) + ")"
    dflts = ("@{" + "; ".join(f"{_ps_literal(k)} = {_ps_literal(v)}"
                              for k, v in defaults.items()) + "}") if defaults else "@{}"
    paths = "@(" + ", ".join(_ps_literal(p) for p in sorted(path_params)) + ")"
    header = (
        f"$NAMES = {names}\n"
        f"$DEFAULTS = {dflts}\n"
        f"$PATH_PARAMS = {paths}\n"
        f"$REQUIRED = {required_count}\n"
        f"$PORT = {port}\n"
        f"$TOOL = {_ps_literal(tool_name)}\n"
    )
    return header + _PS1_BODY


def _build_python_wrapper(
    tool_name: str,
    param_names: List[str],
    required_count: int,
    defaults: Dict[str, Any],
    path_params: set,
    port: int,
) -> str:
    """Build a cross-platform Python wrapper script.

    Supports both positional and --kwargs modes.
    Uses urllib (stdlib) instead of curl for HTTP.
    Resolves "path" type params to absolute paths.
    """
    return f'''import json, sys, os, urllib.request

NAMES = {repr(param_names)}
DEFAULTS = {repr(defaults)}
PATH_PARAMS = {repr(path_params)}
REQUIRED = {required_count}
PORT = {port}
TOOL = {repr(tool_name)}

args = sys.argv[1:]

# Check required args
if len(args) < REQUIRED and not any(a.startswith("--") for a in args):
    usage_pos = " ".join(f"<{{n}}>" for n in NAMES[:REQUIRED]) + " " + " ".join(f"[{{n}}]" for n in NAMES[REQUIRED:])
    usage_kw = " ".join(f"--{{n}} <val>" for n in NAMES)
    print(f"Usage: {{TOOL}} {{usage_pos}}", file=sys.stderr)
    print(f"       {{TOOL}} {{usage_kw}}", file=sys.stderr)
    sys.exit(1)

# Parse args
params = {{}}
if any(a.startswith("--") for a in args):
    i = 0
    while i < len(args):
        if args[i].startswith("--") and i + 1 < len(args):
            params[args[i][2:]] = args[i + 1]
            i += 2
        else:
            i += 1
    for name in NAMES:
        if name not in params and name in DEFAULTS:
            params[name] = DEFAULTS[name]
else:
    for i, name in enumerate(NAMES):
        if i < len(args):
            params[name] = args[i]
        elif name in DEFAULTS:
            params[name] = DEFAULTS[name]

# Resolve path params
for name in PATH_PARAMS:
    if name in params and isinstance(params[name], str):
        params[name] = os.path.abspath(params[name])

# POST to tool server
data = json.dumps(params).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{{PORT}}/{{TOOL}}",
    data=data,
    headers={{"Content-Type": "application/json"}},
    method="POST",
)
try:
    with urllib.request.urlopen(req) as resp:
        r = json.loads(resp.read())
except Exception as e:
    print(f"Error calling {{TOOL}}: {{e}}", file=sys.stderr)
    sys.exit(1)

if r["success"]:
    print(r["output"])
else:
    print(r.get("error", "Error"), file=sys.stderr)
    sys.exit(1)
'''


def cleanup_wrappers(bin_dir: Path) -> None:
    """Remove the wrapper scripts directory."""
    if bin_dir.exists():
        shutil.rmtree(bin_dir)


# --- Tool documentation ---

def build_tool_docs(tools: List[AgentTool]) -> str:
    """Build markdown documentation of available bash tools for agent prompts."""
    if not tools:
        return ""

    lines = ["## Available bash tools", ""]
    for tool in tools:
        ordered_names, required_count, defaults, _ = _get_ordered_params(tool)

        # Usage line
        usage_parts = (
            [f"<{n}>" for n in ordered_names[:required_count]]
            + [f"[{n}]" for n in ordered_names[required_count:]]
        )
        usage = " ".join(usage_parts)

        lines.append(f"### {tool.name}")
        lines.append(tool.description)
        if ordered_names:
            lines.append(f"Usage: {tool.name} {usage}")
            lines.append(f"       {tool.name} " + " ".join(f"--{n} <val>" for n in ordered_names))
        else:
            lines.append(f"Usage: {tool.name}")

        # Parameter details
        params = tool.get_parameters()
        for name in ordered_names:
            schema = params[name]
            desc = schema.get("description", "")
            default = schema.get("default")
            parts = [f"  {name}"]
            if desc:
                parts.append(f" — {desc}")
            if default is not None:
                parts.append(f" (default: {default})")
            lines.append("".join(parts))
        lines.append("")

    return "\n".join(lines)
