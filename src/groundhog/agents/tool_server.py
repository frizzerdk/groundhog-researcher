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

def generate_wrappers(tools: List[AgentTool], bin_dir: Path, port: int) -> None:
    """Generate bash wrapper scripts for each tool in bin_dir."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    python_path = sys.executable

    for tool in tools:
        ordered_names, required_count, defaults, path_params = _get_ordered_params(tool)
        script = _build_wrapper_script(
            tool.name, ordered_names, required_count, defaults, path_params, port, python_path,
        )
        script_path = bin_dir / tool.name
        script_path.write_text(script)
        script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _build_wrapper_script(
    tool_name: str,
    param_names: List[str],
    required_count: int,
    defaults: Dict[str, Any],
    path_params: set,
    port: int,
    python_path: str = "python3",
) -> str:
    """Build a bash wrapper script that supports both positional and --kwargs.

    Auto-detects mode: if any argument starts with --, uses kwargs mode.
    Otherwise falls back to positional mode.
    Params with type "path" are resolved to absolute paths from the agent's cwd.
    """
    param_names_repr = repr(param_names)
    defaults_repr = repr(defaults)
    path_params_repr = repr(path_params)

    # Build usage strings
    usage_positional = " ".join(
        [f"<{n}>" for n in param_names[:required_count]]
        + [f"[{n}]" for n in param_names[required_count:]]
    )
    usage_kwargs = " ".join(f"--{n} <val>" for n in param_names)

    parts = [
        '#!/usr/bin/env bash',
        'set -euo pipefail',
    ]

    if required_count > 0:
        parts.append(f'''
if [ $# -lt {required_count} ] && ! echo "${{@}}" | grep -q "\\-\\-"; then
    echo "Usage: {tool_name} {usage_positional}" >&2
    echo "       {tool_name} {usage_kwargs}" >&2
    exit 1
fi''')

    parts.append(f'''
JSON=$({python_path} -c "
import json, sys, os
names = {param_names_repr}
defaults = {defaults_repr}
path_params = {path_params_repr}
args = sys.argv[1:]
params = {{}}

if any(a.startswith('--') for a in args):
    # kwargs mode
    i = 0
    while i < len(args):
        if args[i].startswith('--') and i + 1 < len(args):
            params[args[i][2:]] = args[i + 1]
            i += 2
        else:
            i += 1
    for name in names:
        if name not in params and name in defaults:
            params[name] = defaults[name]
else:
    # positional mode
    for i, name in enumerate(names):
        if i < len(args):
            params[name] = args[i]
        elif name in defaults:
            params[name] = defaults[name]

# Resolve path params to absolute (agent cwd may differ from tool server cwd)
for name in path_params:
    if name in params and isinstance(params[name], str):
        params[name] = os.path.abspath(params[name])

print(json.dumps(params))
" "$@")

curl -s -X POST "http://127.0.0.1:{port}/{tool_name}" \\
    -H "Content-Type: application/json" -d "$JSON" \\
| {python_path} -c "
import sys, json
r = json.load(sys.stdin)
if r['success']:
    print(r['output'])
else:
    print(r.get('error', 'Error'), file=sys.stderr)
    sys.exit(1)
"''')

    return "\n".join(parts) + "\n"


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
