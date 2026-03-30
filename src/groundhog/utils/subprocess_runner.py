"""Subprocess runner — execute user code in an isolated process.

Runs code strings in a subprocess with timeout and optional memory limits.
Uses pickle for I/O via stdin/stdout. The subprocess can't corrupt the optimizer's process.
"""

import base64
import pickle
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple


_VMAP_HEADROOM_MB = 4096


def _make_memory_limiter(memory_limit_mb: int):
    def _set_limit():
        import resource
        limit_bytes = (memory_limit_mb + _VMAP_HEADROOM_MB) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    return _set_limit


def run_code(
    code: str,
    entry_point: str,
    args: Tuple = (),
    kwargs: Dict[str, Any] = None,
    imports: Dict[str, str] = None,
    timeout: Optional[int] = None,
    memory_limit_mb: Optional[int] = None,
) -> Any:
    """Run user code in a subprocess. Returns the entry_point's return value.

    Args:
        code: Python source code to execute
        entry_point: Function name to call after exec (e.g. "run")
        args: Positional arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        imports: Dict mapping names to module paths (e.g. {"np": "numpy"})
        timeout: Hard timeout in seconds (subprocess is killed)
        memory_limit_mb: Max virtual memory in MB (Linux only)

    Returns:
        Whatever the entry_point function returns (must be picklable)

    Raises:
        TimeoutError: If timeout exceeded
        RuntimeError: If code fails (syntax, runtime, etc.)
    """
    payload = {
        "code": code,
        "entry_point": entry_point,
        "args": args,
        "kwargs": kwargs or {},
        "imports": imports or {},
    }

    script = '''
import sys, pickle, base64, importlib, time

payload = pickle.loads(sys.stdin.buffer.read())

ns = {}
for name, module_path in payload["imports"].items():
    ns[name] = importlib.import_module(module_path)

exec(payload["code"], ns)
func = ns[payload["entry_point"]]

result = func(*payload["args"], **payload["kwargs"])
sys.stdout.buffer.write(pickle.dumps(result))
'''

    preexec = _make_memory_limiter(memory_limit_mb) if memory_limit_mb else None

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            input=pickle.dumps(payload),
            capture_output=True,
            timeout=timeout,
            preexec_fn=preexec,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Timed out after {timeout}s")

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode().strip() or "Subprocess failed")

    if not proc.stdout:
        raise RuntimeError(f"No output from subprocess. stderr: {proc.stderr.decode().strip()}")

    return pickle.loads(proc.stdout)
