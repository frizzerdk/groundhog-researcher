"""Subprocess runner — execute user code in an isolated process.

Runs code strings in a subprocess with timeout and optional memory limits.
Input goes to the child as pickle over stdin; the result comes back through a
dedicated temp file, NOT stdout — user code prints freely (LLM-generated
solutions almost always do) without corrupting the return channel.
"""

import os
import pickle
import subprocess
import sys
import tempfile
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
    result_fd, result_path = tempfile.mkstemp(suffix=".pkl", prefix="groundhog-run-")
    os.close(result_fd)

    payload = {
        "code": code,
        "entry_point": entry_point,
        "args": args,
        "kwargs": kwargs or {},
        "imports": imports or {},
        "result_path": result_path,
    }

    # The child writes its result to result_path — stdout stays free for user
    # prints. (Regression: this used to be pickle-over-stdout, so any print()
    # in evaluated code corrupted the result. Audit 2026-07-01, bug #3.)
    script = '''
import sys, pickle, importlib

payload = pickle.loads(sys.stdin.buffer.read())

ns = {}
for name, module_path in payload["imports"].items():
    ns[name] = importlib.import_module(module_path)

exec(payload["code"], ns)
func = ns[payload["entry_point"]]

result = func(*payload["args"], **payload["kwargs"])
with open(payload["result_path"], "wb") as f:
    pickle.dump(result, f)
'''

    preexec = _make_memory_limiter(memory_limit_mb) if memory_limit_mb else None

    try:
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
            stderr = proc.stderr.decode(errors="replace").strip()
            stdout_tail = proc.stdout.decode(errors="replace").strip()[-500:]
            detail = stderr or "Subprocess failed"
            if stdout_tail:
                detail += f"\n--- subprocess stdout (tail) ---\n{stdout_tail}"
            raise RuntimeError(detail)

        try:
            with open(result_path, "rb") as f:
                data = f.read()
        except OSError:
            data = b""
        if not data:
            stderr = proc.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"No result from subprocess. stderr: {stderr}")

        return pickle.loads(data)
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass
