"""Claude Code CLI backend. Zero-config for anyone with Claude Code installed."""

import json
import subprocess
import sys
import threading
import time

from groundhog.base.backend import LLMBackend, LLMResponse, Prompt, TextPart


class ClaudeCodeBackend(LLMBackend):
    """Claude via the Claude Code CLI. No API key needed -- uses CLI auth.

    Models: "opus", "sonnet", "haiku" (or full model names).
    Effort: "low", "medium", "high" -- caps extended thinking budget.
    """

    def __init__(self, model: str = "sonnet", effort: str = "low", warn_interval: int = 30, max_retries: int = 2):
        self.model = model
        self.effort = effort
        self.warn_interval = warn_interval
        self.max_retries = max_retries

    def generate(self, prompt: Prompt, system_prompt: str = "") -> LLMResponse:
        errors = []
        for attempt in range(self.max_retries + 1):
            try:
                result = self._call(prompt, system_prompt)
                if errors:
                    result.usage["retries"] = len(errors)
                    result.usage["retry_errors"] = [str(e) for e in errors]
                return result
            except RuntimeError as e:
                errors.append(e)
                if attempt < self.max_retries:
                    wait = 5 * (attempt + 1)
                    print(f"(error: {e}, retrying in {wait}s)... ", end="", file=sys.stderr, flush=True)
                    time.sleep(wait)
        raise errors[-1]

    def _call(self, prompt: Prompt, system_prompt: str = "") -> LLMResponse:
        prompt_text = prompt if isinstance(prompt, str) else " ".join(
            p.text for p in prompt if isinstance(p, TextPart))

        # stream-json gives incremental output so we can detect progress
        cmd = ["claude", "-p", "--output-format", "stream-json", "--verbose",
               "--tools", "",
               "--effort", self.effort,
               "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
               "--model", self.model]
        if system_prompt:
            cmd += ["--system-prompt", system_prompt]

        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        except FileNotFoundError as e:
            if getattr(e, 'winerror', None) == 206:
                raise RuntimeError(f"Command line too long ({len(prompt_text)} chars)")
            raise RuntimeError("Claude Code CLI not found. Install from https://claude.ai/code")

        # Read stdout line by line (NDJSON) to track progress
        lines = []
        bytes_received = [0]

        def read_stdout():
            for line in proc.stdout:
                lines.append(line)
                bytes_received[0] += len(line)

        stderr_chunks = []

        def read_stderr():
            while True:
                chunk = proc.stderr.read(4096)
                if not chunk:
                    break
                stderr_chunks.append(chunk)

        t_out = threading.Thread(target=read_stdout, daemon=True)
        t_err = threading.Thread(target=read_stderr, daemon=True)
        t_out.start()
        t_err.start()

        # Write prompt and close stdin
        proc.stdin.write(prompt_text.encode("utf-8"))
        proc.stdin.close()

        # Wait with periodic status updates (overwrite in place)
        start = time.time()
        last_bytes = 0
        last_status_len = 0
        while proc.poll() is None:
            t_out.join(timeout=self.warn_interval)
            if proc.poll() is None:
                elapsed = int(time.time() - start)
                current_bytes = bytes_received[0]
                status = "working" if current_bytes > last_bytes else "waiting"
                msg = f"({status} {elapsed}s)... "
                # Backspace over previous status, write new one
                print("\b" * last_status_len + msg, end="", file=sys.stderr, flush=True)
                last_status_len = len(msg)
                last_bytes = current_bytes

        # Clear status
        if last_status_len:
            print("\b" * last_status_len + " " * last_status_len + "\b" * last_status_len,
                  end="", file=sys.stderr, flush=True)

        t_out.join(timeout=5)
        t_err.join(timeout=5)


        stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")

        if proc.returncode != 0:
            raise RuntimeError(f"Claude Code CLI error: {stderr.strip()}")

        # Parse NDJSON — find the result event
        result_data = None
        for line in lines:
            line_str = line.decode("utf-8", errors="replace").strip()
            if not line_str:
                continue
            try:
                event = json.loads(line_str)
                if event.get("type") == "result":
                    result_data = event
            except json.JSONDecodeError:
                continue

        if not result_data:
            raise RuntimeError("Claude Code CLI returned no result event")

        return LLMResponse(
            text=result_data.get("result", ""),
            model=self.model,
            usage=result_data.get("usage", {}),
            cost=result_data.get("total_cost_usd", 0.0),
        )
