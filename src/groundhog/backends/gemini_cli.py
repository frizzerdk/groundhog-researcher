"""Gemini CLI backend. Uses gemini -p for non-interactive mode."""

import json
import subprocess
import sys
import threading
import time

from groundhog.base.backend import LLMBackend, LLMResponse, Prompt, TextPart


class GeminiCLIBackend(LLMBackend):
    """Google Gemini via the Gemini CLI. Uses Google account auth.

    Free tier: 1000 requests/day with Google sign-in.
    Install: npm install -g @google/gemini-cli
    """

    def __init__(self, model: str = "gemini-2.5-flash", warn_interval: int = 30, max_retries: int = 2):
        self.model = model
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

        # Prepend no-tools instruction to prevent gemini from using tools
        # (without this, gemini spends output tokens reasoning about tools)
        preamble = "Do not use any tools. Just respond with text."
        if system_prompt:
            prompt_text = f"{preamble}\n{system_prompt}\n\n{prompt_text}"
        else:
            prompt_text = f"{preamble}\n\n{prompt_text}"

        # -p "" + stdin for headless mode (avoids command-line length limits)
        # --approval-mode plan as safety net against tool execution
        cmd = ["gemini", "-p", "", "-m", self.model, "-o", "json",
               "--approval-mode", "plan"]

        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        except FileNotFoundError as e:
            if getattr(e, 'winerror', None) == 206:
                raise RuntimeError(f"Command line too long ({len(prompt_text)} chars)")
            raise RuntimeError("Gemini CLI not found. Install: npm install -g @google/gemini-cli")

        stdout_chunks = []
        stderr_chunks = []
        bytes_received = [0]

        def read_stream(stream, chunks, counter):
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    break
                chunks.append(chunk)
                counter[0] += len(chunk)

        t_out = threading.Thread(target=read_stream, args=(proc.stdout, stdout_chunks, bytes_received), daemon=True)
        t_err = threading.Thread(target=read_stream, args=(proc.stderr, stderr_chunks, bytes_received), daemon=True)
        t_out.start()
        t_err.start()

        proc.stdin.write(prompt_text.encode("utf-8"))
        proc.stdin.close()

        # Wait with periodic status updates
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
                print("\b" * last_status_len + msg, end="", file=sys.stderr, flush=True)
                last_status_len = len(msg)
                last_bytes = current_bytes

        if last_status_len:
            print("\b" * last_status_len + " " * last_status_len + "\b" * last_status_len,
                  end="", file=sys.stderr, flush=True)

        t_out.join(timeout=5)
        t_err.join(timeout=5)

        stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
        stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")

        if proc.returncode != 0:
            raise RuntimeError(f"Gemini CLI error: {stderr.strip()}")

        # Parse JSON output -- gemini may output NDJSON or single JSON
        stdout = stdout.strip()
        data = None
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            for line in reversed(stdout.split('\n')):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue

        if data is None:
            return LLMResponse(text=stdout, model=self.model, cost=0.0)

        text = data.get("response", data.get("result", data.get("text", stdout)))

        usage = {}
        stats = data.get("stats", {})
        for model_stats in stats.get("models", {}).values():
            usage = model_stats.get("tokens", {})
            break

        return LLMResponse(
            text=text,
            model=self.model,
            usage=usage,
            cost=0.0,
        )
