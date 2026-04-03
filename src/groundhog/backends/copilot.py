"""GitHub Copilot CLI backend. Uses copilot -p for non-interactive mode."""

import os
import re
import subprocess
import sys
import threading
import time

from groundhog.base.backend import LLMBackend, LLMResponse, Prompt, TextPart


def check_copilot_auth():
    """Check if copilot CLI is authenticated. Returns (ok, message)."""
    # Environment tokens (fine-grained PAT required, classic ghp_ won't work)
    for var in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        val = os.environ.get(var, "")
        if val:
            if val.startswith("ghp_"):
                return False, (
                    f"{var} is a classic PAT (ghp_...) which Copilot doesn't support.\n"
                    "Create a fine-grained PAT with 'Copilot Requests' permission:\n"
                    "  https://github.com/settings/personal-access-tokens/new\n"
                    "Then: export COPILOT_GITHUB_TOKEN=github_pat_..."
                )
            return True, f"authenticated via {var}"

    # Check for stored auth (from copilot login)
    from pathlib import Path
    config = Path.home() / ".copilot" / "config.json"
    if config.exists():
        return True, "authenticated via ~/.copilot"

    # Check gh CLI fallback
    try:
        result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True, timeout=5, encoding="utf-8", errors="replace")
        if result.returncode == 0:
            return True, "authenticated via gh CLI"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return False, "not_authenticated"


def login_copilot():
    """Run copilot login interactively. Returns True if successful."""
    try:
        result = subprocess.run(["copilot", "login"])
        return result.returncode == 0
    except FileNotFoundError:
        return False


class CopilotBackend(LLMBackend):
    """GitHub Copilot via the Copilot CLI. Uses GitHub subscription auth.

    Access to Claude, GPT, and Gemini models under one subscription.
    -p for non-interactive, stats parsed for cost tracking.
    """

    # $10/month for 300 premium requests (Pro plan)
    # Model multipliers: 0x=free, 0.33x=budget, 1x=standard, 3x=premium
    # Free: GPT-4.1, GPT-4o, GPT-5-mini
    # 0.33x: Claude Haiku 4.5, Gemini 3 Flash, GPT-5.4-mini
    # 1x: Claude Sonnet 4.6, Gemini Pro, GPT-5.x
    # 3x: Claude Opus 4.5/4.6
    COST_PER_PREMIUM_REQUEST = 10.0 / 300

    def __init__(self, model: str = "gpt-5-mini", warn_interval: int = 30, max_retries: int = 2):
        self.model = model
        self.warn_interval = warn_interval
        self.max_retries = max_retries

    def _run_cli(self, cmd, input_text):
        """Run CLI with stdin, periodic warnings, no timeout."""
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        except FileNotFoundError:
            raise RuntimeError("Copilot CLI not found. Install from https://github.com/github/copilot-cli")

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

        proc.stdin.write(input_text.encode("utf-8"))
        proc.stdin.close()

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
        return stdout, stderr, proc.returncode

    def generate(self, prompt: Prompt, system_prompt: str = "") -> LLMResponse:
        errors = []
        for attempt in range(self.max_retries + 1):
            try:
                result = self._generate(prompt, system_prompt)
                if errors:
                    result.usage["retries"] = len(errors)
                    result.usage["retry_errors"] = [str(e) for e in errors]
                return result
            except RuntimeError as e:
                # Don't retry auth errors — those need user interaction
                if "Not authenticated" in str(e) or "login" in str(e).lower():
                    raise
                errors.append(e)
                if attempt < self.max_retries:
                    wait = 5 * (attempt + 1)
                    print(f"(error: {e}, retrying in {wait}s)... ", end="", file=sys.stderr, flush=True)
                    time.sleep(wait)
        raise errors[-1]

    def _generate(self, prompt: Prompt, system_prompt: str = "") -> LLMResponse:
        prompt_text = prompt if isinstance(prompt, str) else " ".join(
            p.text for p in prompt if isinstance(p, TextPart))

        if system_prompt:
            prompt_text = f"{system_prompt}\n\n{prompt_text}"

        # Prompt via stdin (avoids Windows ~32k command-line limit)
        # --available-tools with no args = zero tools (pure LLM, no file/search access)
        cmd = ["copilot", "--model", self.model,
               "--no-custom-instructions", "--available-tools"]

        stdout, stderr, returncode = self._run_cli(cmd, prompt_text)

        if returncode != 0 and not stdout.strip():
            if "No authentication" in stderr:
                print("\n[copilot] Not authenticated in this environment.")
                answer = input("  Log in now? (y/n): ").strip().lower()
                if answer in ("y", "yes", ""):
                    if login_copilot():
                        stdout, stderr, returncode = self._run_cli(cmd, prompt_text)
                    else:
                        raise RuntimeError("Copilot login failed.")
                else:
                    raise RuntimeError("Copilot not authenticated. Run 'groundhog backends' to log in.")
            if returncode != 0 and not stdout.strip():
                raise RuntimeError(f"Copilot CLI error: {stderr.strip()}")

        # Strip copilot CLI noise (● lines are CLI artifacts, not LLM output)
        lines = stdout.split('\n')
        last_noise = -1
        for i, line in enumerate(lines):
            if line.lstrip().startswith('●'):
                last_noise = i
        if last_noise >= 0:
            discarded = '\n'.join(lines[:last_noise + 1]).strip()
            text = '\n'.join(lines[last_noise + 1:]).strip()
        else:
            discarded = ""
            text = stdout.strip()
        _, usage, cost = self._parse_output(stderr)
        if discarded:
            usage["cli_output"] = discarded

        return LLMResponse(text=text, model=self.model, usage=usage, cost=cost)

    def _parse_output(self, output):
        """Split response text from usage stats, extract premium request cost."""
        # Stats start after "Total usage est:"
        marker = "Total usage est:"
        if marker in output:
            parts = output.split(marker, 1)
            text = parts[0].rstrip()
            stats = marker + parts[1]
        else:
            return output.strip(), {}, 0.0

        # Parse premium requests
        import re
        usage = {"raw_stats": stats.strip()}
        match = re.search(r"([\d.]+)\s+Premium requests", stats)
        cost = 0.0
        if match:
            premium = float(match.group(1))
            usage["premium_requests"] = premium
            cost = premium * self.COST_PER_PREMIUM_REQUEST

        return text, usage, cost
