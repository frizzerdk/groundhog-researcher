"""Claude Code CLI backend. Zero-config for anyone with Claude Code installed."""

import json
import subprocess

from groundhog.base.backend import LLMBackend, LLMResponse, Prompt, TextPart


class ClaudeCodeBackend(LLMBackend):
    """Claude via the Claude Code CLI. No API key needed — uses CLI auth.

    Models: "opus", "sonnet", "haiku" (or full model names).
    """

    def __init__(self, model: str = "sonnet", timeout: int = 600):
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: Prompt, system_prompt: str = "") -> LLMResponse:
        prompt_text = prompt if isinstance(prompt, str) else " ".join(
            p.text for p in prompt if isinstance(p, TextPart))

        # Prompt via stdin (avoids Windows ~32k command-line limit)
        cmd = ["claude", "-p", "--output-format", "json", "--tools", "",
               "--model", self.model]
        if system_prompt:
            cmd += ["--system-prompt", system_prompt]

        try:
            result = subprocess.run(cmd, input=prompt_text, capture_output=True, text=True,
                                    timeout=self.timeout, encoding="utf-8",
                                    errors="replace")
        except FileNotFoundError as e:
            if getattr(e, 'winerror', None) == 206:
                raise RuntimeError(f"Command line too long ({len(prompt_text)} chars)")
            raise RuntimeError("Claude Code CLI not found. Install from https://claude.ai/code")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Claude Code CLI timed out after {self.timeout}s")

        if result.returncode != 0:
            raise RuntimeError(f"Claude Code CLI error: {result.stderr.strip()}")

        data = json.loads(result.stdout)

        return LLMResponse(
            text=data.get("result", ""),
            model=self.model,
            usage=data.get("usage", {}),
            cost=data.get("total_cost_usd", 0.0),
        )
