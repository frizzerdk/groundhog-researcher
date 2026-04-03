"""OpenCode CLI backend. Supports 75+ providers via provider/model format."""

import json
import subprocess

from groundhog.base.backend import LLMBackend, LLMResponse, Prompt, TextPart


class OpenCodeBackend(LLMBackend):
    """OpenCode CLI (opencode.ai). Supports any provider the user has configured.

    Models specified as provider/model, e.g. "anthropic/claude-sonnet-4-6-20260217".
    """

    def __init__(self, model: str = "anthropic/claude-sonnet-4-6-20260217", timeout: int = 300):
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: Prompt, system_prompt: str = "") -> LLMResponse:
        prompt_text = prompt if isinstance(prompt, str) else " ".join(
            p.text for p in prompt if isinstance(p, TextPart))

        if system_prompt:
            prompt_text = f"{system_prompt}\n\n{prompt_text}"

        cmd = ["opencode", "run", "--format", "json", "--model", self.model]

        try:
            result = subprocess.run(cmd, input=prompt_text, capture_output=True, text=True,
                                    timeout=self.timeout, encoding="utf-8",
                                    errors="replace")
        except FileNotFoundError:
            raise RuntimeError("OpenCode CLI not found. Install from https://opencode.ai")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"OpenCode CLI timed out after {self.timeout}s")

        if result.returncode != 0:
            raise RuntimeError(f"OpenCode CLI error: {result.stderr.strip()}")

        data = json.loads(result.stdout)

        return LLMResponse(
            text=data.get("result", ""),
            model=self.model,
            cost=0.0,
        )
