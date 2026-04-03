"""Gemini CLI backend. Uses gemini -p for non-interactive mode."""

import json
import subprocess

from groundhog.base.backend import LLMBackend, LLMResponse, Prompt, TextPart


class GeminiCLIBackend(LLMBackend):
    """Google Gemini via the Gemini CLI. Uses Google account auth."""

    def __init__(self, model: str = "gemini-2.5-flash", timeout: int = 300):
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: Prompt, system_prompt: str = "") -> LLMResponse:
        prompt_text = prompt if isinstance(prompt, str) else " ".join(
            p.text for p in prompt if isinstance(p, TextPart))

        if system_prompt:
            prompt_text = f"{system_prompt}\n\n{prompt_text}"

        cmd = ["gemini", "-p", "--output-format", "json"]

        try:
            result = subprocess.run(cmd, input=prompt_text, capture_output=True, text=True,
                                    timeout=self.timeout, encoding="utf-8",
                                    errors="replace")
        except FileNotFoundError:
            raise RuntimeError("Gemini CLI not found. Install from https://github.com/google-gemini/gemini-cli")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Gemini CLI timed out after {self.timeout}s")

        if result.returncode != 0:
            raise RuntimeError(f"Gemini CLI error: {result.stderr.strip()}")

        data = json.loads(result.stdout)

        return LLMResponse(
            text=data.get("result", ""),
            model=self.model,
            usage=data.get("usage", {}),
            cost=0.0,
        )
