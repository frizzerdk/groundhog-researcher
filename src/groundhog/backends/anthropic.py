"""Anthropic LLM backend. Direct Messages API via urllib."""

import json
import os
import urllib.request
import urllib.error

from groundhog.base.backend import LLMBackend, LLMResponse, Prompt, TextPart
from groundhog.backends._http import _urlopen_with_warnings

PRICING = {
    "claude-opus-4-6-20260205":   {"input": 5.00,  "output": 25.00},
    "claude-sonnet-4-6-20260217": {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5-20250414":  {"input": 1.00,  "output": 5.00},
}


class AnthropicBackend(LLMBackend):
    """Anthropic Claude via Messages API. No SDK dependency."""

    def __init__(self, model: str = "claude-sonnet-4-6-20260217", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def generate(self, prompt: Prompt, system_prompt: str = "") -> LLMResponse:
        url = "https://api.anthropic.com/v1/messages"

        if isinstance(prompt, str):
            content = prompt
        else:
            content = []
            for part in prompt:
                if isinstance(part, TextPart):
                    content.append({"type": "text", "text": part.text})

        body = {
            "model": self.model,
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": content}],
        }
        if system_prompt:
            body["system"] = system_prompt

        data = json.dumps(body).encode()
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        req = urllib.request.Request(url, data=data, headers=headers)

        try:
            with _urlopen_with_warnings(req, label=f"Anthropic ({self.model})") as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            raise RuntimeError(f"Anthropic API error {e.code}: {error_body}") from e

        text = result["content"][0]["text"]
        usage = result.get("usage", {})
        cost = self._compute_cost(usage)

        return LLMResponse(text=text, model=self.model, usage=usage, cost=cost)

    def _compute_cost(self, usage):
        rates = PRICING.get(self.model)
        if not rates:
            return 0.0
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        return (
            input_tokens * rates["input"] / 1_000_000
            + output_tokens * rates["output"] / 1_000_000
        )
