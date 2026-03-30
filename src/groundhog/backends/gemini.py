"""Gemini LLM backend. Minimal REST implementation via urllib."""

import base64
import json
import os
import urllib.request
import urllib.error

from groundhog.base.backend import (
    LLMBackend, LLMResponse, Prompt,
    PromptPart, TextPart, ImagePart, AudioPart,
)


    # Pricing per million tokens (input, output — thinking included in output)
    # Source: https://ai.google.dev/gemini-api/docs/pricing (2026-03-26)
PRICING = {
    "gemini-3.1-pro-preview":        {"input": 2.00, "output": 12.00},
    "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.50},
    "gemini-3-flash-preview":        {"input": 0.50, "output": 3.00},
    "gemini-2.5-pro":                {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash":              {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite":         {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash":              {"input": 0.10, "output": 0.40},
}


class GeminiBackend(LLMBackend):
    """Google Gemini via REST API. No SDK dependency."""

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str = None,
                 thinking_level: str = None):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.thinking_level = thinking_level

    def _to_gemini_parts(self, prompt: Prompt) -> list:
        if isinstance(prompt, str):
            return [{"text": prompt}]

        parts = []
        for part in prompt:
            if isinstance(part, TextPart):
                parts.append({"text": part.text})
            elif isinstance(part, (ImagePart, AudioPart)):
                parts.append({
                    "inline_data": {
                        "mime_type": part.mime_type,
                        "data": base64.b64encode(part.data).decode(),
                    }
                })
        return parts

    def generate(self, prompt: Prompt, system_prompt: str = "") -> LLMResponse:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}"
            f":generateContent?key={self.api_key}"
        )

        parts = self._to_gemini_parts(prompt)
        body = {
            "contents": [{"role": "user", "parts": parts}],
        }

        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        if self.thinking_level:
            body["generationConfig"] = {
                "thinkingConfig": {"thinkingLevel": self.thinking_level}
            }

        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            raise RuntimeError(f"Gemini API error {e.code}: {error_body}") from e

        candidate = result.get("candidates", [{}])[0]
        try:
            text = candidate["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            # Gemini sometimes puts content in finishMessage (e.g. MALFORMED_FUNCTION_CALL)
            reason = candidate.get("finishReason", "unknown")
            text = candidate.get("finishMessage", "")
            if text:
                print(f"  [Gemini] Recovered from {reason}, using finishMessage")
            else:
                raise RuntimeError(f"Gemini returned no content (finishReason: {reason})")

        usage = result.get("usageMetadata", {})
        cost = self._compute_cost(usage)

        return LLMResponse(text=text, model=self.model, usage=usage, cost=cost)

    def _compute_cost(self, usage):
        rates = PRICING.get(self.model)
        if not rates:
            return 0.0
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)
        thinking_tokens = usage.get("thoughtsTokenCount", 0)
        return (
            input_tokens * rates["input"] / 1_000_000
            + (output_tokens + thinking_tokens) * rates["output"] / 1_000_000
        )
