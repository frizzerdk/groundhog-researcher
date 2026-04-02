"""
Custom LLM backend template.

A BACKEND wraps an LLM provider so strategies can call it. The only required
method is generate(prompt, system_prompt) -> LLMResponse.

Two patterns shown below — pick one and delete the other:
  1. API backend  — REST calls via urllib (no SDK dependency)
  2. CLI backend  — wraps a command-line tool via subprocess

THE INTERFACE:
  class MyBackend(LLMBackend):
      def generate(self, prompt, system_prompt="") -> LLMResponse:
          ...
          return LLMResponse(text=..., model=..., usage={...}, cost=0.0)

PROMPT TYPES:
  prompt is Union[str, Sequence[PromptPart]]:
  - String: the common case, just text
  - Sequence: multimodal — TextPart, ImagePart, AudioPart
  Most backends only need to handle strings. Extract text with:
    text = prompt if isinstance(prompt, str) else " ".join(
        p.text for p in prompt if isinstance(p, TextPart))

LLMResponse FIELDS:
  text: str     — the LLM's response text
  model: str    — model identifier (for logging)
  usage: dict   — token counts (provider-specific format, stored as-is)
  cost: float   — estimated cost in USD (0.0 if unknown)

REGISTERING:
  # Direct
  from groundhog import BackendRegistry
  registry = BackendRegistry(default=MyBackend(), high=MyBackend(model="big"))

  # With auto_registry, add your backend to discover_backends() in
  # groundhog/backends/discover.py

BACKEND TIERS:
  Strategies request by tier: toolkit.llm.get("default"), .get("high"), etc.
  Missing tiers fall back to "default". Common tiers:
    max     — best reasoning, price doesn't matter
    high    — strong reasoning, good value
    default — workhorse for most tasks
    budget  — cheap but capable
    cheap   — cheapest possible, bulk generation
"""

import json
import os
import subprocess
import urllib.request
import urllib.error

from groundhog.base.backend import LLMBackend, LLMResponse, Prompt, TextPart


# ==========================================================================
# OPTION 1: API Backend (REST via urllib)
# ==========================================================================
# Use this for providers with HTTP APIs. No SDK dependency needed.
# See also: OpenAICompatibleBackend for OpenAI-protocol providers,
#           AnthropicBackend, GeminiBackend for existing implementations.

# Pricing per million tokens — for cost tracking
PRICING = {
    "my-model-large": {"input": 2.00, "output": 10.00},
    "my-model-small": {"input": 0.20, "output": 1.00},
}


class MyAPIBackend(LLMBackend):
    """Custom API backend. Replace with your provider's details."""

    def __init__(self, model: str = "my-model-large", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.environ.get("MY_API_KEY", "")

    def generate(self, prompt: Prompt, system_prompt: str = "") -> LLMResponse:
        # Extract text from prompt (handles both str and multimodal)
        prompt_text = prompt if isinstance(prompt, str) else " ".join(
            p.text for p in prompt if isinstance(p, TextPart))

        # Build request body (adapt to your provider's format)
        url = "https://api.myprovider.com/v1/chat/completions"
        body = {
            "model": self.model,
            "messages": [],
        }
        if system_prompt:
            body["messages"].append({"role": "system", "content": system_prompt})
        body["messages"].append({"role": "user", "content": prompt_text})

        data = json.dumps(body).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(url, data=data, headers=headers)

        # Make request
        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            raise RuntimeError(f"API error {e.code}: {error_body}") from e

        # Parse response (adapt to your provider's format)
        text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        cost = self._compute_cost(usage)

        return LLMResponse(text=text, model=self.model, usage=usage, cost=cost)

    def _compute_cost(self, usage):
        rates = PRICING.get(self.model)
        if not rates:
            return 0.0
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        return (
            input_tokens * rates["input"] / 1_000_000
            + output_tokens * rates["output"] / 1_000_000
        )


# ==========================================================================
# OPTION 2: CLI Subprocess Backend
# ==========================================================================
# Use this for tools that have a command-line interface.
# See also: ClaudeCodeBackend, CopilotBackend, GeminiCLIBackend, OpenCodeBackend

class MyCLIBackend(LLMBackend):
    """Custom CLI backend. Replace with your tool's command syntax."""

    def __init__(self, model: str = "default", timeout: int = 300):
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: Prompt, system_prompt: str = "") -> LLMResponse:
        prompt_text = prompt if isinstance(prompt, str) else " ".join(
            p.text for p in prompt if isinstance(p, TextPart))

        # Prepend system prompt if the CLI doesn't have a separate flag
        if system_prompt:
            prompt_text = f"{system_prompt}\n\n{prompt_text}"

        # Build command (adapt to your tool)
        cmd = ["my-tool", "--model", self.model, "--json", prompt_text]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=self.timeout)
        except FileNotFoundError:
            raise RuntimeError("my-tool not found. Install from https://...")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"my-tool timed out after {self.timeout}s")

        if result.returncode != 0:
            raise RuntimeError(f"my-tool error: {result.stderr.strip()}")

        # Parse output (adapt to your tool's format)
        # If JSON output:
        data = json.loads(result.stdout)
        text = data.get("result", "")
        usage = data.get("usage", {})
        cost = data.get("cost", 0.0)

        # If plain text output:
        # text = result.stdout.strip()
        # usage = {}
        # cost = 0.0

        return LLMResponse(text=text, model=self.model, usage=usage, cost=cost)
