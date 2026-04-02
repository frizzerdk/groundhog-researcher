"""OpenAI-compatible LLM backend. Covers 15+ providers via one class.

Works with any provider that speaks the OpenAI chat completions protocol.
Use factory methods for common providers, or pass custom base_url + api_key.
"""

import json
import os
import urllib.request
import urllib.error

from groundhog.base.backend import LLMBackend, LLMResponse, Prompt, TextPart
from groundhog.backends._http import _urlopen_with_warnings


class OpenAICompatibleBackend(LLMBackend):
    """Generic backend for any OpenAI-compatible API. No SDK dependency."""

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1",
                 api_key: str = None, api_key_env: str = "OPENAI_API_KEY"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get(api_key_env, "")

    def generate(self, prompt: Prompt, system_prompt: str = "") -> LLMResponse:
        url = f"{self.base_url}/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            # Multimodal — convert parts to content array
            content = []
            for part in prompt:
                if isinstance(part, TextPart):
                    content.append({"type": "text", "text": part.text})
            messages.append({"role": "user", "content": content})

        body = {"model": self.model, "messages": messages}
        data = json.dumps(body).encode()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(url, data=data, headers=headers)

        try:
            with _urlopen_with_warnings(req, label=f"{self.model}") as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            raise RuntimeError(f"API error {e.code} from {self.base_url}: {error_body}") from e

        text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        cost = self._compute_cost(usage)

        return LLMResponse(text=text, model=self.model, usage=usage, cost=cost)

    def _compute_cost(self, usage):
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        # Cost computation left to subclasses or pricing tables
        return 0.0

    # --- Factory methods for common providers ---

    @classmethod
    def openai(cls, model="gpt-5.4-mini", **kwargs):
        return cls(model=model, base_url="https://api.openai.com/v1",
                   api_key_env="OPENAI_API_KEY", **kwargs)

    @classmethod
    def openrouter(cls, model="anthropic/claude-sonnet-4-6-20260217", **kwargs):
        return cls(model=model, base_url="https://openrouter.ai/api/v1",
                   api_key_env="OPENROUTER_API_KEY", **kwargs)

    @classmethod
    def deepseek(cls, model="deepseek-chat", **kwargs):
        return cls(model=model, base_url="https://api.deepseek.com/v1",
                   api_key_env="DEEPSEEK_API_KEY", **kwargs)

    @classmethod
    def groq(cls, model="llama-3.3-70b-versatile", **kwargs):
        return cls(model=model, base_url="https://api.groq.com/openai/v1",
                   api_key_env="GROQ_API_KEY", **kwargs)

    @classmethod
    def cerebras(cls, model="llama-3.3-70b", **kwargs):
        return cls(model=model, base_url="https://api.cerebras.ai/v1",
                   api_key_env="CEREBRAS_API_KEY", **kwargs)

    @classmethod
    def xai(cls, model="grok-4-1-fast-non-reasoning", **kwargs):
        return cls(model=model, base_url="https://api.x.ai/v1",
                   api_key_env="XAI_API_KEY", **kwargs)

    @classmethod
    def together(cls, model="meta-llama/Llama-3.3-70B-Instruct-Turbo", **kwargs):
        return cls(model=model, base_url="https://api.together.xyz/v1",
                   api_key_env="TOGETHER_API_KEY", **kwargs)

    @classmethod
    def fireworks(cls, model="accounts/fireworks/models/llama-v3p3-70b-instruct", **kwargs):
        return cls(model=model, base_url="https://api.fireworks.ai/inference/v1",
                   api_key_env="FIREWORKS_API_KEY", **kwargs)

    @classmethod
    def sambanova(cls, model="Meta-Llama-3.3-70B-Instruct", **kwargs):
        return cls(model=model, base_url="https://api.sambanova.ai/v1",
                   api_key_env="SAMBANOVA_API_KEY", **kwargs)

    @classmethod
    def deepinfra(cls, model="meta-llama/Llama-3.3-70B-Instruct", **kwargs):
        return cls(model=model, base_url="https://api.deepinfra.com/v1/openai",
                   api_key_env="DEEPINFRA_API_KEY", **kwargs)

    @classmethod
    def mistral(cls, model="mistral-large-latest", **kwargs):
        return cls(model=model, base_url="https://api.mistral.ai/v1",
                   api_key_env="MISTRAL_API_KEY", **kwargs)

    @classmethod
    def perplexity(cls, model="sonar-pro", **kwargs):
        return cls(model=model, base_url="https://api.perplexity.ai",
                   api_key_env="PERPLEXITY_API_KEY", **kwargs)

    @classmethod
    def ollama(cls, model="llama3", host="http://localhost:11434", **kwargs):
        return cls(model=model, base_url=f"{host}/v1",
                   api_key="ollama", **kwargs)

    @classmethod
    def lmstudio(cls, model="default", host="http://localhost:1234", **kwargs):
        return cls(model=model, base_url=f"{host}/v1",
                   api_key="lm-studio", **kwargs)

    @classmethod
    def vllm(cls, model="default", host="http://localhost", port=8000, **kwargs):
        return cls(model=model, base_url=f"{host}:{port}/v1",
                   api_key="EMPTY", **kwargs)
