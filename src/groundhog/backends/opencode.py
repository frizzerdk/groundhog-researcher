"""OpenCode CLI backend. Supports 75+ providers via provider/model format."""

import json
import os
import shutil
import subprocess
from pathlib import Path

from groundhog.base.backend import LLMBackend, LLMResponse, Prompt, TextPart

DEFAULT_MODEL = "openrouter/deepseek/deepseek-v4-flash"


class OpenCodeBackend(LLMBackend):
    """OpenCode CLI (opencode.ai). Supports any provider the user has configured.

    Models specified as provider/model, e.g. "anthropic/claude-sonnet-4-6-20260217".
    """

    def __init__(self, model: str = DEFAULT_MODEL, timeout: int = 300):
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: Prompt, system_prompt: str = "") -> LLMResponse:
        prompt_text = prompt if isinstance(prompt, str) else " ".join(
            p.text for p in prompt if isinstance(p, TextPart))

        if system_prompt:
            prompt_text = f"{system_prompt}\n\n{prompt_text}"

        cmd = [_resolve_opencode_bin(), "run", "--format", "json", "--model", self.model]

        try:
            result = subprocess.run(cmd, input=prompt_text, capture_output=True, text=True,
                                    timeout=self.timeout, encoding="utf-8",
                                    errors="replace", env=_build_env(self.model))
        except FileNotFoundError:
            raise RuntimeError("OpenCode CLI not found. Install from https://opencode.ai")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"OpenCode CLI timed out after {self.timeout}s")

        if result.returncode != 0:
            raise RuntimeError(f"OpenCode CLI error: {result.stderr.strip()}")

        events = _parse_json_events(result.stdout)

        return LLMResponse(
            text=_extract_text(events),
            model=self.model,
            cost=_extract_cost(events),
        )


def _resolve_opencode_bin() -> str:
    found = shutil.which("opencode")
    if found:
        return found
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidate = Path(appdata) / "npm" / "opencode.cmd"
            try:
                if candidate.exists():
                    return str(candidate)
            except OSError:
                return str(candidate)
    return "opencode"


def _build_env(model: str) -> dict:
    env = os.environ.copy()
    if model.startswith("openrouter/"):
        model_id = model.split("/", 1)[1]
        provider = {"models": {model_id: {}}}
        if os.environ.get("OPENROUTER_API_KEY"):
            provider["options"] = {"apiKey": "{env:OPENROUTER_API_KEY}"}
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps({
            "$schema": "https://opencode.ai/config.json",
            "model": model,
            "provider": {"openrouter": provider},
        })
    return env


def _parse_json_events(output: str) -> list:
    events = []
    text = output.strip()
    if not text:
        return events

    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        pass

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _extract_text(events: list) -> str:
    parts = []
    for event in events:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype in ("text", "message"):
            text = event.get("text", event.get("content", ""))
            if text:
                parts.append(str(text))
        elif etype == "assistant.message":
            text = (event.get("data", {}) or {}).get("content", "")
            if text:
                parts.append(str(text))
        elif "result" in event and isinstance(event["result"], str):
            parts.append(event["result"])
        else:
            part = event.get("part", {}) or {}
            if part.get("type") in ("text", "message", "assistant-message"):
                text = part.get("text", part.get("content", ""))
                if text:
                    parts.append(str(text))
    return "\n".join(parts).strip()


def _extract_cost(events: list) -> float:
    total = 0.0
    for event in events:
        if not isinstance(event, dict):
            continue
        usage = event.get("usage")
        if not isinstance(usage, dict):
            usage = (event.get("data", {}) or {}).get("usage")
        if isinstance(usage, dict):
            total += float(usage.get("cost", usage.get("cost_usd", 0.0)) or 0.0)
            continue
        part = event.get("part", {}) or {}
        if part.get("cost") is not None:
            total += float(part.get("cost") or 0.0)
    return total
