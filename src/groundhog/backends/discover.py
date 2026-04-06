"""Auto-discovery of available LLM backends.

Checks what CLI tools are on PATH, what local servers are running,
and what API keys are set. Builds a BackendRegistry from the best available.
Respects user preferences from ~/.groundhog/config.json.
"""

import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Dict

from groundhog.base.backend import LLMBackend, BackendRegistry


# ---------------------------------------------------------------------------
# Preferences (persistent config)
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path.home() / ".groundhog" / "config.json"


def _load_preferences():
    if not _CONFIG_PATH.exists():
        return {}
    return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))


def _save_preferences(prefs):
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not prefs:
        _CONFIG_PATH.unlink(missing_ok=True)
    else:
        _CONFIG_PATH.write_text(json.dumps(prefs, indent=2) + "\n", encoding="utf-8")


_auth_warnings = []


def discover_backends() -> Dict[str, LLMBackend]:
    """Check what LLM tools are available on this machine.

    Returns a dict of {name: backend_instance} for everything found.
    Auth issues are stored in _auth_warnings (list of (name, message) tuples).
    """
    _auth_warnings.clear()
    backends = {}

    # CLI tools on PATH
    if shutil.which("claude"):
        from groundhog.backends.claude_code import ClaudeCodeBackend
        backends["claude_code"] = ClaudeCodeBackend(model="sonnet")

    if shutil.which("copilot"):
        from groundhog.backends.copilot import CopilotBackend, check_copilot_auth
        ok, msg = check_copilot_auth()
        if ok:
            backends["copilot"] = CopilotBackend()
        else:
            _auth_warnings.append(("copilot", "not authenticated"))

    if shutil.which("gemini"):
        from groundhog.backends.gemini_cli import GeminiCLIBackend
        backends["gemini_cli"] = GeminiCLIBackend()

    if shutil.which("opencode"):
        from groundhog.backends.opencode import OpenCodeBackend
        backends["opencode"] = OpenCodeBackend()

    # Local servers
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1)
        from groundhog.backends.openai_compat import OpenAICompatibleBackend
        backends["ollama"] = OpenAICompatibleBackend.ollama()
    except Exception:
        pass

    # API keys (skip placeholder values)
    def _has_key(env_var):
        val = os.environ.get(env_var, "")
        return val and val not in ("your-key-here", "sk-...", "YOUR_KEY", "")

    if _has_key("ANTHROPIC_API_KEY"):
        from groundhog.backends.anthropic import AnthropicBackend
        backends["anthropic"] = AnthropicBackend()

    if _has_key("OPENAI_API_KEY"):
        from groundhog.backends.openai_compat import OpenAICompatibleBackend
        backends["openai"] = OpenAICompatibleBackend.openai()

    if _has_key("GEMINI_API_KEY"):
        from groundhog.backends.gemini import GeminiBackend
        backends["gemini"] = GeminiBackend()

    if _has_key("OPENROUTER_API_KEY"):
        from groundhog.backends.openai_compat import OpenAICompatibleBackend
        backends["openrouter"] = OpenAICompatibleBackend.openrouter()

    if _has_key("DEEPSEEK_API_KEY"):
        from groundhog.backends.openai_compat import OpenAICompatibleBackend
        backends["deepseek"] = OpenAICompatibleBackend.deepseek()

    if _has_key("GROQ_API_KEY"):
        from groundhog.backends.openai_compat import OpenAICompatibleBackend
        backends["groq"] = OpenAICompatibleBackend.groq()

    return backends


# Alias map: user-friendly names -> discovery names
BACKEND_ALIASES = {
    "gemini": "gemini_cli",   # "gemini" prefers CLI over API
    "claude": "claude_code",
}

# Tier definitions: name, priority list, variant function
# Tiers from strongest to cheapest: max > high > default > budget > cheap
TIER_DEFS = {
    "max":     ["anthropic", "openai", "gemini", "gemini_cli", "claude_code", "copilot", "openrouter"],
    "high":    ["gemini", "gemini_cli", "anthropic", "openai", "claude_code", "copilot", "openrouter"],
    "default": ["claude_code", "anthropic", "gemini", "gemini_cli", "openai", "deepseek", "copilot", "openrouter", "opencode", "groq"],
    "budget":  ["deepseek", "groq", "gemini", "gemini_cli", "openai", "claude_code", "copilot", "ollama"],
    "cheap":   ["ollama", "deepseek", "groq", "gemini", "gemini_cli", "openai", "claude_code", "copilot"],
}


def _create_backend(backend_name, model=None):
    """Create a backend instance by name with optional model override."""
    from groundhog.backends.claude_code import ClaudeCodeBackend
    from groundhog.backends.copilot import CopilotBackend
    from groundhog.backends.gemini_cli import GeminiCLIBackend
    from groundhog.backends.opencode import OpenCodeBackend
    from groundhog.backends.anthropic import AnthropicBackend
    from groundhog.backends.gemini import GeminiBackend
    from groundhog.backends.openai_compat import OpenAICompatibleBackend

    factories = {
        "claude_code": lambda m: ClaudeCodeBackend(model=m or "sonnet"),
        "copilot":     lambda m: CopilotBackend(model=m or "gpt-5-mini"),
        "gemini_cli":  lambda m: GeminiCLIBackend(model=m or "gemini-2.5-flash"),
        "opencode":    lambda m: OpenCodeBackend(model=m or "anthropic/claude-sonnet-4-6-20260217"),
        "anthropic":   lambda m: AnthropicBackend(model=m or "claude-sonnet-4-6-20260217"),
        "gemini":      lambda m: GeminiBackend(model=m or "gemini-2.5-flash"),
        "openai":      lambda m: OpenAICompatibleBackend.openai(model=m or "gpt-5.4-mini"),
        "openrouter":  lambda m: OpenAICompatibleBackend.openrouter(model=m or "anthropic/claude-sonnet-4-6-20260217"),
        "deepseek":    lambda m: OpenAICompatibleBackend.deepseek(model=m or "deepseek-chat"),
        "groq":        lambda m: OpenAICompatibleBackend.groq(model=m or "llama-3.3-70b-versatile"),
        "ollama":      lambda m: OpenAICompatibleBackend.ollama(model=m or "llama3"),
    }
    factory = factories.get(backend_name)
    if not factory:
        return None
    return factory(model)


def auto_registry() -> BackendRegistry:
    """Build a BackendRegistry from the best available backends.

    Respects user preferences from ~/.groundhog/config.json:
      - "prefer": backend name -> use this backend for all tiers
      - "tiers": {tier: {backend, model}} -> override specific tiers
    """
    available = discover_backends()
    if not available:
        raise RuntimeError(
            "No LLM backends found. Install Claude Code, set an API key "
            "(ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY), or start Ollama."
        )

    prefs = _load_preferences()

    # If a global preference is set and available, put it first in all priority lists
    preferred = prefs.get("prefer")
    if preferred and preferred in available:
        tier_defs = {}
        for tier_name, priority in TIER_DEFS.items():
            tier_defs[tier_name] = [preferred] + [p for p in priority if p != preferred]
    else:
        tier_defs = TIER_DEFS

    # Build tiers from discovery + priority
    tiers = {}
    for tier_name, priority in tier_defs.items():
        variant_fn = _TIER_VARIANTS.get(tier_name)
        for name in priority:
            if name in available:
                if variant_fn:
                    backend = variant_fn(name, available[name])
                else:
                    backend = available[name]
                if backend:
                    tiers[tier_name] = backend
                    break

    # Default tier is required
    if "default" not in tiers:
        tiers["default"] = next(iter(available.values()))

    # Apply specific tier overrides from preferences
    for tier_name, override in prefs.get("tiers", {}).items():
        backend_name = override["backend"]
        model = override.get("model")
        if backend_name in available:
            backend = _create_backend(backend_name, model)
            if backend:
                tiers[tier_name] = backend

    return BackendRegistry(**tiers)


def _get_max_variant(name, backend):
    """Best reasoning model regardless of cost."""
    from groundhog.backends.anthropic import AnthropicBackend
    from groundhog.backends.openai_compat import OpenAICompatibleBackend
    from groundhog.backends.claude_code import ClaudeCodeBackend
    from groundhog.backends.gemini import GeminiBackend
    from groundhog.backends.gemini_cli import GeminiCLIBackend
    from groundhog.backends.copilot import CopilotBackend

    variants = {
        "anthropic": lambda: AnthropicBackend(model="claude-opus-4-6-20260205"),
        "openai":    lambda: OpenAICompatibleBackend.openai(model="gpt-5.4"),
        "gemini":    lambda: GeminiBackend(model="gemini-3.1-pro-preview"),
        "gemini_cli": lambda: GeminiCLIBackend(model="gemini-3.1-pro-preview"),
        "claude_code": lambda: ClaudeCodeBackend(model="opus"),
        "copilot":   lambda: CopilotBackend(model="claude-sonnet-4.6"),
        "openrouter": lambda: OpenAICompatibleBackend.openrouter(model="anthropic/claude-opus-4-6-20260205"),
    }
    return variants.get(name, lambda: None)()


def _get_high_variant(name, backend):
    """Strong reasoning, good value."""
    from groundhog.backends.anthropic import AnthropicBackend
    from groundhog.backends.openai_compat import OpenAICompatibleBackend
    from groundhog.backends.claude_code import ClaudeCodeBackend
    from groundhog.backends.gemini import GeminiBackend
    from groundhog.backends.gemini_cli import GeminiCLIBackend
    from groundhog.backends.copilot import CopilotBackend

    variants = {
        "gemini":    lambda: GeminiBackend(model="gemini-3-flash-preview"),
        "gemini_cli": lambda: GeminiCLIBackend(model="gemini-3-flash-preview"),
        "anthropic": lambda: AnthropicBackend(model="claude-sonnet-4-6-20260217"),
        "openai":    lambda: OpenAICompatibleBackend.openai(model="gpt-5.4-mini"),
        "claude_code": lambda: ClaudeCodeBackend(model="sonnet"),
        "copilot":   lambda: CopilotBackend(model="gpt-5-mini"),
        "openrouter": lambda: OpenAICompatibleBackend.openrouter(model="google/gemini-3-flash-preview"),
    }
    return variants.get(name, lambda: None)()


def _get_budget_variant(name, backend):
    """Cheap but capable."""
    from groundhog.backends.openai_compat import OpenAICompatibleBackend
    from groundhog.backends.gemini import GeminiBackend
    from groundhog.backends.gemini_cli import GeminiCLIBackend
    from groundhog.backends.claude_code import ClaudeCodeBackend
    from groundhog.backends.copilot import CopilotBackend

    variants = {
        "deepseek":   lambda: backend,
        "groq":       lambda: backend,
        "gemini":     lambda: GeminiBackend(model="gemini-2.5-flash"),
        "gemini_cli": lambda: GeminiCLIBackend(model="gemini-3.1-flash-lite-preview"),
        "openai":     lambda: OpenAICompatibleBackend.openai(model="gpt-5.4-nano"),
        "claude_code": lambda: ClaudeCodeBackend(model="haiku"),
        "copilot":   lambda: CopilotBackend(model="gpt-5-mini"),
        "ollama":     lambda: backend,
    }
    return variants.get(name, lambda: None)()


def _get_cheap_variant(name, backend):
    """Cheapest possible."""
    from groundhog.backends.openai_compat import OpenAICompatibleBackend
    from groundhog.backends.gemini import GeminiBackend
    from groundhog.backends.gemini_cli import GeminiCLIBackend
    from groundhog.backends.claude_code import ClaudeCodeBackend
    from groundhog.backends.copilot import CopilotBackend

    variants = {
        "ollama":     lambda: backend,
        "deepseek":   lambda: backend,
        "groq":       lambda: backend,
        "gemini":     lambda: GeminiBackend(model="gemini-2.5-flash-lite"),
        "gemini_cli": lambda: GeminiCLIBackend(model="gemini-2.5-flash-lite"),
        "openai":     lambda: OpenAICompatibleBackend.openai(model="gpt-5.4-nano"),
        "claude_code": lambda: ClaudeCodeBackend(model="haiku"),
        "copilot":   lambda: CopilotBackend(model="gpt-5-mini"),
    }
    return variants.get(name, lambda: None)()


_TIER_VARIANTS = {
    "max": _get_max_variant,
    "high": _get_high_variant,
    "default": None,  # use discovered backend as-is
    "budget": _get_budget_variant,
    "cheap": _get_cheap_variant,
}


# ---------------------------------------------------------------------------
# Agent backend discovery
# ---------------------------------------------------------------------------

def discover_agent_backends():
    """Check what agent-capable CLI tools are available.

    Returns a dict of {name: AgentBackend} for everything found.
    Only CLI tools qualify — API backends are stateless LLMs, not agents.
    """
    from groundhog.base.agent import AgentBackend
    backends = {}

    if shutil.which("claude"):
        try:
            from groundhog.agents.claude_code import ClaudeCodeAgentBackend
            backends["claude_code"] = ClaudeCodeAgentBackend()
        except Exception:
            pass

    if shutil.which("gemini"):
        try:
            from groundhog.agents.gemini_cli import GeminiCliAgentBackend
            backends["gemini_cli"] = GeminiCliAgentBackend()
        except Exception:
            pass

    if shutil.which("copilot"):
        try:
            from groundhog.agents.copilot import CopilotAgentBackend
            backends["copilot"] = CopilotAgentBackend()
        except Exception:
            pass

    return backends


_AGENT_TIER_DEFS = {
    "default": ["claude_code", "copilot", "gemini_cli"],
    "high":    ["claude_code", "copilot", "gemini_cli"],
    "budget":  ["gemini_cli", "copilot", "claude_code"],
}


def auto_agent_registry():
    """Build an AgentRegistry from available agent backends.

    Returns None if no agent backends are found (agents are optional).
    Respects ~/.groundhog/config.json preferences.
    """
    from groundhog.base.agent import AgentRegistry

    available = discover_agent_backends()
    if not available:
        return None

    prefs = _load_preferences()
    tiers = {}

    for tier, priority in _AGENT_TIER_DEFS.items():
        # Check preferences first
        tier_pref = prefs.get("agent_tiers", {}).get(tier)
        if tier_pref and tier_pref in available:
            tiers[tier] = available[tier_pref]
            continue

        # Fall through priority list
        for name in priority:
            if name in available:
                tiers[tier] = available[name]
                break

    if not tiers:
        return None

    return AgentRegistry(**tiers)
