"""Cost estimation from conversation logs.

Reads conversation.json files, extracts token counts, estimates cost
using per-model pricing tables.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from groundhog.backends.gemini import PRICING as GEMINI_PRICING

# Default pricing — imports from backends that define their own rates
DEFAULT_PRICING = {**GEMINI_PRICING}


def estimate_cost(path: Path, pricing: Optional[Dict] = None) -> Dict[str, Any]:
    """Estimate cost from a single conversation.json file.

    Args:
        path: path to conversation.json or directory containing it
        pricing: optional pricing override dict

    Returns:
        {"total_cost": float, "input_tokens": int, "output_tokens": int,
         "thinking_tokens": int, "calls": [...], "unknown_models": [...]}
    """
    pricing = pricing or DEFAULT_PRICING
    json_path = path if path.name == "conversation.json" else path / "conversation.json"

    if not json_path.exists():
        return {"total_cost": 0, "input_tokens": 0, "output_tokens": 0,
                "thinking_tokens": 0, "calls": [], "unknown_models": []}

    entries = json.loads(json_path.read_text())
    total_cost = 0.0
    total_input = 0
    total_output = 0
    total_thinking = 0
    calls = []
    unknown_models = set()

    for entry in entries:
        usage = entry.get("usage")
        if not usage:
            continue

        model = entry.get("role", "")
        tokens = _extract_tokens(model, usage)
        cost = _price_tokens(model, tokens, pricing)

        if cost is None:
            unknown_models.add(model)
            cost = 0.0

        total_input += tokens.get("input", 0)
        total_output += tokens.get("output", 0)
        total_thinking += tokens.get("thinking", 0)
        total_cost += cost

        calls.append({"model": model, **tokens, "cost": round(cost, 6)})

    return {
        "total_cost": round(total_cost, 6),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "thinking_tokens": total_thinking,
        "calls": calls,
        "unknown_models": sorted(unknown_models),
    }


def estimate_total_cost(base_path: Path, pricing: Optional[Dict] = None) -> Dict[str, Any]:
    """Aggregate cost across all conversation.json files under base_path."""
    pricing = pricing or DEFAULT_PRICING
    total_cost = 0.0
    total_input = 0
    total_output = 0
    total_thinking = 0
    all_calls = []
    unknown_models = set()

    for json_file in Path(base_path).rglob("conversation.json"):
        result = estimate_cost(json_file, pricing)
        total_cost += result["total_cost"]
        total_input += result["input_tokens"]
        total_output += result["output_tokens"]
        total_thinking += result["thinking_tokens"]
        all_calls.extend(result["calls"])
        unknown_models.update(result["unknown_models"])

    return {
        "total_cost": round(total_cost, 6),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "thinking_tokens": total_thinking,
        "calls": all_calls,
        "unknown_models": sorted(unknown_models),
    }


# --- Token extraction (provider-specific) ---

def _extract_tokens(model: str, usage: Dict) -> Dict[str, int]:
    """Extract normalized token counts from provider-specific usage dict."""
    if model.startswith("gemini"):
        return _extract_gemini(usage)
    # OpenAI / OpenRouter format
    if "prompt_tokens" in usage:
        return {
            "input": usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0),
            "thinking": 0,
        }
    return {"input": 0, "output": 0, "thinking": 0}


def _extract_gemini(usage: Dict) -> Dict[str, int]:
    return {
        "input": usage.get("promptTokenCount", 0),
        "output": usage.get("candidatesTokenCount", 0),
        "thinking": usage.get("thoughtsTokenCount", 0),
    }


def _price_tokens(model: str, tokens: Dict[str, int], pricing: Dict) -> Optional[float]:
    """Calculate cost in dollars. Returns None if model not in pricing."""
    rates = pricing.get(model)
    if not rates:
        return None
    cost = 0.0
    for token_type in ("input", "output", "thinking"):
        count = tokens.get(token_type, 0)
        rate = rates.get(token_type, 0)
        cost += count * rate / 1_000_000
    return cost
