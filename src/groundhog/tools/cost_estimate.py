"""Cost estimation from attempt logs.

Reads attemptlog.jsonl files, extracts per-event token usage, and
re-prices it with per-model pricing tables — so totals can be recomputed
when pricing changes, without re-running anything.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from groundhog.backends.anthropic import PRICING as ANTHROPIC_PRICING
from groundhog.backends.gemini import PRICING as GEMINI_PRICING

DEFAULT_PRICING = {**GEMINI_PRICING, **ANTHROPIC_PRICING}


def estimate_cost(path: Path, pricing: Optional[Dict] = None) -> Dict[str, Any]:
    """Estimate cost from a single attemptlog.jsonl file.

    Args:
        path: path to attemptlog.jsonl or a directory containing it
        pricing: optional pricing override dict

    Returns:
        {"total_cost": float, "input_tokens": int, "output_tokens": int,
         "thinking_tokens": int, "calls": [...], "unknown_models": [...]}
    """
    pricing = pricing or DEFAULT_PRICING
    p = Path(path)
    jsonl = p if p.name == "attemptlog.jsonl" else p / "attemptlog.jsonl"

    total_cost = 0.0
    total_input = 0
    total_output = 0
    total_thinking = 0
    calls = []
    unknown_models = set()

    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            usage = entry.get("usage")
            if not usage:
                continue

            model = entry.get("role", "")
            tokens = _extract_tokens(model, usage)
            embedded = usage.get("cost")
            if embedded:
                # OpenRouter reports the actual charge per call — trust it
                # over our tables.
                cost = float(embedded)
            else:
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
    """Aggregate cost across all attemptlog.jsonl files under base_path."""
    pricing = pricing or DEFAULT_PRICING
    total_cost = 0.0
    total_input = 0
    total_output = 0
    total_thinking = 0
    all_calls = []
    unknown_models = set()

    for jsonl in Path(base_path).rglob("attemptlog.jsonl"):
        result = estimate_cost(jsonl, pricing)
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


# --- Token extraction (provider-specific usage shapes) ---

def _extract_tokens(model: str, usage: Dict) -> Dict[str, int]:
    if model.startswith("gemini") or "promptTokenCount" in usage:
        return {
            "input": usage.get("promptTokenCount", 0),
            "output": usage.get("candidatesTokenCount", 0),
            "thinking": usage.get("thoughtsTokenCount", 0),
        }
    if "input_tokens" in usage:  # Anthropic
        return {
            "input": usage.get("input_tokens", 0),
            "output": usage.get("output_tokens", 0),
            "thinking": 0,
        }
    if "prompt_tokens" in usage:  # OpenAI / OpenRouter
        return {
            "input": usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0),
            "thinking": 0,
        }
    return {"input": 0, "output": 0, "thinking": 0}


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
