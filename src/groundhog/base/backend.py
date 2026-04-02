"""LLM Backend — interface for language model calls.

Vault: Backend Tiers.md — strategies request LLMs by purpose tier, not model name.

Prompts can be a plain string (simple case) or a sequence of PromptParts
for multimodal content (text interleaved with images, audio, etc.).
Backend implementations convert parts to their provider's API format.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Union


# --- Prompt parts ---

@dataclass(frozen=True)
class PromptPart:
    """Base prompt part."""


@dataclass(frozen=True)
class TextPart(PromptPart):
    """Text content."""
    text: str


@dataclass(frozen=True)
class ImagePart(PromptPart):
    """Image content."""
    data: bytes
    mime_type: str = "image/png"


@dataclass(frozen=True)
class AudioPart(PromptPart):
    """Audio content."""
    data: bytes
    mime_type: str = "audio/wav"


Prompt = Union[str, Sequence[PromptPart]]


# --- Response ---

@dataclass
class LLMResponse:
    """Response from an LLM call."""
    text: str
    model: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0


# --- Backend interface ---

class LLMBackend(ABC):
    """Interface for a language model backend.

    generate() accepts a Prompt — either a plain string or a sequence of
    PromptParts for multimodal content. Implementations convert to their API format.
    """

    @abstractmethod
    def generate(self, prompt: Prompt, system_prompt: str = "") -> LLMResponse: ...


# --- Registry ---

class BackendRegistry:
    """Maps tier names to LLM backends. Strategies request by tier, not model.

    Usage:
        registry = BackendRegistry(default=gemini, cheap=flash)
        backend = registry.get("default")
        response = backend.generate("Write a function...")
        response = backend.generate([TextPart("Describe this:"), ImagePart(png_bytes)])
    """

    def __init__(self, **tiers: LLMBackend):
        self._tiers = tiers

    def set(self, tier: str, backend: LLMBackend):
        """Set or override a tier's backend."""
        self._tiers[tier] = backend

    def get(self, tier: str = "default") -> LLMBackend:
        if tier in self._tiers:
            return self._tiers[tier]
        if "default" in self._tiers:
            return self._tiers["default"]
        raise KeyError(f"No backend for tier '{tier}' and no default. Available: {list(self._tiers.keys())}")
