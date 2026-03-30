"""Mock LLM backend for testing. No network calls."""

from groundhog.base.backend import LLMBackend, LLMResponse, Prompt


class MockBackend(LLMBackend):
    """Cycles through predefined responses. For testing without network."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._index = 0

    def generate(self, prompt: Prompt, system_prompt: str = "") -> LLMResponse:
        text = self._responses[self._index % len(self._responses)]
        self._index += 1
        return LLMResponse(text=text, model="mock")
