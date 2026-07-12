"""LLM robustness: empty/timed-out backend responses must not crash a worker.

Provenance: an MNIST integration run hit a reasoning model (minimax-m3) that
spent its whole 65536-token completion budget on hidden reasoning and returned
content: null. The Improve strategy then regexed over None, crashing the
optimizer worker and leaving an orphaned workspace with no failed record and a
vanished cost. These tests pin the three defenses:

  1. OpenAICompatibleBackend surfaces null content as a clear RuntimeError
     naming the finish_reason, and honors a request timeout.
  2. generate_text turns a raised/empty response into a retryable failure and,
     on exhaustion, a typed GenerationFailed.
  3. A strategy whose generation dies records a FAILED attempt (nothing
     discarded) instead of crashing and orphaning the workspace.
"""

import json
import tempfile
from pathlib import Path

import pytest

from groundhog import Task
from groundhog.assemble import assemble_toolkit
from groundhog.backends import openai_compat
from groundhog.backends.openai_compat import OpenAICompatibleBackend
from groundhog.base.backend import BackendRegistry, LLMBackend, LLMResponse
from groundhog.base.types import (
    Context, Data, EvalStage, EvaluationResult, Evaluator, StageResult,
)
from groundhog.strategies.improve import Improve
from groundhog.utils.codegen import GenerationFailed, extract_code, generate_text
from groundhog.utils.direction import write_direction
from groundhog.utils.finalize import finalize_attempt


class _FakeResp:
    """Minimal urlopen context-manager stand-in."""

    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# --- Backend: timeout + empty-content -----------------------------------

def test_backend_passes_timeout_to_urlopen(monkeypatch):
    captured = {}

    def fake_urlopen(req, label="", timeout=None, **kw):
        captured["timeout"] = timeout
        return _FakeResp(
            {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}
        )

    monkeypatch.setattr(openai_compat, "_urlopen_with_warnings", fake_urlopen)

    OpenAICompatibleBackend(model="m", api_key="k").generate("hello")
    assert captured["timeout"] == 120  # default

    OpenAICompatibleBackend(model="m", api_key="k", timeout_s=7).generate("hello")
    assert captured["timeout"] == 7


def test_backend_empty_content_raises_with_finish_reason(monkeypatch):
    monkeypatch.setattr(
        openai_compat, "_urlopen_with_warnings",
        lambda *a, **k: _FakeResp(
            {"choices": [{"message": {"content": None}, "finish_reason": "length"}]}
        ),
    )
    backend = OpenAICompatibleBackend(model="minimax-m3", api_key="k")
    with pytest.raises(RuntimeError) as ei:
        backend.generate("hi")
    msg = str(ei.value)
    assert "empty content" in msg
    assert "finish_reason=length" in msg


# --- generate_text: retry + typed failure -------------------------------

class _EmptyBackend(LLMBackend):
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, system_prompt=""):
        self.calls += 1
        return LLMResponse(text=None, model="m")


class _RaisingBackend(LLMBackend):
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, system_prompt=""):
        self.calls += 1
        raise RuntimeError("minimax-m3 returned empty content (finish_reason=length)")


def test_generate_text_retries_then_raises_on_empty():
    backend = _EmptyBackend()
    with pytest.raises(GenerationFailed):
        generate_text(backend, "p", retries=2)
    assert backend.calls == 3  # retry path exercised: retries + 1


def test_generate_text_retries_a_raised_backend_error():
    backend = _RaisingBackend()
    with pytest.raises(GenerationFailed) as ei:
        generate_text(backend, "p", retries=1)
    assert backend.calls == 2
    assert "finish_reason=length" in str(ei.value)


def test_generate_text_returns_first_usable_response():
    class _EventuallyOK(LLMBackend):
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, system_prompt=""):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(text="", model="m")
            return LLMResponse(text="def f(): return 1", model="m", cost=0.01)

    backend = _EventuallyOK()
    response = generate_text(backend, "p", retries=2)
    assert response.text == "def f(): return 1"
    assert backend.calls == 2


def test_extract_code_is_none_safe():
    code, diff = extract_code(None, "prior")
    assert code == ""
    assert diff.method == "none"


# --- Strategy level: empty content -> recorded failure, not a crash -----

class _Data(Data):
    def get_train(self):
        return None

    def get_test(self):
        return None


class _Ctx(Context):
    def get_brief(self):
        return "brief"

    def get_extended(self):
        return "extended"


class _Eval(Evaluator):
    def evaluate(self, code_or_path, data):
        return StageResult(metrics={"score": 0.5})

    def get_stages(self, data):
        return [
            EvalStage("eval", "eval",
                      lambda cp: StageResult(metrics={"score": 0.5}),
                      scorer=lambda r: r.metrics.get("score", 0.0))
        ]


def _seed_prior(tk):
    ws = tk.history.workspace()
    (ws.path / "solution.py").write_text("def solve(): return 0\n", encoding="utf-8")
    write_direction(ws.path, "rollout")
    result = EvaluationResult(stages={"eval": StageResult(metrics={"score": 0.5})})
    return finalize_attempt(tk, ws, result, None)


def test_improve_records_failed_attempt_when_generation_dies():
    with tempfile.TemporaryDirectory() as tmp:
        task = Task(data=_Data(), context=_Ctx(), evaluator=_Eval(), name="t")
        tk = assemble_toolkit(task, path=Path(tmp))

        prior = _seed_prior(tk)
        assert len(tk.history.list(only_done=False)) == 1

        backend = _EmptyBackend()
        tk.llm = BackendRegistry(default=backend, high=backend)

        strat = Improve()
        strat(tk)

        attempts = tk.history.list(only_done=False)
        # The failed attempt is recorded, not orphaned: exactly one new entry.
        assert len(attempts) == 2
        failed = [a for a in attempts if a.id != prior.id][0]
        assert failed.status == "fail"
        assert failed.metadata.get("generation_failed")
        # Retry path exercised at generation level: max_retries + 1 calls.
        assert backend.calls == strat.cfg.max_retries + 1
