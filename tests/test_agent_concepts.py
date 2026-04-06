"""Tests for the agent system: AgentTool, agent_tool(), ToolServer, AgentRegistry.

Concept → Test mapping:
    AgentTool + agent_tool() (base/agent.py)
        - Factory creates callable tools from any function
        - Type coercion from strings to declared types
        - Return conversion (str, dict, list, None)
        - Error handling wraps exceptions in ToolResult
    AgentRegistry (base/agent.py)
        - Tier-based lookup with fallback to "default"
    ToolServer (agents/tool_server.py)
        - HTTP POST to tool endpoints
        - Bash wrapper generation (positional + kwargs)
    Default agent tools (agents/tools.py)
        - Learnings tool wraps MarkdownLearnings.get()
"""

import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from groundhog.base.agent import (
    AgentTool, ToolResult, agent_tool,
    AgentSpec, AgentResult,
    AgentBackend, AgentRegistry,
)


# === agent_tool() factory ===

def test_agent_tool_no_params():
    """Tool with no parameters — just calls the function."""
    tool = agent_tool("ping", "Returns pong", func=lambda: "pong")
    assert tool.name == "ping"
    assert tool.description == "Returns pong"
    assert tool.get_parameters() == {}
    result = tool.execute()
    assert result.success
    assert result.output == "pong"


def test_agent_tool_with_params_and_coercion():
    """Parameters arrive as strings from bash — must coerce to declared types."""
    def add(a: int, b: int) -> int:
        return a + b

    tool = agent_tool(
        "add", "Add two numbers", func=add,
        params={
            "a": {"type": "int", "description": "First number"},
            "b": {"type": "int", "description": "Second number"},
        },
    )
    # Simulate bash wrapper sending strings
    result = tool.execute(a="3", b="7")
    assert result.success
    assert result.output == "10"


def test_agent_tool_defaults():
    """Optional params with defaults work when not provided."""
    def greet(name: str = "world") -> str:
        return f"hello {name}"

    tool = agent_tool(
        "greet", "Say hello", func=greet,
        params={"name": {"type": "str", "default": "world", "description": "Who to greet"}},
    )
    # Call without providing name — should use default via function signature
    # (the wrapper doesn't inject defaults, the function has them)
    result = tool.execute()
    assert result.success
    assert result.output == "hello world"


def test_agent_tool_dict_return():
    """Dict/list returns are JSON-serialized."""
    tool = agent_tool("info", "Get info", func=lambda: {"status": "ok", "count": 42})
    result = tool.execute()
    assert result.success
    parsed = json.loads(result.output)
    assert parsed["status"] == "ok"
    assert parsed["count"] == 42


def test_agent_tool_none_return():
    """None return becomes empty string."""
    tool = agent_tool("noop", "Do nothing", func=lambda: None)
    result = tool.execute()
    assert result.success
    assert result.output == ""


def test_agent_tool_error_handling():
    """Exceptions are caught and returned as ToolResult errors."""
    def fail():
        raise ValueError("something broke")

    tool = agent_tool("fail", "Always fails", func=fail)
    result = tool.execute()
    assert not result.success
    assert "something broke" in result.error


def test_agent_tool_float_coercion():
    """Float coercion from string."""
    tool = agent_tool(
        "scale", "Scale a value", func=lambda x: float(x) * 2,
        params={"x": {"type": "float"}},
    )
    result = tool.execute(x="3.14")
    assert result.success
    assert "6.28" in result.output


def test_agent_tool_bool_coercion():
    """Bool coercion from string."""
    tool = agent_tool(
        "flag", "Check flag", func=lambda v: f"flag={v}",
        params={"v": {"type": "bool"}},
    )
    assert tool.execute(v="true").output == "flag=True"
    assert tool.execute(v="false").output == "flag=False"
    assert tool.execute(v="1").output == "flag=True"


# === AgentSpec / AgentResult ===

def test_agent_spec_construction():
    """AgentSpec can be constructed with minimal args."""
    spec = AgentSpec(goal="do stuff", workspace_path=Path("/tmp"))
    assert spec.goal == "do stuff"
    assert spec.tools == []
    assert spec.model is None
    assert spec.effort is None
    assert spec.session_id is None


def test_agent_result_construction():
    """AgentResult with all fields."""
    result = AgentResult(
        success=True, output="done",
        session_id="abc-123", cost=0.05, turns=3, duration_ms=1500,
    )
    assert result.success
    assert result.session_id == "abc-123"
    assert result.cost == 0.05


# === AgentRegistry ===

class MockAgentBackend(AgentBackend):
    def __init__(self, name="mock"):
        self.name = name
    def run(self, spec):
        return AgentResult(success=True, output=f"from {self.name}")


def test_registry_get_default():
    """Registry returns default tier."""
    reg = AgentRegistry(default=MockAgentBackend("default"))
    assert reg.get().name == "default"
    assert reg.get("default").name == "default"


def test_registry_fallback_to_default():
    """Unknown tier falls back to default."""
    reg = AgentRegistry(default=MockAgentBackend("default"))
    assert reg.get("nonexistent").name == "default"


def test_registry_multiple_tiers():
    """Multiple tiers, each returns correct backend."""
    reg = AgentRegistry(
        default=MockAgentBackend("default"),
        budget=MockAgentBackend("budget"),
    )
    assert reg.get("default").name == "default"
    assert reg.get("budget").name == "budget"


def test_registry_no_default_raises():
    """No default and unknown tier raises KeyError."""
    reg = AgentRegistry(budget=MockAgentBackend("budget"))
    try:
        reg.get("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_registry_set_override():
    """set() overrides a tier."""
    reg = AgentRegistry(default=MockAgentBackend("old"))
    reg.set("default", MockAgentBackend("new"))
    assert reg.get().name == "new"


# === ToolServer ===

def test_tool_server_http():
    """ToolServer serves tools via HTTP POST."""
    from groundhog.agents.tool_server import ToolServer

    tool = agent_tool("echo", "Echo back", func=lambda msg: msg,
                       params={"msg": {"type": "str"}})
    server = ToolServer([tool])
    try:
        port = server.start()
        assert port > 0

        # POST to tool endpoint
        body = json.dumps({"msg": "hello"}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/echo",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())

        assert data["success"]
        assert data["output"] == "hello"
    finally:
        server.stop()


def test_tool_server_unknown_tool():
    """Unknown tool returns 404."""
    from groundhog.agents.tool_server import ToolServer

    server = ToolServer([])
    try:
        port = server.start()
        body = json.dumps({}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/nonexistent",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        assert not data["success"]
    except urllib.error.HTTPError as e:
        assert e.code == 404
    finally:
        server.stop()


# === Bash wrapper generation ===

def test_wrapper_generation_positional():
    """Generated wrappers work with positional args."""
    from groundhog.agents.tool_server import ToolServer, generate_wrappers, cleanup_wrappers

    tool = agent_tool("double", "Double a number",
                       func=lambda n: str(int(n) * 2),
                       params={"n": {"type": "int"}})
    server = ToolServer([tool])
    bin_dir = Path(tempfile.mkdtemp(prefix="test_wrappers_"))

    try:
        port = server.start()
        generate_wrappers([tool], bin_dir, port)

        wrapper = bin_dir / "double"
        assert wrapper.exists()
        assert wrapper.stat().st_mode & stat.S_IEXEC

        # Run wrapper with positional arg
        result = subprocess.run(
            [str(wrapper), "21"],
            capture_output=True, text=True,
            env={**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]},
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "42"
    finally:
        server.stop()
        cleanup_wrappers(bin_dir)


def test_wrapper_generation_kwargs():
    """Generated wrappers work with --kwargs."""
    from groundhog.agents.tool_server import ToolServer, generate_wrappers, cleanup_wrappers

    tool = agent_tool("greet", "Greet someone",
                       func=lambda name, style="hello": f"{style} {name}",
                       params={
                           "name": {"type": "str"},
                           "style": {"type": "str", "default": "hello"},
                       })
    server = ToolServer([tool])
    bin_dir = Path(tempfile.mkdtemp(prefix="test_wrappers_"))

    try:
        port = server.start()
        generate_wrappers([tool], bin_dir, port)

        wrapper = bin_dir / "greet"

        # Test --kwargs mode
        result = subprocess.run(
            [str(wrapper), "--name", "world", "--style", "hi"],
            capture_output=True, text=True,
            env={**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]},
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "hi world"
    finally:
        server.stop()
        cleanup_wrappers(bin_dir)


# === Tool docs ===

def test_build_tool_docs():
    """build_tool_docs() produces readable markdown."""
    from groundhog.agents.tool_server import build_tool_docs

    tools = [
        agent_tool("eval", "Run evaluation", func=lambda: "ok"),
        agent_tool("get-learnings", "Read learnings", func=lambda last="10": "...",
                    params={"last": {"type": "int", "default": 10, "description": "Recent entries"}}),
    ]
    docs = build_tool_docs(tools)
    assert "### eval" in docs
    assert "### get-learnings" in docs
    assert "Recent entries" in docs
    assert "--last" in docs


# === Default agent tools (learnings) ===

def test_learnings_tool():
    """Learnings tool wraps MarkdownLearnings correctly."""
    from groundhog.learnings.markdown import MarkdownLearnings
    from groundhog.agents.tools import build_default_agent_tools
    from groundhog.base.toolkit import Toolkit

    with tempfile.TemporaryDirectory() as tmpdir:
        learnings = MarkdownLearnings(Path(tmpdir))
        learnings.add("KNN works well for small datasets")
        learnings.add("Augmentation beyond 10x hurts accuracy")

        toolkit = Toolkit()
        toolkit.learnings = learnings
        tools = build_default_agent_tools(toolkit)

        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "get-learnings"

        result = tool.execute(last="2", random="0")
        assert result.success
        assert "KNN" in result.output
        assert "Augmentation" in result.output


# === Eval tools via build_default_agent_tools ===

def test_eval_tools_from_toolkit():
    """Eval tools are built from toolkit.task eval stages."""
    from groundhog.agents.tools import build_default_agent_tools
    from groundhog.base.toolkit import Toolkit
    from groundhog.base.types import Task, Data, Context, Evaluator, EvalStage, StageResult

    class SimpleData(Data):
        def get_train(self): return {}
        def get_test(self): return {}

    class SimpleContext(Context):
        def get_brief(self): return "Test"
        def get_extended(self): return "Test task"

    class SimpleEvaluator(Evaluator):
        def evaluate(self, code_or_path, data):
            ns = {}
            exec(code_or_path, ns)
            return StageResult(score=ns.get("SCORE", 0.0), metrics={"score": ns.get("SCORE", 0.0)})

        def get_stages(self, data):
            return [
                EvalStage("smoke", "Quick check",
                          lambda cp: StageResult(score=1.0) if "SCORE" in cp else StageResult(errors={"missing": "no SCORE"})),
                EvalStage("evaluate", "Full eval",
                          lambda cp, d=data: self.evaluate(cp, d)),
            ]

    task = Task(data=SimpleData(), context=SimpleContext(), evaluator=SimpleEvaluator(), name="Test")
    toolkit = Toolkit(task=task)
    tools = build_default_agent_tools(toolkit)

    # Should have smoke + evaluate (no learnings since toolkit.learnings not set)
    tool_names = [t.name for t in tools]
    assert "smoke" in tool_names
    assert "evaluate" in tool_names

    # Write a test file and evaluate it
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.py"
        test_file.write_text("SCORE = 0.85\n")

        eval_tool = [t for t in tools if t.name == "evaluate"][0]
        result = eval_tool.execute(path=str(test_file))
        assert result.success
        assert "0.85" in result.output


def test_eval_tool_with_bad_code():
    """Eval tool handles errors gracefully."""
    from groundhog.agents.tools import build_default_agent_tools
    from groundhog.base.toolkit import Toolkit
    from groundhog.base.types import Task, Data, Context, Evaluator, EvalStage, StageResult

    class SimpleData(Data):
        def get_train(self): return {}
        def get_test(self): return {}

    class SimpleContext(Context):
        def get_brief(self): return "Test"
        def get_extended(self): return "Test"

    class SimpleEvaluator(Evaluator):
        def evaluate(self, code_or_path, data):
            return StageResult(score=0.0)
        def get_stages(self, data):
            return [EvalStage("evaluate", "Full eval",
                              lambda cp, d=data: self.evaluate(cp, d))]

    task = Task(data=SimpleData(), context=SimpleContext(), evaluator=SimpleEvaluator(), name="Test")
    toolkit = Toolkit(task=task)
    tools = build_default_agent_tools(toolkit)
    eval_tool = tools[0]

    # Non-existent file
    result = eval_tool.execute(path="/nonexistent/file.py")
    assert not result.success
    assert result.error  # should have error message


# === eval_to_dir ===

def test_eval_to_dir_writes_results_json():
    """eval_to_dir writes results.json with score and metrics."""
    from groundhog.agents.tools import eval_to_dir
    from groundhog.base.types import EvalStage, StageResult

    stage = EvalStage("validate", "Quick test",
                       lambda code: StageResult(score=0.85, metrics={"accuracy": 0.85, "time": 1.2}))

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "solution.py"
        test_file.write_text("SCORE = 0.85\n")

        result = eval_to_dir(stage, str(test_file), str(Path(tmpdir) / "output" / "validate"))

        # results.json written
        results_path = Path(tmpdir) / "output" / "validate" / "results.json"
        assert results_path.exists()
        data = json.loads(results_path.read_text())
        assert data["score"] == 0.85
        assert data["metrics"]["accuracy"] == 0.85

        # Result unchanged
        assert result.score == 0.85


def test_eval_to_dir_writes_artifacts():
    """eval_to_dir writes artifacts to disk and replaces with paths."""
    from groundhog.agents.tools import eval_to_dir
    from groundhog.base.types import EvalStage, StageResult

    stage = EvalStage("evaluate", "Full eval",
                       lambda code: StageResult(
                           score=0.9,
                           metrics={"score": 0.9},
                           artifacts={
                               "report.txt": "detailed report here",
                               "data.json": {"classes": [0.9, 0.8, 0.7]},
                               "plot.png": b"\x89PNG fake image data",
                           }))

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "solution.py"
        test_file.write_text("pass\n")
        out_dir = str(Path(tmpdir) / "output" / "evaluate")

        result = eval_to_dir(stage, str(test_file), out_dir)

        # Artifacts written to disk
        assert (Path(out_dir) / "report.txt").exists()
        assert (Path(out_dir) / "data.json").exists()
        assert (Path(out_dir) / "plot.png").exists()

        # Text artifact content
        assert (Path(out_dir) / "report.txt").read_text() == "detailed report here"

        # Binary artifact content
        assert (Path(out_dir) / "plot.png").read_bytes() == b"\x89PNG fake image data"

        # Result artifacts replaced with paths
        assert "report.txt" in result.artifacts
        assert result.artifacts["report.txt"].endswith("report.txt")
        assert isinstance(result.artifacts["report.txt"], str)  # path, not content


def test_eval_to_dir_does_not_mutate_original():
    """eval_to_dir returns a copy, original StageResult untouched."""
    from groundhog.agents.tools import eval_to_dir
    from groundhog.base.types import EvalStage, StageResult

    original_artifacts = {"data.txt": "original content"}
    stage = EvalStage("test", "Test",
                       lambda code: StageResult(score=1.0, artifacts=dict(original_artifacts)))

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "code.py"
        test_file.write_text("pass\n")

        result = eval_to_dir(stage, str(test_file), str(Path(tmpdir) / "out"))

        # Returned result has path
        assert result.artifacts["data.txt"].endswith("data.txt")

        # Call again — original stage still returns content, not paths
        fresh = stage.call("pass")
        assert fresh.artifacts["data.txt"] == "original content"


# === Cost model ===

def test_cost_model_default():
    """AgentBackend defaults to per_token."""
    from groundhog.base.agent import AgentBackend
    class TestBackend(AgentBackend):
        def run(self, spec): pass
    assert TestBackend().cost_model == "per_token"


def test_cost_model_copilot():
    """CopilotAgentBackend declares per_request."""
    from groundhog.agents.copilot import CopilotAgentBackend
    assert CopilotAgentBackend().cost_model == "per_request"


def test_cost_model_claude():
    """ClaudeCodeAgentBackend keeps per_token default."""
    from groundhog.agents.claude_code import ClaudeCodeAgentBackend
    assert ClaudeCodeAgentBackend().cost_model == "per_token"


# === Permission translation ===

def test_copilot_permission_translation():
    """Copilot translates Claude-style permissions to its own syntax."""
    from groundhog.agents.copilot import _translate_permission

    assert _translate_permission("Bash(git:*)") == "shell(git:*)"
    assert _translate_permission("Bash(validate)") == "shell(validate)"
    assert _translate_permission("Write(*)") == "write"
    assert _translate_permission("Write(workspace/*)") == "write(workspace/*)"
    assert _translate_permission("Edit(solution.py)") == "write(solution.py)"
    assert _translate_permission("Read(*)") == "read"
    assert _translate_permission("Read(src/*)") == "read(src/*)"


# === Copilot event parsing ===

def test_copilot_event_summarize():
    """Copilot event summarizer extracts key content using actual schema."""
    from groundhog.agents.copilot import _summarize_event

    # assistant.message with text
    lines = _summarize_event({
        "type": "assistant.message",
        "data": {"content": "hello world", "toolRequests": []},
    })
    assert len(lines) == 1
    assert lines[0]["role"] == "assistant"
    assert lines[0]["content"] == "hello world"

    # assistant.message with tool request (actual copilot schema)
    lines = _summarize_event({
        "type": "assistant.message",
        "data": {
            "content": "",
            "toolRequests": [{
                "toolCallId": "toolu_abc",
                "name": "bash",
                "arguments": {"command": "ls"},
                "type": "function",
            }],
        },
    })
    assert len(lines) == 1
    assert lines[0]["type"] == "tool_use"
    assert lines[0]["tool"] == "bash"
    assert lines[0]["input"]["command"] == "ls"

    # tool.execution_complete (actual schema: result.content, toolCallId)
    lines = _summarize_event({
        "type": "tool.execution_complete",
        "data": {
            "toolCallId": "toolu_abc",
            "result": {"content": "file.txt"},
        },
    })
    assert len(lines) == 1
    assert lines[0]["role"] == "tool_result"
    assert lines[0]["tool_use_id"] == "toolu_abc"
    assert lines[0]["content"] == "file.txt"

    # result event
    lines = _summarize_event({
        "type": "result",
        "sessionId": "sess-123",
        "usage": {"premiumRequests": 1, "sessionDurationMs": 5000},
    })
    assert len(lines) == 1
    assert lines[0]["session_id"] == "sess-123"

    # ephemeral events return nothing
    assert _summarize_event({"type": "assistant.message_delta", "data": {}}) == []
    assert _summarize_event({"type": "session.tools_updated", "data": {}}) == []


# === Run all tests ===

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in tests:
        name = test.__name__
        try:
            test()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
