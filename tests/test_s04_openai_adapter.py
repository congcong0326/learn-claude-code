from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "s04_hooks" / "code.py"


def load_s04_module():
    fake_openai = types.ModuleType("openai")
    fake_anthropic = types.ModuleType("anthropic")
    fake_dotenv = types.ModuleType("dotenv")

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=None)
            )

    class FakeAnthropic:
        def __init__(self, *args, **kwargs):
            self.messages = types.SimpleNamespace(create=None)

    fake_openai.OpenAI = FakeOpenAI
    fake_anthropic.Anthropic = FakeAnthropic
    fake_dotenv.load_dotenv = lambda override=True: None

    previous_modules = {
        name: sys.modules.get(name) for name in ("openai", "anthropic", "dotenv")
    }
    previous_env = {
        name: os.environ.get(name)
        for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL_ID", "MODEL_ID")
    }

    spec = importlib.util.spec_from_file_location("s04_under_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)

    try:
        sys.modules["openai"] = fake_openai
        sys.modules["anthropic"] = fake_anthropic
        sys.modules["dotenv"] = fake_dotenv
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["OPENAI_BASE_URL"] = "https://example.test/v1"
        os.environ["OPENAI_MODEL_ID"] = "test-model"
        os.environ["MODEL_ID"] = "legacy-model"
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        for name, previous in previous_env.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


def function_tool(name: str, description: str, properties: dict, required: list[str]):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def test_s04_uses_openai_function_tool_schema():
    module = load_s04_module()

    assert module.MODEL == "test-model"
    assert module.TOOLS == [
        function_tool(
            "bash",
            "Run a shell command.",
            {"command": {"type": "string"}},
            ["command"],
        ),
        function_tool(
            "read_file",
            "Read file contents.",
            {"path": {"type": "string"}, "limit": {"type": "integer"}},
            ["path"],
        ),
        function_tool(
            "write_file",
            "Write content to a file.",
            {"path": {"type": "string"}, "content": {"type": "string"}},
            ["path", "content"],
        ),
        function_tool(
            "edit_file",
            "Replace exact text in a file once.",
            {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            ["path", "old_text", "new_text"],
        ),
        function_tool(
            "glob",
            "Find files matching a glob pattern.",
            {"pattern": {"type": "string"}},
            ["pattern"],
        ),
    ]


def test_s04_hooks_receive_openai_tool_block_and_append_tool_result(monkeypatch):
    module = load_s04_module()
    calls = []
    seen = []

    def fake_read_file(path: str, limit: int | None = None) -> str:
        return f"read {path} limit={limit}"

    def pre_hook(block):
        seen.append(("pre", block.id, block.name, block.input))
        return None

    def post_hook(block, output):
        seen.append(("post", block.id, block.name, output))
        return None

    tool_call = types.SimpleNamespace(
        id="call_1",
        type="function",
        function=types.SimpleNamespace(
            name="read_file",
            arguments='{"path": "README.md", "limit": 3}',
        ),
    )
    first_message = types.SimpleNamespace(content=None, tool_calls=[tool_call])
    second_message = types.SimpleNamespace(content="done", tool_calls=None)
    responses = [
        types.SimpleNamespace(choices=[types.SimpleNamespace(message=first_message)]),
        types.SimpleNamespace(choices=[types.SimpleNamespace(message=second_message)]),
    ]

    def fake_create(**kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    module.client.chat.completions.create = fake_create
    monkeypatch.setitem(module.TOOL_HANDLERS, "read_file", fake_read_file)
    module.HOOKS["PreToolUse"] = [pre_hook]
    module.HOOKS["PostToolUse"] = [post_hook]
    module.HOOKS["Stop"] = []

    messages = [{"role": "user", "content": "read the README"}]
    module.agent_loop(messages)

    assert len(calls) == 2
    assert messages[1]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path": "README.md", "limit": 3}',
            },
        }
    ]
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "read README.md limit=3",
    }
    assert messages[-1] == {
        "role": "assistant",
        "content": "done",
    }
    assert calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "read README.md limit=3",
    }
    assert seen == [
        ("pre", "call_1", "read_file", {"path": "README.md", "limit": 3}),
        ("post", "call_1", "read_file", "read README.md limit=3"),
    ]


def test_s04_pre_tool_hook_can_block_openai_tool_call(monkeypatch):
    module = load_s04_module()
    calls = []

    def fake_run_bash(command: str) -> str:
        raise AssertionError(f"bash should not run: {command}")

    tool_call = types.SimpleNamespace(
        id="call_1",
        type="function",
        function=types.SimpleNamespace(
            name="bash",
            arguments='{"command": "rm scratch.txt"}',
        ),
    )
    first_message = types.SimpleNamespace(content=None, tool_calls=[tool_call])
    second_message = types.SimpleNamespace(content="done", tool_calls=None)
    responses = [
        types.SimpleNamespace(choices=[types.SimpleNamespace(message=first_message)]),
        types.SimpleNamespace(choices=[types.SimpleNamespace(message=second_message)]),
    ]

    def fake_create(**kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    module.client.chat.completions.create = fake_create
    monkeypatch.setitem(module.TOOL_HANDLERS, "bash", fake_run_bash)
    module.HOOKS["PreToolUse"] = [lambda block: "Permission denied by hook"]
    module.HOOKS["PostToolUse"] = []
    module.HOOKS["Stop"] = []

    messages = [{"role": "user", "content": "delete scratch"}]
    module.agent_loop(messages)

    assert len(calls) == 2
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "Permission denied by hook",
    }
    assert calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "Permission denied by hook",
    }
