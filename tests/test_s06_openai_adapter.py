from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "s06_subagent" / "code.py"


def load_s06_module():
    fake_openai = types.ModuleType("openai")
    fake_anthropic = types.ModuleType("anthropic")
    fake_dotenv = types.ModuleType("dotenv")

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.init_args = args
            self.init_kwargs = kwargs
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

    spec = importlib.util.spec_from_file_location("s06_under_test", MODULE_PATH)
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


def test_s06_uses_openai_function_tool_schema_with_task_only_on_parent():
    module = load_s06_module()

    base_tools = [
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
    todo_tool = function_tool(
        "todo_write",
        "Create and manage a task list for your current coding session.",
        {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                        },
                    },
                    "required": ["content", "status"],
                },
            }
        },
        ["todos"],
    )
    task_tool = function_tool(
        "task",
        "Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
        {"description": {"type": "string"}},
        ["description"],
    )

    assert module.MODEL == "test-model"
    assert module.client.init_kwargs == {
        "api_key": "test-key",
        "base_url": "https://example.test/v1",
    }
    assert module.TOOLS == [*base_tools, todo_tool, task_tool]
    assert module.SUB_TOOLS == base_tools


def test_s06_agent_loop_dispatches_task_as_openai_tool_result(monkeypatch):
    module = load_s06_module()
    calls = []

    def fake_spawn_subagent(description: str) -> str:
        return f"subagent summary for {description}"

    tool_call = types.SimpleNamespace(
        id="call_task",
        type="function",
        function=types.SimpleNamespace(
            name="task",
            arguments='{"description": "inspect the project"}',
        ),
    )
    first_message = types.SimpleNamespace(content=None, tool_calls=[tool_call])
    second_message = types.SimpleNamespace(content="parent done", tool_calls=None)
    responses = [
        types.SimpleNamespace(choices=[types.SimpleNamespace(message=first_message)]),
        types.SimpleNamespace(choices=[types.SimpleNamespace(message=second_message)]),
    ]

    def fake_create(**kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    module.client.chat.completions.create = fake_create
    monkeypatch.setitem(module.TOOL_HANDLERS, "task", fake_spawn_subagent)
    module.HOOKS["PreToolUse"] = []
    module.HOOKS["PostToolUse"] = []
    module.HOOKS["Stop"] = []

    messages = [{"role": "user", "content": "delegate this"}]
    module.agent_loop(messages)

    assert len(calls) == 2
    assert calls[0]["messages"][0] == {"role": "system", "content": module.SYSTEM}
    assert calls[0]["tools"] == module.TOOLS
    assert messages[1]["tool_calls"] == [
        {
            "id": "call_task",
            "type": "function",
            "function": {
                "name": "task",
                "arguments": '{"description": "inspect the project"}',
            },
        }
    ]
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call_task",
        "content": "subagent summary for inspect the project",
    }
    assert messages[-1] == {"role": "assistant", "content": "parent done"}
    assert calls[1]["messages"][-1] == messages[2]


def test_s06_spawn_subagent_uses_fresh_openai_messages_and_returns_final_text(monkeypatch):
    module = load_s06_module()
    calls = []
    seen = []

    def fake_read_file(path: str) -> str:
        seen.append(path)
        return "read result"

    tool_call = types.SimpleNamespace(
        id="call_read",
        type="function",
        function=types.SimpleNamespace(
            name="read_file",
            arguments='{"path": "README.md"}',
        ),
    )
    first_message = types.SimpleNamespace(content=None, tool_calls=[tool_call])
    second_message = types.SimpleNamespace(content="subagent final", tool_calls=None)
    responses = [
        types.SimpleNamespace(choices=[types.SimpleNamespace(message=first_message)]),
        types.SimpleNamespace(choices=[types.SimpleNamespace(message=second_message)]),
    ]

    def fake_create(**kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    module.client.chat.completions.create = fake_create
    monkeypatch.setitem(module.SUB_HANDLERS, "read_file", fake_read_file)
    module.HOOKS["PreToolUse"] = []
    module.HOOKS["PostToolUse"] = []

    result = module.spawn_subagent("read the README")

    assert result == "subagent final"
    assert seen == ["README.md"]
    assert len(calls) == 2
    assert calls[0]["messages"] == [
        {"role": "system", "content": module.SUB_SYSTEM},
        {"role": "user", "content": "read the README"},
    ]
    assert calls[0]["tools"] == module.SUB_TOOLS
    assert all(
        tool["function"]["name"] != "task"
        for tool in calls[0]["tools"]
    )
    assert calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_read",
        "content": "read result",
    }
