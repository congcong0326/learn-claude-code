from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "s05_todo_write" / "code.py"


def load_s05_module():
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

    spec = importlib.util.spec_from_file_location("s05_under_test", MODULE_PATH)
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


def test_s05_uses_openai_function_tool_schema():
    module = load_s05_module()

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
        function_tool(
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
        ),
    ]


def test_s05_agent_loop_dispatches_todo_write_and_resets_nag_counter(monkeypatch):
    module = load_s05_module()
    calls = []
    seen = []

    def fake_todo_write(todos: list) -> str:
        seen.append(todos)
        return f"Updated {len(todos)} tasks"

    tool_call = types.SimpleNamespace(
        id="call_1",
        type="function",
        function=types.SimpleNamespace(
            name="todo_write",
            arguments=(
                '{"todos": ['
                '{"content": "inspect files", "status": "completed"}, '
                '{"content": "patch code", "status": "in_progress"}'
                "]}"
            ),
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
    monkeypatch.setitem(module.TOOL_HANDLERS, "todo_write", fake_todo_write)
    module.HOOKS["PreToolUse"] = []
    module.HOOKS["PostToolUse"] = []
    module.HOOKS["Stop"] = []
    module.rounds_since_todo = 2

    messages = [{"role": "user", "content": "plan this"}]
    module.agent_loop(messages)

    assert len(calls) == 2
    assert messages[1]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "todo_write",
                "arguments": (
                    '{"todos": ['
                    '{"content": "inspect files", "status": "completed"}, '
                    '{"content": "patch code", "status": "in_progress"}'
                    "]}"
                ),
            },
        }
    ]
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "Updated 2 tasks",
    }
    assert messages[-1] == {
        "role": "assistant",
        "content": "done",
    }
    assert calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "Updated 2 tasks",
    }
    assert seen == [
        [
            {"content": "inspect files", "status": "completed"},
            {"content": "patch code", "status": "in_progress"},
        ]
    ]
    assert module.rounds_since_todo == 0


def test_s05_nag_reminder_is_sent_as_openai_user_message():
    module = load_s05_module()
    calls = []
    final_message = types.SimpleNamespace(content="done", tool_calls=None)

    def fake_create(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=final_message)]
        )

    module.client.chat.completions.create = fake_create
    module.HOOKS["Stop"] = []
    module.rounds_since_todo = 3

    messages = [{"role": "user", "content": "continue"}]
    module.agent_loop(messages)

    assert messages[1] == {
        "role": "user",
        "content": "<reminder>Update your todos.</reminder>",
    }
    assert calls[0]["messages"][-1] == {
        "role": "user",
        "content": "<reminder>Update your todos.</reminder>",
    }
    assert module.rounds_since_todo == 0
