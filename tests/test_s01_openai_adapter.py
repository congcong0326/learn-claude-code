from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "s01_agent_loop" / "code.py"


def load_s01_module():
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

    spec = importlib.util.spec_from_file_location("s01_under_test", MODULE_PATH)
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


def test_s01_uses_openai_function_tool_schema():
    module = load_s01_module()

    assert module.MODEL == "test-model"
    assert module.TOOLS == [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]


def test_s01_agent_loop_appends_openai_tool_result_message(monkeypatch):
    module = load_s01_module()
    calls = []

    def fake_run_bash(command: str) -> str:
        return f"ran: {command}"

    tool_call = types.SimpleNamespace(
        id="call_1",
        function=types.SimpleNamespace(name="bash", arguments='{"command": "pwd"}'),
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
    monkeypatch.setattr(module, "run_bash", fake_run_bash)

    messages = [{"role": "user", "content": "where am I?"}]
    module.agent_loop(messages)

    assert len(calls) == 2
    assert messages[1]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": '{"command": "pwd"}',
            },
        }
    ]
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "ran: pwd",
    }
    assert messages[-1] == {
        "role": "assistant",
        "content": "done",
    }
    assert calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "ran: pwd",
    }
