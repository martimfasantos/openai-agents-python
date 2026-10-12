from __future__ import annotations

import json
from enum import Enum
from typing import Any, cast

import pytest

from agents import Agent, RunConfig, Runner, WebSearchTool
from agents.decorators import tool
from agents.exceptions import UserError
from agents.items import ToolCallOutputItem
from agents.models.openai_decisions import (
    DEFAULT_AUTO_MODE_INSTRUCTIONS,
    OpenAIAutoMode,
    OpenAIDecisions,
)
from agents.testing import ScriptedModel, function_call
from agents.tool import FunctionTool, tool_namespace
from tests.models.openai_decisions.helpers import make_client, make_decision
from tests.test_responses import get_function_tool_call, get_text_message

CONFIG = RunConfig(tracing_disabled=True)
BLOCKED_RISKY = (
    "The tool call `delete_file` was blocked because it was classified as risky "
    "(probability: 0.90). The tool was not executed."
)
BLOCKED_REFUSED = (
    "The tool call `delete_file` was blocked because its risk could not be assessed. "
    "The tool was not executed."
)


def risk_decisions(*probabilities: float | None) -> tuple[OpenAIDecisions, Any]:
    client = make_client(
        *[
            make_decision(
                {"type": "refusal", "name": "is_risky"}
                if probability is None
                else {"type": "predicate", "name": "is_risky", "probability": probability}
            )
            for probability in probabilities
        ]
    )
    return OpenAIDecisions("gpt-6-luna", openai_client=client), client


def agent_calling(guarded: FunctionTool, arguments: str) -> Agent[Any]:
    model = ScriptedModel(
        [
            [get_function_tool_call(guarded.name, arguments, call_id="call_1")],
            [get_text_message("done")],
        ]
    )
    return Agent(name="assistant", model=model, tools=[guarded])


def delete_file_tool(calls: list[str]) -> FunctionTool:
    @tool
    def delete_file(path: str) -> str:
        """Delete a file."""
        calls.append(path)
        return "deleted"

    return delete_file


@pytest.mark.parametrize(
    "probability,expected_output",
    [(0.2, "deleted"), (0.9, BLOCKED_RISKY), (None, BLOCKED_REFUSED)],
)
async def test_reject_content_mode_blocks_risky_and_refused_calls(
    probability: float | None, expected_output: str
) -> None:
    calls: list[str] = []
    decisions, _ = risk_decisions(probability)
    [guarded] = OpenAIAutoMode(model=decisions).guard([delete_file_tool(calls)])

    result = await Runner.run(
        agent_calling(guarded, '{"path": "notes.txt"}'), "Summarize notes.txt", run_config=CONFIG
    )

    outputs = [item.output for item in result.new_items if isinstance(item, ToolCallOutputItem)]
    assert outputs == [expected_output]
    assert calls == (["notes.txt"] if expected_output == "deleted" else [])
    assert result.final_output == "done"


async def test_classification_payload_has_user_context_and_tool_call_without_media() -> None:
    @tool
    def send_email(to: str) -> str:
        """Send an email to a recipient."""
        return "sent"

    decisions, client = risk_decisions(0.1)
    [guarded] = OpenAIAutoMode(model=decisions).guard([send_email])
    user_input: list[Any] = [
        {
            "type": "computer_call_output",
            "call_id": "computer_1",
            "output": {"type": "computer_screenshot", "image_url": "data:image/png;base64,SHOT"},
        },
        {
            "type": "image_generation_call",
            "id": "image_1",
            "status": "completed",
            "result": "GENERATED_IMAGE_BYTES",
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Email the report to Ana."},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AAAA",
                    "detail": "auto",
                },
            ],
        },
    ]

    await Runner.run(
        agent_calling(guarded, '{"to": "ana@example.com"}'), user_input, run_config=CONFIG
    )

    kwargs = client.decisions.create.await_args.kwargs
    assert kwargs["questions"] == [
        {"type": "predicate", "name": "is_risky", "instructions": DEFAULT_AUTO_MODE_INSTRUCTIONS}
    ]
    assert "base64" not in kwargs["input"]
    assert "GENERATED_IMAGE_BYTES" not in kwargs["input"]
    payload = json.loads(kwargs["input"])
    assert payload["messages"][0]["output"] == "[computer_screenshot omitted]"
    assert payload["messages"][1] == "[image_generation_call omitted]"
    assert payload["messages"][-1]["role"] == "user"
    assert payload["messages"][-1]["content"] == [
        {"type": "input_text", "text": "Email the report to Ana."},
        "[input_image omitted]",
    ]
    assert payload["tool_call"] == {"name": "send_email", "arguments": '{"to": "ana@example.com"}'}
    assert payload["tool_description"] == "Send an email to a recipient."


async def test_require_approval_mode_runs_safe_calls_without_approval() -> None:
    calls: list[str] = []
    decisions, _ = risk_decisions(0.1)
    [guarded] = OpenAIAutoMode(model=decisions, on_risky="require_approval").guard(
        [delete_file_tool(calls)]
    )

    result = await Runner.run(
        agent_calling(guarded, '{"path": "tmp.txt"}'), "Delete tmp.txt", run_config=CONFIG
    )

    assert result.interruptions == []
    assert calls == ["tmp.txt"]


async def test_require_approval_mode_interrupts_risky_calls_and_resumes_after_approval() -> None:
    calls: list[str] = []
    decisions, client = risk_decisions(0.9)
    [guarded] = OpenAIAutoMode(model=decisions, on_risky="require_approval").guard(
        [delete_file_tool(calls)]
    )
    agent = agent_calling(guarded, '{"path": "notes.txt"}')

    result = await Runner.run(agent, "Summarize notes.txt", run_config=CONFIG)
    assert len(result.interruptions) == 1
    assert calls == []

    state = result.to_state()
    state.approve(result.interruptions[0])
    resumed = await Runner.run(agent, state, run_config=CONFIG)

    assert calls == ["notes.txt"]
    assert resumed.final_output == "done"
    assert client.decisions.create.await_count == 1


class Speed(Enum):
    SLOW = "slow"
    FAST = "fast"


async def test_require_approval_mode_pauses_calls_whose_arguments_need_conversion() -> None:
    @tool
    def set_speed(speed: Speed) -> str:
        """Set the fan speed."""
        return speed.value

    decisions, client = risk_decisions(0.01)
    [guarded] = OpenAIAutoMode(model=decisions, on_risky="require_approval").guard([set_speed])

    result = await Runner.run(
        agent_calling(guarded, '{"speed": "slow"}'), "Slow the fan down", run_config=CONFIG
    )

    # The SDK requires approval before converting arguments, so the classifier is never asked.
    assert len(result.interruptions) == 1
    client.decisions.create.assert_not_awaited()


async def test_classification_errors_fail_the_run_without_running_the_tool() -> None:
    calls: list[str] = []
    client = make_client(RuntimeError("decisions unavailable"))
    auto_mode = OpenAIAutoMode(model=OpenAIDecisions("gpt-6-luna", openai_client=client))
    [guarded] = auto_mode.guard([delete_file_tool(calls)])

    with pytest.raises(UserError, match="decisions unavailable") as exc_info:
        await Runner.run(agent_calling(guarded, '{"path": "a.txt"}'), "hi", run_config=CONFIG)
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert calls == []


def test_guard_returns_copies_and_leaves_original_tools_unchanged() -> None:
    original = delete_file_tool([])

    [guarded] = OpenAIAutoMode(model="gpt-6-luna").guard([original])

    assert guarded is not original
    assert original.tool_input_guardrails is None
    assert [g.get_name() for g in guarded.tool_input_guardrails or []] == ["openai_auto_mode"]


def test_guard_rejects_non_function_tools_and_existing_approval_policies() -> None:
    @tool(needs_approval=True)
    def wipe_disk(device: str) -> str:
        """Erase a disk."""
        return "wiped"

    approval_mode = OpenAIAutoMode(model="gpt-6-luna", on_risky="require_approval")

    with pytest.raises(UserError, match="needs_approval"):
        approval_mode.guard([wipe_disk])
    with pytest.raises(UserError, match="FunctionTool"):
        approval_mode.guard([cast(Any, WebSearchTool())])


async def test_classification_keeps_json_schemas_from_replayed_history() -> None:
    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return "contents"

    decisions, client = risk_decisions(0.1)
    [guarded] = OpenAIAutoMode(model=decisions).guard([read_file])
    schema = {
        "type": "object",
        "properties": {"type": {"type": "string"}, "limit": {"type": ["integer", "null"]}},
    }
    user_input: list[Any] = [
        {
            "type": "mcp_list_tools",
            "id": "list_1",
            "server_label": "docs",
            "tools": [{"name": "search", "input_schema": schema}],
        },
        {"role": "user", "content": "Read notes.txt"},
    ]

    result = await Runner.run(
        agent_calling(guarded, '{"path": "notes.txt"}'), user_input, run_config=CONFIG
    )

    assert result.final_output == "done"
    payload = json.loads(client.decisions.create.await_args.kwargs["input"])
    assert payload["messages"][0]["tools"][0]["input_schema"] == schema


async def test_classification_uses_the_namespaced_tool_name() -> None:
    calls: list[str] = []
    decisions, client = risk_decisions(0.9)
    [guarded] = OpenAIAutoMode(model=decisions).guard(
        tool_namespace(name="files", description="File tools.", tools=[delete_file_tool(calls)])
    )
    model = ScriptedModel(
        [
            [
                function_call(
                    "delete_file", '{"path": "a.txt"}', call_id="call_1", namespace="files"
                )
            ],
            [get_text_message("done")],
        ]
    )

    result = await Runner.run(
        Agent(name="assistant", model=model, tools=[guarded]), "hi", run_config=CONFIG
    )

    payload = json.loads(client.decisions.create.await_args.kwargs["input"])
    assert payload["tool_call"]["name"] == "files.delete_file"
    outputs = [item.output for item in result.new_items if isinstance(item, ToolCallOutputItem)]
    assert outputs[0].startswith("The tool call `files.delete_file` was blocked")


async def test_classification_sends_only_the_latest_run_input_items() -> None:
    decisions, client = risk_decisions(0.1)
    [guarded] = OpenAIAutoMode(model=decisions).guard([delete_file_tool([])])
    history: list[Any] = [{"role": "user", "content": f"message {index}"} for index in range(31)]

    await Runner.run(agent_calling(guarded, '{"path": "a.txt"}'), history, run_config=CONFIG)

    payload = json.loads(client.decisions.create.await_args.kwargs["input"])
    assert [item["content"] for item in payload["messages"]] == [
        f"message {index}" for index in range(1, 31)
    ]


async def test_require_approval_mode_classifies_with_the_namespaced_tool_name() -> None:
    decisions, client = risk_decisions(0.9)
    [guarded] = OpenAIAutoMode(model=decisions, on_risky="require_approval").guard(
        tool_namespace(name="files", description="File tools.", tools=[delete_file_tool([])])
    )
    model = ScriptedModel(
        [[function_call("delete_file", '{"path": "a.txt"}', call_id="call_1", namespace="files")]]
    )

    result = await Runner.run(
        Agent(name="assistant", model=model, tools=[guarded]), "hi", run_config=CONFIG
    )

    assert len(result.interruptions) == 1
    payload = json.loads(client.decisions.create.await_args.kwargs["input"])
    assert payload["tool_call"]["name"] == "files.delete_file"
