from __future__ import annotations

import asyncio
import json
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import omit
from openai.types import Decision

from agents import Agent, ModelSettings, RunConfig, Runner, WebSearchTool
from agents.decorators import tool
from agents.exceptions import UserError
from agents.items import ToolCallOutputItem
from agents.models import _openai_shared, openai_decisions
from agents.models.default_models import get_default_model_settings
from agents.models.interface import Model, ModelProvider, ModelTracing
from agents.models.openai_decisions import (
    DEFAULT_AUTO_MODE_INSTRUCTIONS,
    Choice,
    Level,
    ModelChoice,
    OpenAIAutoMode,
    OpenAIDecisions,
    OpenAIModelRouter,
    Predicate,
    Score,
)
from agents.retry import (
    ModelRetryAdvice,
    ModelRetryBackoffSettings,
    ModelRetrySettings,
    retry_policies,
)
from agents.testing import ModelStep, ScriptedModel
from agents.tool import FunctionTool
from tests.test_responses import get_function_tool_call, get_text_message

USAGE = {
    "input_tokens": 42,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": 0,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 42,
}


def make_decision(*answers: dict[str, Any], request_id: str | None = "req_123") -> Decision:
    decision = Decision.model_validate(
        {"model": "gpt-6-luna", "answers": list(answers), "usage": USAGE}
    )
    decision._request_id = request_id
    return decision


def make_client(*decisions: Decision) -> Any:
    client = MagicMock()
    client.decisions.create = AsyncMock(side_effect=list(decisions))
    return client


async def test_decide_sends_questions_in_order_and_parses_every_answer_type() -> None:
    client = make_client(
        make_decision(
            {"type": "predicate", "name": "urgent", "probability": 0.91},
            {
                "type": "choice",
                "name": "team",
                "choice": "infra",
                "confidence": 0.8,
                "probabilities": [
                    {"value": "infra", "probability": 0.85},
                    {"value": "billing", "probability": 0.15},
                ],
            },
            {
                "type": "score",
                "name": "severity",
                "score": 1.2,
                "confidence": 0.7,
                "probabilities": [
                    {"value": 0, "label": "Cosmetic", "probability": 0.1},
                    {"value": 1, "label": "Degraded", "probability": 0.6},
                    {"value": 2, "label": "Down", "probability": 0.3},
                ],
            },
            {"type": "refusal", "name": "blame"},
        )
    )

    result = await OpenAIDecisions("gpt-6-luna", openai_client=client).decide(
        "The deploy failed twice and customers see 500 errors.",
        {
            "urgent": Predicate("Does this need attention now?"),
            "team": Choice(
                "Which team owns this?", {"infra": "Deploys and uptime.", "billing": None}
            ),
            "severity": Score(
                "How severe is it?",
                ["Cosmetic", Level("Degraded", "Slower but working."), "Down"],
            ),
            "blame": Predicate("Which person caused this?"),
        },
    )

    client.decisions.create.assert_awaited_once_with(
        model="gpt-6-luna",
        input="The deploy failed twice and customers see 500 errors.",
        questions=[
            {
                "type": "predicate",
                "name": "urgent",
                "instructions": "Does this need attention now?",
            },
            {
                "type": "choice",
                "name": "team",
                "instructions": "Which team owns this?",
                "choices": [
                    {"value": "infra", "description": "Deploys and uptime."},
                    {"value": "billing"},
                ],
            },
            {
                "type": "score",
                "name": "severity",
                "instructions": "How severe is it?",
                "levels": [
                    {"label": "Cosmetic"},
                    {"label": "Degraded", "description": "Slower but working."},
                    {"label": "Down"},
                ],
            },
            {"type": "predicate", "name": "blame", "instructions": "Which person caused this?"},
        ],
        safety_identifier=omit,
    )
    assert list(result.answers) == ["urgent", "team", "severity", "blame"]
    assert result.predicates["urgent"].probability == 0.91
    assert result.choices["team"].choice == "infra"
    assert result.scores["severity"].score == 1.2
    assert list(result.refusals) == ["blame"]
    assert result.model == "gpt-6-luna"
    assert result.usage.input_tokens == 42
    assert result.request_id == "req_123"


async def test_decide_keeps_bool_choice_values_and_sends_optional_fields() -> None:
    client = make_client(
        make_decision(
            {
                "type": "choice",
                "name": "approve",
                "choice": False,
                "confidence": 0.9,
                "probabilities": [
                    {"value": True, "probability": 0.05},
                    {"value": False, "probability": 0.95},
                ],
            }
        )
    )

    result = await OpenAIDecisions("gpt-6-luna", openai_client=client).decide(
        [{"role": "user", "content": "Refund $10,000 without a receipt."}],
        {"approve": Choice("Should this refund be approved?", {True: None, False: None})},
        safety_identifier="user-1",
    )

    kwargs = client.decisions.create.await_args.kwargs
    assert kwargs["input"] == [{"role": "user", "content": "Refund $10,000 without a receipt."}]
    assert kwargs["questions"][0]["choices"] == [{"value": True}, {"value": False}]
    assert kwargs["safety_identifier"] == "user-1"
    assert result.choices["approve"].choice is False


async def test_default_client_is_used_when_none_is_injected() -> None:
    client = make_client(make_decision({"type": "predicate", "name": "q", "probability": 0.5}))
    _openai_shared.set_default_openai_client(client)

    await OpenAIDecisions("gpt-6-luna").decide("text", {"q": Predicate("Is it true?")})

    client.decisions.create.assert_awaited_once()


async def test_default_key_builds_a_client_lazily(monkeypatch) -> None:
    created: list[dict[str, Any]] = []
    client = make_client(make_decision({"type": "predicate", "name": "q", "probability": 0.5}))

    def fake_async_openai(**kwargs: Any) -> Any:
        created.append(kwargs)
        return client

    monkeypatch.setattr(openai_decisions, "AsyncOpenAI", fake_async_openai)
    _openai_shared.set_default_openai_key("sk-test-key")

    decisions = OpenAIDecisions("gpt-6-luna")
    assert created == []

    await decisions.decide("text", {"q": Predicate("Is it true?")})
    assert created == [{"api_key": "sk-test-key"}]


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
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Email the report to Ana."},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AAAA",
                    "detail": "auto",
                },
            ],
        }
    ]

    await Runner.run(
        agent_calling(guarded, '{"to": "ana@example.com"}'), user_input, run_config=CONFIG
    )

    kwargs = client.decisions.create.await_args.kwargs
    assert kwargs["questions"] == [
        {"type": "predicate", "name": "is_risky", "instructions": DEFAULT_AUTO_MODE_INSTRUCTIONS}
    ]
    assert "base64" not in kwargs["input"]
    payload = json.loads(kwargs["input"])
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


async def test_classification_errors_fail_the_run_without_running_the_tool() -> None:
    calls: list[str] = []
    client = MagicMock()
    client.decisions.create = AsyncMock(side_effect=RuntimeError("decisions unavailable"))
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


def route_decisions(*routes: str | None) -> tuple[OpenAIDecisions, Any]:
    client = make_client(
        *[
            make_decision(
                {"type": "refusal", "name": "model_route"}
                if route is None
                else {
                    "type": "choice",
                    "name": "model_route",
                    "choice": route,
                    "confidence": 0.9,
                    "probabilities": [{"value": route, "probability": 0.9}],
                }
            )
            for route in routes
        ]
    )
    return OpenAIDecisions("gpt-6-luna", openai_client=client), client


class CleanupTrackingModel(ScriptedModel):
    """Records the run owners whose cleanup the router forwards."""

    def __init__(self, steps: list[Any]) -> None:
        super().__init__(steps)
        self.cleaned_owners: list[object] = []

    async def _cleanup_on_run_end(self, owner: object) -> None:
        self.cleaned_owners.append(owner)


def make_router(fast: Model, powerful: Model, decisions: OpenAIDecisions) -> OpenAIModelRouter:
    return OpenAIModelRouter(
        choices={
            "fast": ModelChoice(fast, "Simple questions."),
            "powerful": ModelChoice(powerful, "Hard reasoning."),
        },
        instructions="Which model should answer?",
        default="fast",
        model=decisions,
    )


@pytest.mark.parametrize("streamed", [False, True])
async def test_router_classifies_once_per_run_and_uses_the_chosen_model(streamed: bool) -> None:
    @tool
    def lookup(term: str) -> str:
        """Look up a term."""
        return "found"

    fast = ScriptedModel()
    powerful = CleanupTrackingModel(
        [
            [get_function_tool_call("lookup", '{"term": "BFT"}', call_id="call_1")],
            [get_text_message("proof")],
        ]
    )
    decisions, client = route_decisions("powerful")
    router = make_router(fast, powerful, decisions)
    agent = Agent(name="assistant", model=router, tools=[lookup])

    if streamed:
        result = Runner.run_streamed(agent, "Prove BFT consensus safety.", run_config=CONFIG)
        async for _ in result.stream_events():
            pass
    else:
        result = await Runner.run(agent, "Prove BFT consensus safety.", run_config=CONFIG)

    assert result.final_output == "proof"
    assert len(powerful.calls) == 2
    assert fast.calls == ()
    client.decisions.create.assert_awaited_once()
    kwargs = client.decisions.create.await_args.kwargs
    assert kwargs["input"] == [{"role": "user", "content": "Prove BFT consensus safety."}]
    assert kwargs["questions"] == [
        {
            "type": "choice",
            "name": "model_route",
            "instructions": "Which model should answer?",
            "choices": [
                {"value": "fast", "description": "Simple questions."},
                {"value": "powerful", "description": "Hard reasoning."},
            ],
        }
    ]
    # The run-end cleanup must release the per-run route and reach the routed model.
    assert router._routes == {}
    assert len(powerful.cleaned_owners) == 1


async def test_router_outside_a_run_classifies_each_call_and_defaults_without_user_text() -> None:
    fast = ScriptedModel(
        [[get_text_message("one")], [get_text_message("two")], [get_text_message("three")]]
    )
    decisions, client = route_decisions("fast", "fast")
    router = make_router(fast, ScriptedModel(), decisions)
    user_turn: list[Any] = [{"role": "user", "content": "hi"}]
    tool_turn: list[Any] = [{"type": "function_call_output", "call_id": "call_1", "output": "ok"}]

    for turn in (user_turn, user_turn, tool_turn):
        await router.get_response(
            None,
            turn,
            ModelSettings(),
            [],
            None,
            [],
            ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )

    assert len(fast.calls) == 3
    assert client.decisions.create.await_count == 2
    assert router._routes == {}


async def test_router_uses_default_when_the_route_is_refused() -> None:
    fast = ScriptedModel([[get_text_message("quick answer")]])
    decisions, _ = route_decisions(None)

    result = await Runner.run(
        Agent(name="assistant", model=make_router(fast, ScriptedModel(), decisions)),
        "What is 2 + 2?",
        run_config=CONFIG,
    )

    assert result.final_output == "quick answer"


async def test_router_keeps_routes_separate_for_concurrent_runs() -> None:
    both_runs_routed = asyncio.Event()
    waiting = 0

    @tool
    async def wait_for_other_run() -> str:
        """Wait until the other run reaches its tool call."""
        nonlocal waiting
        waiting += 1
        if waiting == 2:
            both_runs_routed.set()
        await both_runs_routed.wait()
        return "ok"

    def two_turn_model(final_text: str) -> ScriptedModel:
        return ScriptedModel(
            [
                [get_function_tool_call("wait_for_other_run", "{}", call_id="call_1")],
                [get_text_message(final_text)],
            ]
        )

    fast = two_turn_model("fast done")
    powerful = two_turn_model("powerful done")
    decisions, client = route_decisions("fast", "powerful")
    agent = Agent(
        name="assistant", model=make_router(fast, powerful, decisions), tools=[wait_for_other_run]
    )

    results = await asyncio.gather(
        Runner.run(agent, "first", run_config=CONFIG),
        Runner.run(agent, "second", run_config=CONFIG),
    )

    assert sorted(result.final_output for result in results) == ["fast done", "powerful done"]
    assert len(fast.calls) == 2
    assert len(powerful.calls) == 2
    assert client.decisions.create.await_count == 2


async def test_router_layers_choice_settings_over_the_implicit_defaults() -> None:
    fast = ScriptedModel([[get_text_message("ok")]])
    decisions, _ = route_decisions("fast")
    router = OpenAIModelRouter(
        choices={
            "fast": ModelChoice(fast, "Simple.", model_settings=ModelSettings(temperature=0.1)),
            "powerful": ModelChoice(ScriptedModel(), "Hard."),
        },
        instructions="Which model should answer?",
        default="fast",
        model=decisions,
    )

    await Runner.run(Agent(name="assistant", model=router), "hi", run_config=CONFIG)

    assert fast.calls[0].model_settings == ModelSettings(temperature=0.1)


async def test_router_resolves_model_names_with_their_own_default_settings(monkeypatch) -> None:
    resolved = ScriptedModel([[get_text_message("ok")]])
    names: list[str | None] = []

    class RecordingProvider(ModelProvider):
        def get_model(self, model_name: str | None) -> Model:
            names.append(model_name)
            return resolved

    monkeypatch.setattr(openai_decisions, "MultiProvider", RecordingProvider)
    decisions, _ = route_decisions("fast")
    router = OpenAIModelRouter(
        choices={
            "fast": ModelChoice("gpt-4.1", "Simple."),
            "powerful": ModelChoice("gpt-5.5", "Hard."),
        },
        instructions="Which model should answer?",
        default="fast",
        model=decisions,
    )
    # The agent's default settings target the GPT-5 default model, so they must not reach gpt-4.1.
    assert get_default_model_settings() != ModelSettings()

    await Runner.run(Agent(name="assistant", model=router), "hi", run_config=CONFIG)

    assert names == ["gpt-4.1"]
    assert resolved.calls[0].model_settings == ModelSettings()


async def test_router_forwards_retry_advice_to_the_routed_model() -> None:
    fast = ScriptedModel(
        [
            ModelStep.raise_error(
                RuntimeError("temporary failure"),
                retry_advice=ModelRetryAdvice(suggested=True, replay_safety="safe"),
            ),
            [get_text_message("recovered")],
        ]
    )
    decisions, client = route_decisions("fast")
    agent = Agent(
        name="assistant",
        model=make_router(fast, ScriptedModel(), decisions),
        model_settings=ModelSettings(
            retry=ModelRetrySettings(
                max_retries=1,
                backoff=ModelRetryBackoffSettings(initial_delay=0, jitter=False),
                policy=retry_policies.provider_suggested(),
            )
        ),
    )

    result = await Runner.run(agent, "hi", run_config=CONFIG)

    assert result.final_output == "recovered"
    client.decisions.create.assert_awaited_once()


async def test_router_sends_only_supported_parts_of_the_latest_user_message() -> None:
    fast = ScriptedModel([[get_text_message("ok")]])
    decisions, client = route_decisions("fast")
    user_input: list[Any] = [
        {"role": "user", "content": "An older question."},
        {"role": "assistant", "content": "An older answer."},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Which plan fits this chart?"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AAAA",
                    "detail": "high",
                    "file_id": None,
                },
                {"type": "input_file", "file_id": "file-123"},
            ],
        },
    ]

    await Runner.run(
        Agent(name="assistant", model=make_router(fast, ScriptedModel(), decisions)),
        user_input,
        run_config=CONFIG,
    )

    assert client.decisions.create.await_args.kwargs["input"] == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Which plan fits this chart?"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AAAA",
                    "detail": "high",
                },
                {"type": "input_text", "text": "[input_file omitted]"},
            ],
        }
    ]


def test_router_rejects_invalid_choices() -> None:
    only = {"fast": ModelChoice(ScriptedModel(), "Simple.")}
    two = {**only, "powerful": ModelChoice(ScriptedModel(), "Hard.")}

    with pytest.raises(UserError, match="at least two"):
        OpenAIModelRouter(choices=only, instructions="Pick.", default="fast", model="gpt-6-luna")
    with pytest.raises(UserError, match="default"):
        OpenAIModelRouter(choices=two, instructions="Pick.", default="other", model="gpt-6-luna")
