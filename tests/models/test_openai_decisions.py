from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from openai import omit
from openai.types import Decision

from agents.models import _openai_shared, openai_decisions
from agents.models.openai_decisions import Choice, Level, OpenAIDecisions, Predicate, Score

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
