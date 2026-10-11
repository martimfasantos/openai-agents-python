"""Live check of the Decisions API contract. Set OPENAI_RUN_LIVE_DECISIONS_TESTS=1 to run it."""

from __future__ import annotations

import os

import pytest

from agents.models.openai_decisions import Choice, OpenAIDecisions, Predicate, Score

pytestmark = pytest.mark.skipif(
    os.environ.get("OPENAI_RUN_LIVE_DECISIONS_TESTS") != "1",
    reason="Set OPENAI_RUN_LIVE_DECISIONS_TESTS=1 to run live Decisions API tests.",
)


async def test_live_decide_answers_every_question_type() -> None:
    result = await OpenAIDecisions("gpt-6-luna").decide(
        "The checkout page returns 500 errors for every customer.",
        {
            "urgent": Predicate("Does this need attention right now?"),
            "team": Choice(
                "Which team should own this?",
                {"infra": "Servers, deploys, and uptime.", "billing": "Payments and invoices."},
            ),
            "severity": Score("How severe is the impact?", ["Cosmetic", "Degraded", "Down"]),
        },
    )

    assert list(result.answers) == ["urgent", "team", "severity"]
    assert result.request_id
    assert result.usage.input_tokens > 0
