"""Ask the Decisions API typed questions, then use a predicate as an input guardrail."""

from __future__ import annotations

import asyncio
import json

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
)
from agents.decorators import input_guardrail
from agents.models.openai_decisions import Choice, Level, OpenAIDecisions, Predicate, Score

decisions = OpenAIDecisions("gpt-6-luna")


@input_guardrail
async def off_topic_guardrail(
    context: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    text = input if isinstance(input, str) else json.dumps(input)
    result = await decisions.decide(
        text,
        {"off_topic": Predicate("Is this unrelated to support for our software product?")},
    )
    answer = result.predicates.get("off_topic")
    # A refused question is treated as off topic.
    probability = 1.0 if answer is None else answer.probability
    return GuardrailFunctionOutput(output_info=probability, tripwire_triggered=probability >= 0.5)


async def main() -> None:
    result = await decisions.decide(
        "The deploy failed twice and customers see 500 errors at checkout.",
        {
            "urgent": Predicate("Does this need attention right now?"),
            "team": Choice(
                "Which team should own this?",
                {"infra": "Servers, deploys, and uptime.", "billing": "Payments and invoices."},
            ),
            "severity": Score(
                "How severe is the impact?",
                ["Cosmetic", Level("Degraded", "Slower or partly broken."), "Down"],
            ),
        },
    )
    if urgent := result.predicates.get("urgent"):
        print(f"Urgent: {urgent.probability:.2f}")
    if team := result.choices.get("team"):
        print(f"Team: {team.choice} (confidence {team.confidence:.2f})")
    if severity := result.scores.get("severity"):
        print(f"Severity: {severity.score:.2f} on a 0-2 scale")
    if result.refusals:
        print(f"Refused: {', '.join(result.refusals)}")

    agent = Agent(
        name="Support agent",
        instructions="Help customers with our software product.",
        input_guardrails=[off_topic_guardrail],
    )
    try:
        await Runner.run(agent, "Write me a poem about the ocean.")
    except InputGuardrailTripwireTriggered:
        print("Blocked an off-topic request.")


if __name__ == "__main__":
    asyncio.run(main())
