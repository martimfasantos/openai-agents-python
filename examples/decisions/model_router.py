"""Route each run to a fast or a powerful model with the Decisions API."""

from __future__ import annotations

import asyncio

from agents import Agent, Runner
from agents.models.openai_decisions import Choice, OpenAIDecisions

MODELS = {"fast": "gpt-5.6-luna", "powerful": "gpt-5.6-sol"}
ROUTE = Choice(
    "Which model should answer this request?",
    {
        "fast": "Short factual questions and simple tasks.",
        "powerful": "Multi-step reasoning, proofs, and complex analysis.",
    },
)

decisions = OpenAIDecisions("gpt-6-luna")
agent = Agent(name="Assistant", instructions="Answer concisely.")


async def answer(question: str) -> None:
    # Classify once before the run, then run a copy of the agent on the chosen model.
    result = await decisions.decide(question, {"route": ROUTE})
    route = result.choices.get("route")
    name = str(route.choice) if route is not None else "fast"
    run = await Runner.run(agent.clone(model=MODELS[name]), question)
    print(f"[{name}] {question}\n{run.final_output}\n")


async def main() -> None:
    for question in ["What is the capital of France?", "Prove that sqrt(2) is irrational."]:
        await answer(question)


if __name__ == "__main__":
    asyncio.run(main())
