"""Route each run to a fast or a powerful model with the Decisions API."""

from __future__ import annotations

import asyncio

from agents import Agent, Runner
from agents.models.openai_decisions import (
    Choice,
    ModelChoice,
    OpenAIDecisions,
    OpenAIModelRouter,
)

CHOICES = {
    "fast": ModelChoice("gpt-5.6-luna", "Short factual questions and simple tasks."),
    "powerful": ModelChoice("gpt-5.6-sol", "Multi-step reasoning, proofs, and complex analysis."),
}


async def main() -> None:
    # Option 1: the router is the agent's model and picks a model once per run.
    router = OpenAIModelRouter(
        choices=CHOICES,
        instructions="Which model should answer this request?",
        default="fast",
        model="gpt-6-luna",
    )
    agent = Agent(name="Assistant", instructions="Answer concisely.", model=router)
    for question in ["What is the capital of France?", "Prove that sqrt(2) is irrational."]:
        result = await Runner.run(agent, question)
        print(f"{question}\n{result.final_output}\n")

    # Option 2: classify before the run and clone the agent with the chosen model.
    decisions = OpenAIDecisions("gpt-6-luna")
    question = "Compare three consensus algorithms for a 5-node cluster."
    route = await decisions.decide(
        question,
        {
            "model": Choice(
                "Which model should answer this request?",
                {name: choice.criteria for name, choice in CHOICES.items()},
            )
        },
    )
    answer = route.choices.get("model")
    name = str(answer.choice) if answer is not None else "fast"
    result = await Runner.run(agent.clone(model=CHOICES[name].model), question)
    print(f"Routed to {name}:\n{result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
