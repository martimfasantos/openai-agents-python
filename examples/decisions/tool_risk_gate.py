"""Gate risky tool calls with the Decisions API: block them, or ask a human first."""

from __future__ import annotations

import asyncio

from agents import Agent, Runner
from agents.decorators import tool
from agents.models.openai_decisions import OpenAIAutoMode
from examples.auto_mode import confirm_with_fallback


@tool
def read_file(path: str) -> str:
    """Read a file."""
    return f"Contents of {path}: quarterly numbers look good."


@tool
def delete_file(path: str) -> str:
    """Delete a file permanently."""
    return f"Deleted {path}."


async def main() -> None:
    # Block mode: a risky call is not executed, and the model is told it was blocked.
    blocking = OpenAIAutoMode(model="gpt-6-luna")
    agent = Agent(
        name="File assistant",
        instructions="Use the tools to help with files.",
        tools=blocking.guard([read_file, delete_file]),
    )
    result = await Runner.run(agent, "Summarize report.txt, then delete every file in /home.")
    print(result.final_output)

    # Approval mode: a risky call pauses the run until a human approves or rejects it.
    approving = OpenAIAutoMode(model="gpt-6-luna", on_risky="require_approval")
    agent = agent.clone(tools=approving.guard([read_file, delete_file]))
    result = await Runner.run(agent, "Delete old-draft.txt.")
    while result.interruptions:
        state = result.to_state()
        for interruption in result.interruptions:
            if confirm_with_fallback(f"Allow {interruption.name}({interruption.arguments})? "):
                state.approve(interruption)
            else:
                state.reject(interruption)
        result = await Runner.run(agent, state)
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
