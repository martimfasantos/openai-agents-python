# Decisions API

The [OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions) answers named questions about an input with typed answers and probabilities instead of generated text. It suits the small judgment calls that surround an agent run: whether a request is in scope, which model should handle a request, or whether a tool call is safe to run. The Decisions API is in public beta, and `gpt-6-luna` is the supported model.

The Agents SDK provides the following Decisions API support in `agents.models.openai_decisions`:

-   [`OpenAIDecisions`][agents.models.openai_decisions.OpenAIDecisions], a client that asks questions and returns typed answers.
-   [`OpenAIAutoMode`][agents.models.openai_decisions.OpenAIAutoMode], a tool risk gate that classifies each call to a guarded function tool before the tool runs.

The Decisions API requires `openai>=3.26.0`, which the Agents SDK installs.

## Ask questions

An `OpenAIDecisions` client sends one request with one or more questions about the same input. Each question has a name, and the answers use the same names.

```python
from agents.models.openai_decisions import Choice, Level, OpenAIDecisions, Predicate, Score

decisions = OpenAIDecisions("gpt-6-luna")

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
    print(urgent.probability)
if team := result.choices.get("team"):
    print(team.choice, team.confidence)
if severity := result.scores.get("severity"):
    print(severity.score)
```

The three question types are:

| Question type | Answer |
|---|---|
| [`Predicate`][agents.models.openai_decisions.Predicate] | The probability that the condition is true. |
| [`Choice`][agents.models.openai_decisions.Choice] | One of the allowed values, a probability for each value, and a confidence. Values can be strings or booleans, and each value can have an optional description. |
| [`Score`][agents.models.openai_decisions.Score] | A position on an ordered scale of two to ten levels, from lowest to highest. The `score` is the probability-weighted level index, so the `score` can fall between two levels. A [`Level`][agents.models.openai_decisions.Level] adds an optional description to a level label. |

`decide()` returns a [`DecisionResult`][agents.models.openai_decisions.DecisionResult]. The `answers` dictionary holds every answer by question name, and the `predicates`, `choices`, `scores`, and `refusals` dictionaries hold the answers of each type. The Decisions API can decline to answer one question while it still answers the other questions, so check `refusals` or use `.get()` when you read an answer. `DecisionResult` also includes the `model`, the token `usage`, and the `request_id` of the API response.

The input can be a string, or a list of user messages with `input_text` and `input_image` parts. To classify a conversation that contains other item types, serialize the conversation to a string first, for example with `json.dumps(result.to_input_list())`. Pass `safety_identifier` to `decide()` to forward an end-user identifier to the Decisions API for abuse detection.

### Client configuration

`OpenAIDecisions` uses the following client, in this order:

1.  The `openai_client` argument, if you pass one.
2.  The client set with [`set_default_openai_client`][agents.set_default_openai_client].
3.  A new `AsyncOpenAI` client that uses the key set with [`set_default_openai_key`][agents.set_default_openai_key] or the `OPENAI_API_KEY` environment variable.

`OpenAIDecisions` creates the client on the first `decide()` call, so constructing `OpenAIDecisions` does not require an API key. Configure the base URL, timeouts, retries, and headers on the `AsyncOpenAI` client. The Decisions API enforces its own limits, such as the number of questions and choices, and returns an API error when a request exceeds a limit.

## Use a predicate as an input guardrail

A predicate can back an [input guardrail](guardrails.md) that blocks requests before the agent runs:

```python
import json

from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    TResponseInputItem,
)
from agents.decorators import input_guardrail
from agents.models.openai_decisions import OpenAIDecisions, Predicate

decisions = OpenAIDecisions("gpt-6-luna")


@input_guardrail
async def off_topic_guardrail(
    context: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    text = input if isinstance(input, str) else json.dumps(input)
    result = await decisions.decide(
        text, {"off_topic": Predicate("Is this unrelated to support for our product?")}
    )
    answer = result.predicates.get("off_topic")
    # A refused question is treated as off topic.
    probability = 1.0 if answer is None else answer.probability
    return GuardrailFunctionOutput(output_info=probability, tripwire_triggered=probability >= 0.5)


agent = Agent(name="Support agent", input_guardrails=[off_topic_guardrail])
```

## Gate tool calls with auto mode

`OpenAIAutoMode.guard()` returns copies of function tools. Before each call to a guarded copy, `OpenAIAutoMode` asks the Decisions API whether the call is risky. The original tools are not modified.

```python
from agents import Agent
from agents.decorators import tool
from agents.models.openai_decisions import OpenAIAutoMode


@tool
def delete_file(path: str) -> str:
    """Delete a file permanently."""
    return f"Deleted {path}."


auto_mode = OpenAIAutoMode(model="gpt-6-luna")
agent = Agent(name="File assistant", tools=[*auto_mode.guard([delete_file])])
```

The `on_risky` argument controls what happens to a risky call:

| `on_risky` | Behavior for a risky call | Behavior for a safe call |
|---|---|---|
| `"reject_content"` (default) | The tool does not run. The model receives a message that says the call was blocked, and the run continues. | The tool runs. |
| `"require_approval"` | The run pauses with a tool approval interruption, as described in [Human-in-the-loop](human_in_the_loop.md). After a person approves the call and the run resumes, the tool runs without a second classification. | The tool runs without approval. |

In `"reject_content"` mode, `OpenAIAutoMode` adds a [tool input guardrail](guardrails.md#tool-guardrails) to each copy. In `"require_approval"` mode, `OpenAIAutoMode` sets `needs_approval` on each copy, so `guard()` raises a `UserError` for a tool that already sets `needs_approval`. `guard()` also raises a `UserError` for any tool that is not a `FunctionTool`.

A call is risky when the risk probability is at least `threshold` (default `0.5`), or when the Decisions API declines to answer. If the Decisions API request fails, the run stops before the tool runs. Tune `threshold` with labeled examples from your application, and pass `instructions` to replace [`DEFAULT_AUTO_MODE_INSTRUCTIONS`][agents.models.openai_decisions.DEFAULT_AUTO_MODE_INSTRUCTIONS] with your own risk policy.

For each guarded call, the Decisions API receives the following information:

-   The latest 30 items of the run input, with images, files, audio, screenshots, and generated images replaced by short placeholders.
-   The namespace-qualified tool name and the raw tool arguments.
-   The tool description.

The default instructions tell the classifier to treat all of this information as data, and to accept only explicit requests in user messages as authorization.

### Auto mode limitations

-   In an agent that runs as a tool through [`Agent.as_tool()`][agents.agent.Agent.as_tool], the run input was written by the calling agent, not by the user. Guard the outer agent tool instead of the inner agent's tools.
-   In `"require_approval"` mode, the SDK requires approval without calling `needs_approval` when the SDK must convert the tool arguments, for example for enums, dates, or omitted defaults. These calls always pause for approval without classification. Use `"reject_content"` mode to classify every call.
-   The classifier does not see authorization that a user gives only inside an attachment, or in input items older than the latest 30 items. The classifier treats these calls as unauthorized.
-   In `"require_approval"` mode, apply [`tool_namespace()`][agents.tool.tool_namespace] before `guard()`, so that the classifier receives the namespaced tool name.
-   Realtime sessions do not provide the conversation to tools, so the classifier sees only the tool call for realtime agent tools.

## Route a run to a model

To choose a model for each run, classify the request with a `Choice` before the run, and then run a copy of the agent with the chosen model:

```python
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


async def answer(question: str) -> str:
    result = await decisions.decide(question, {"route": ROUTE})
    route = result.choices.get("route")
    name = str(route.choice) if route is not None else "fast"
    run = await Runner.run(agent.clone(model=MODELS[name]), question)
    return str(run.final_output)
```

`agent.clone()` resolves the chosen model name with the model provider of the run, and applies the default model settings of the chosen model.

## Examples

The [`examples/decisions`](https://github.com/openai/openai-agents-python/tree/main/examples/decisions) directory contains runnable examples:

-   `classify.py` asks all three question types and uses a predicate as an input guardrail.
-   `tool_risk_gate.py` gates tool calls in both auto mode modes.
-   `model_router.py` routes each run to a fast model or a powerful model.

For the full API, see the [Decisions API reference](ref/models/openai_decisions.md).
