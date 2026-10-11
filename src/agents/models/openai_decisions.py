"""OpenAI Decisions API support.

The Decisions API answers named questions about an input with probabilities instead of generated
text. See https://developers.openai.com/api/docs/guides/decisions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from openai import AsyncOpenAI, omit
from openai.types import DecisionInputMessageParam
from openai.types.decision import (
    Answer,
    AnswerAnswerResourceChoice,
    AnswerAnswerResourcePredicate,
    AnswerAnswerResourceRefusal,
    AnswerAnswerResourceScore,
    Usage,
)
from openai.types.decision_create_params import (
    Question,
    QuestionQuestionParamChoiceChoice,
    QuestionQuestionParamScoreLevel,
)

from ._openai_shared import get_default_openai_client, get_default_openai_key


@dataclass(frozen=True)
class Predicate:
    """A yes-or-no question. The answer is the probability that the condition is true."""

    instructions: str
    """The condition to evaluate against the input."""

    def _to_param(self, name: str) -> Question:
        return {"type": "predicate", "name": name, "instructions": self.instructions}


@dataclass(frozen=True)
class Choice:
    """A question whose answer is one of several values."""

    instructions: str
    """What to choose."""

    choices: Mapping[str | bool, str | None]
    """The allowed values, each mapped to an optional description."""

    def _to_param(self, name: str) -> Question:
        choices: list[QuestionQuestionParamChoiceChoice] = []
        for value, description in self.choices.items():
            choice: QuestionQuestionParamChoiceChoice = {"value": value}
            if description is not None:
                choice["description"] = description
            choices.append(choice)
        return {
            "type": "choice",
            "name": name,
            "instructions": self.instructions,
            "choices": choices,
        }


@dataclass(frozen=True)
class Level:
    """One level of a `Score` question."""

    label: str
    description: str | None = None


@dataclass(frozen=True)
class Score:
    """A question whose answer places the input on an ordered scale."""

    instructions: str
    """What to score."""

    levels: Sequence[str | Level]
    """The scale, from lowest to highest. A string is a level with only a label."""

    def _to_param(self, name: str) -> Question:
        levels: list[QuestionQuestionParamScoreLevel] = []
        for level in self.levels:
            if isinstance(level, str):
                level = Level(level)
            param: QuestionQuestionParamScoreLevel = {"label": level.label}
            if level.description is not None:
                param["description"] = level.description
            levels.append(param)
        return {"type": "score", "name": name, "instructions": self.instructions, "levels": levels}


@dataclass(frozen=True)
class DecisionResult:
    """The answers from one Decisions API request, keyed by question name."""

    answers: dict[str, Answer]
    """Every answer, in question order."""

    model: str
    """The model that answered."""

    usage: Usage
    """Token usage for the request."""

    request_id: str | None
    """The `x-request-id` of the API response."""

    @property
    def predicates(self) -> dict[str, AnswerAnswerResourcePredicate]:
        """Predicate answers by question name."""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, AnswerAnswerResourcePredicate)
        }

    @property
    def choices(self) -> dict[str, AnswerAnswerResourceChoice]:
        """Choice answers by question name."""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, AnswerAnswerResourceChoice)
        }

    @property
    def scores(self) -> dict[str, AnswerAnswerResourceScore]:
        """Score answers by question name."""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, AnswerAnswerResourceScore)
        }

    @property
    def refusals(self) -> dict[str, AnswerAnswerResourceRefusal]:
        """Questions the model declined to answer, by question name."""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, AnswerAnswerResourceRefusal)
        }


class OpenAIDecisions:
    """A client for the OpenAI Decisions API."""

    def __init__(self, model: str, *, openai_client: AsyncOpenAI | None = None) -> None:
        """Create a Decisions API client.

        Args:
            model: The decisions model to use, for example `gpt-6-luna`.
            openai_client: The client to use. Defaults to the client set with
                `set_default_openai_client`, else a new `AsyncOpenAI` that uses the key set with
                `set_default_openai_key` or the `OPENAI_API_KEY` environment variable.
        """
        self.model = model
        self._client = openai_client

    def _get_client(self) -> AsyncOpenAI:
        # The client is created on first use so that construction does not require an API key.
        if self._client is None:
            self._client = get_default_openai_client() or AsyncOpenAI(
                api_key=get_default_openai_key()
            )
        return self._client

    async def decide(
        self,
        input: str | list[DecisionInputMessageParam],
        questions: Mapping[str, Predicate | Choice | Score],
        *,
        safety_identifier: str | None = None,
    ) -> DecisionResult:
        """Answer each question about the input.

        Args:
            input: A string, or user messages with `input_text` and `input_image` parts.
            questions: The questions to ask, keyed by name. Answers use the same names.
            safety_identifier: An opaque identifier for the end user, used for abuse detection.

        Returns:
            The answers, keyed by question name.
        """
        decision = await self._get_client().decisions.create(
            model=self.model,
            input=input,
            questions=[question._to_param(name) for name, question in questions.items()],
            safety_identifier=omit if safety_identifier is None else safety_identifier,
        )
        return DecisionResult(
            answers={cast(str, answer.name): answer for answer in decision.answers},
            model=decision.model,
            usage=decision.usage,
            request_id=decision._request_id,
        )
