"""The result of an OpenAI Decisions API request."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from openai.types.decision import (
    Answer,
    AnswerAnswerResourceChoice,
    AnswerAnswerResourcePredicate,
    AnswerAnswerResourceRefusal,
    AnswerAnswerResourceScore,
    Usage,
)

_AnswerT = TypeVar("_AnswerT")


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

    def _answers_of(self, kind: type[_AnswerT]) -> dict[str, _AnswerT]:
        return {name: answer for name, answer in self.answers.items() if isinstance(answer, kind)}

    @property
    def predicates(self) -> dict[str, AnswerAnswerResourcePredicate]:
        """Predicate answers by question name."""
        return self._answers_of(AnswerAnswerResourcePredicate)

    @property
    def choices(self) -> dict[str, AnswerAnswerResourceChoice]:
        """Choice answers by question name."""
        return self._answers_of(AnswerAnswerResourceChoice)

    @property
    def scores(self) -> dict[str, AnswerAnswerResourceScore]:
        """Score answers by question name."""
        return self._answers_of(AnswerAnswerResourceScore)

    @property
    def refusals(self) -> dict[str, AnswerAnswerResourceRefusal]:
        """Questions the model declined to answer, by question name."""
        return self._answers_of(AnswerAnswerResourceRefusal)
