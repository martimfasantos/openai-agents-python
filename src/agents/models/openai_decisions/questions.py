"""Question types for the OpenAI Decisions API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from openai.types.decision_create_params import (
    Question,
    QuestionQuestionParamChoiceChoice,
    QuestionQuestionParamScoreLevel,
)


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
