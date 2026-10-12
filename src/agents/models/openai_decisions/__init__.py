"""OpenAI Decisions API support.

The Decisions API answers named questions about an input with probabilities instead of generated
text. See https://developers.openai.com/api/docs/guides/decisions.
"""

from .auto_mode import DEFAULT_AUTO_MODE_INSTRUCTIONS, OpenAIAutoMode
from .client import OpenAIDecisions
from .questions import Choice, Level, Predicate, Score
from .result import DecisionResult

__all__ = [
    "Choice",
    "DecisionResult",
    "DEFAULT_AUTO_MODE_INSTRUCTIONS",
    "Level",
    "OpenAIAutoMode",
    "OpenAIDecisions",
    "Predicate",
    "Score",
]
