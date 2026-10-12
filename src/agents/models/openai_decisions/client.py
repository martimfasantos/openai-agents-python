"""A client for the OpenAI Decisions API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from openai import AsyncOpenAI, omit
from openai.types import DecisionInputMessageParam

from .._openai_shared import get_default_openai_client, get_default_openai_key
from ..openai_provider import shared_http_client
from .questions import Choice, Predicate, Score
from .result import DecisionResult


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
                api_key=get_default_openai_key(), http_client=shared_http_client()
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
