"""Fake Decisions API responses and clients shared by the Decisions tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from openai.types import Decision

USAGE = {
    "input_tokens": 42,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": 0,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 42,
}


def make_decision(*answers: dict[str, Any]) -> Decision:
    decision = Decision.model_validate(
        {"model": "gpt-6-luna", "answers": list(answers), "usage": USAGE}
    )
    decision._request_id = "req_123"
    return decision


def make_client(*results: Decision | Exception) -> Any:
    client = MagicMock()
    client.decisions.create = AsyncMock(side_effect=list(results))
    return client
