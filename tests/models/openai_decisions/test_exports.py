"""Verify the public names of the Decisions package and where they are defined."""

import agents.models.openai_decisions as decisions
from agents.models.openai_decisions import auto_mode, client, questions, result


def test_package_exports_the_public_names_from_their_submodules() -> None:
    expected = {
        "Choice": questions.Choice,
        "DEFAULT_AUTO_MODE_INSTRUCTIONS": auto_mode.DEFAULT_AUTO_MODE_INSTRUCTIONS,
        "DecisionResult": result.DecisionResult,
        "Level": questions.Level,
        "OpenAIAutoMode": auto_mode.OpenAIAutoMode,
        "OpenAIDecisions": client.OpenAIDecisions,
        "Predicate": questions.Predicate,
        "Score": questions.Score,
    }

    assert sorted(decisions.__all__) == sorted(expected)
    for name, value in expected.items():
        assert getattr(decisions, name) is value
