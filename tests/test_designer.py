"""Tests for the QCMDesigner."""

from __future__ import annotations

from typing import Any

import pytest

from mcts_qcm.designer import QCMDesigner
from mcts_qcm.llm import LLMError


def _designer_responder(payload: dict[str, Any]) -> dict[str, Any]:
    """Fake LLM response returning a valid rubric structure."""
    return {
        "criteria": [
            {
                "key": "feasibility",
                "name": "Feasibility",
                "description": "Can the idea be executed?",
                "weight": 1.0,
                "sub_questions": [
                    {
                        "key": "feasibility_tech",
                        "question": "Is the technology available?",
                        "tier_anchors": {
                            "STRONG": "Off-the-shelf.",
                            "ADEQUATE": "Available with adaptation.",
                            "WEAK": "Requires R&D.",
                            "FAIL": "Impossible.",
                        },
                        "axiomatic": True,
                    },
                    {
                        "key": "feasibility_time",
                        "question": "Can it be done quickly?",
                        "tier_anchors": {
                            "STRONG": "Under 6 months.",
                            "ADEQUATE": "6-18 months.",
                            "WEAK": "2-5 years.",
                            "FAIL": "Over a decade.",
                        },
                        "axiomatic": False,
                    },
                ],
            },
            {
                "key": "cost",
                "name": "Cost",
                "description": "Is the cost reasonable?",
                "weight": 1.5,
                "sub_questions": [
                    {
                        "key": "cost_capital",
                        "question": "Is the capital expenditure reasonable?",
                        "tier_anchors": {
                            "STRONG": "Under $1000.",
                            "ADEQUATE": "$1000-$10000.",
                            "WEAK": "Expensive.",
                            "FAIL": "Prohibitively expensive.",
                        },
                        "axiomatic": False,
                    },
                ],
            },
        ]
    }


def test_designer_produces_valid_rubric(base_config, fake_client_factory) -> None:
    client = fake_client_factory(_designer_responder)
    designer = QCMDesigner(base_config, client=client)
    rubric = designer.propose("Design a low-cost desalination process.")

    assert len(rubric.criteria) == 2
    assert rubric.criteria[0].key == "feasibility"
    assert len(rubric.criteria[0].sub_questions) == 2
    assert rubric.criteria[1].weight == 1.5
    assert rubric.axiomatic_keys() == {"feasibility_tech"}


def test_designer_validates_minimum_criteria(base_config, fake_client_factory) -> None:
    """Designer rejects rubrics with fewer than 2 criteria."""
    def bad_responder(_p: dict) -> dict:
        return {
            "criteria": [
                {
                    "key": "only_one",
                    "name": "Only",
                    "description": "Just one criterion.",
                    "sub_questions": [{"key": "only_sq", "question": "Q?"}],
                }
            ]
        }

    client = fake_client_factory(bad_responder)
    designer = QCMDesigner(base_config, client=client)
    with pytest.raises(ValueError, match="at least 2 criteria"):
        designer.propose("Test problem")


def test_designer_validates_empty_sub_questions(base_config, fake_client_factory) -> None:
    """Designer rejects criteria with no sub-questions."""
    def bad_responder(_p: dict) -> dict:
        return {
            "criteria": [
                {"key": "a", "name": "A", "description": "d", "sub_questions": [{"key": "a_1", "question": "Q?"}]},
                {"key": "b", "name": "B", "description": "d", "sub_questions": []},
            ]
        }

    client = fake_client_factory(bad_responder)
    designer = QCMDesigner(base_config, client=client)
    with pytest.raises(ValueError, match="no sub-questions"):
        designer.propose("Test problem")


def test_designer_raises_on_llm_failure(base_config, fake_client_factory) -> None:
    """Designer raises LLMError when the LLM call itself fails."""
    def failing_responder(_p: dict) -> dict:
        raise LLMError("Connection refused")

    client = fake_client_factory(failing_responder)
    designer = QCMDesigner(base_config, client=client)
    with pytest.raises(LLMError):
        designer.propose("Test problem")
