"""Tests for the Rubric data model."""

from __future__ import annotations

from mcts_qcm.rubric import Criterion, Rubric, SubQuestion, TIER_VALUES


def test_sub_question_roundtrip() -> None:
    sq = SubQuestion(
        key="test_key",
        question="Is it good?",
        tier_anchors={"STRONG": "Yes.", "FAIL": "No."},
        axiomatic=True,
    )
    restored = SubQuestion.from_dict(sq.to_dict())
    assert restored.key == sq.key
    assert restored.question == sq.question
    assert restored.tier_anchors == sq.tier_anchors
    assert restored.axiomatic is True


def test_criterion_roundtrip() -> None:
    sq = SubQuestion(key="c_a", question="Q?", tier_anchors={})
    c = Criterion(key="c", name="C", description="Desc", weight=2.0, sub_questions=[sq])
    restored = Criterion.from_dict(c.to_dict())
    assert restored.key == "c"
    assert restored.weight == 2.0
    assert len(restored.sub_questions) == 1
    assert restored.sub_questions[0].key == "c_a"


def test_rubric_roundtrip() -> None:
    rubric = Rubric(
        criteria=[
            Criterion(
                key="x",
                name="X",
                description="Desc X",
                weight=1.0,
                sub_questions=[
                    SubQuestion(key="x_1", question="Q1?", axiomatic=True),
                    SubQuestion(key="x_2", question="Q2?"),
                ],
            ),
            Criterion(
                key="y",
                name="Y",
                description="Desc Y",
                weight=1.5,
                sub_questions=[
                    SubQuestion(key="y_1", question="Q3?"),
                ],
            ),
        ]
    )
    data = rubric.to_dict()
    restored = Rubric.from_dict(data)
    assert len(restored.criteria) == 2
    assert restored.criteria[0].key == "x"
    assert restored.criteria[1].weight == 1.5
    assert restored.tier_values == TIER_VALUES


def test_all_sub_questions_flattens(sample_rubric) -> None:
    all_sqs = sample_rubric.all_sub_questions()
    assert len(all_sqs) == 8
    keys = [sq.key for sq in all_sqs]
    assert "feasibility_tech" in keys
    assert "cost_capital" in keys
    assert "alignment_user" in keys


def test_axiomatic_keys(sample_rubric) -> None:
    axio = sample_rubric.axiomatic_keys()
    assert axio == {"feasibility_tech", "alignment_direct"}


def test_sub_question_count(sample_rubric) -> None:
    assert sample_rubric.sub_question_count() == 8


def test_rubric_from_dict_with_custom_tier_values() -> None:
    data = {
        "criteria": [
            {
                "key": "a",
                "name": "A",
                "description": "Test",
                "sub_questions": [{"key": "a_1", "question": "?"}],
            }
        ],
        "tier_values": {"STRONG": 1.0, "ADEQUATE": 0.5, "WEAK": 0.25, "FAIL": 0.0},
    }
    rubric = Rubric.from_dict(data)
    assert rubric.tier_values["ADEQUATE"] == 0.5
