"""Rubric data model for the tiered QCM evaluation system.

A Rubric defines the complete evaluation framework for a problem:
- 4-6 top-level Criteria (e.g., Feasibility, Cost, Novelty)
- Each Criterion has 2-3 atomic SubQuestions
- Each SubQuestion is scored via categorical tier classification (STRONG/ADEQUATE/WEAK/FAIL)
- Tiers map deterministically to numerical values in Python
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Deterministic tier-to-value mapping. The LLM does categorical classification;
# Python resolves the score.
TIER_VALUES: dict[str, float] = {
    "STRONG": 1.0,
    "ADEQUATE": 0.66,
    "WEAK": 0.33,
    "FAIL": 0.0,
}

TIER_NAMES: list[str] = list(TIER_VALUES.keys())  # ordered best → worst


@dataclass
class SubQuestion:
    """A single atomic evaluation question under a criterion.

    Attributes:
        key: Machine-readable identifier (e.g. 'feasibility_materials').
        question: Human-readable question text.
        tier_anchors: Maps each tier name to a one-sentence description of what
            that tier means for this specific sub-question. This anchors the
            LLM's classification.
        axiomatic: If True, a FAIL on this sub-question triggers immediate
            node pruning (hard constraint). Use for non-negotiable requirements.
    """

    key: str
    question: str
    tier_anchors: dict[str, str] = field(default_factory=dict)
    axiomatic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "question": self.question,
            "tier_anchors": self.tier_anchors,
            "axiomatic": self.axiomatic,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubQuestion:
        return cls(
            key=data["key"],
            question=data["question"],
            tier_anchors=data.get("tier_anchors", {}),
            axiomatic=data.get("axiomatic", False),
        )


@dataclass
class Criterion:
    """A top-level evaluation criterion with its sub-questions.

    Attributes:
        key: Machine-readable identifier (e.g. 'feasibility').
        name: Human-readable display name (e.g. 'Feasibility').
        description: One-line summary of what this criterion evaluates.
        weight: Relative importance weight (default 1.0). Used in weighted
            score computation.
        sub_questions: List of 2-3 atomic SubQuestions under this criterion.
    """

    key: str
    name: str
    description: str
    weight: float = 1.0
    sub_questions: list[SubQuestion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
            "sub_questions": [sq.to_dict() for sq in self.sub_questions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Criterion:
        return cls(
            key=data["key"],
            name=data["name"],
            description=data["description"],
            weight=data.get("weight", 1.0),
            sub_questions=[SubQuestion.from_dict(sq) for sq in data.get("sub_questions", [])],
        )


@dataclass
class Rubric:
    """The complete evaluation rubric for a problem.

    Defines all criteria, their sub-questions, tier anchors, axiomatic flags,
    and the tier-to-value mapping used for deterministic scoring.
    """

    criteria: list[Criterion] = field(default_factory=list)
    tier_values: dict[str, float] = field(default_factory=lambda: dict(TIER_VALUES))

    def all_sub_questions(self) -> list[SubQuestion]:
        """Flat list of all sub-questions across all criteria."""
        return [sq for c in self.criteria for sq in c.sub_questions]

    def axiomatic_keys(self) -> set[str]:
        """Keys of sub-questions that trigger hard pruning on FAIL."""
        return {sq.key for sq in self.all_sub_questions() if sq.axiomatic}

    def sub_question_count(self) -> int:
        """Total number of atomic sub-questions."""
        return sum(len(c.sub_questions) for c in self.criteria)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation for tree.json and .mcts_rubric.json."""
        return {
            "criteria": [c.to_dict() for c in self.criteria],
            "tier_values": self.tier_values,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rubric:
        """Reconstruct from JSON."""
        return cls(
            criteria=[Criterion.from_dict(c) for c in data.get("criteria", [])],
            tier_values=data.get("tier_values", dict(TIER_VALUES)),
        )
