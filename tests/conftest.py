"""Shared test fixtures: a deterministic ``FakeLLMClient`` and sample rubric."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

import pytest

from mcts_qcm.config import MCTSConfig
from mcts_qcm.rubric import Criterion, Rubric, SubQuestion


@dataclass
class FakeLLMClient:
    """Deterministic LLM client for tests.

    `responder` is called with the prompt kwargs and must return the dict that
    ``LiteLLMClient.chat_json`` would have produced. A counter records every
    call so tests can assert how many times the engine hit the LLM.
    """

    responder: Callable[[dict[str, Any]], dict[str, Any]]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.7,
        seed: int | None = None,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "system": system,
            "user": user,
            "temperature": temperature,
            "seed": seed,
            "timeout": timeout,
        }
        self.calls.append(payload)
        return self.responder(payload)


@pytest.fixture
def base_config() -> MCTSConfig:
    return MCTSConfig(
        model_gen="fake/gen",
        model_audit="fake/audit",
        k_children=2,
        iterations=3,
        max_depth=3,
        max_nodes=50,
        prune_threshold=0.0,  # disable threshold pruning by default in tests
    )


@pytest.fixture
def sample_rubric() -> Rubric:
    """A simple 3-criterion, 8-sub-question rubric for tests."""
    return Rubric(
        criteria=[
            Criterion(
                key="feasibility",
                name="Feasibility",
                description="Can the idea be executed?",
                weight=1.0,
                sub_questions=[
                    SubQuestion(
                        key="feasibility_tech",
                        question="Is the technology available?",
                        tier_anchors={
                            "STRONG": "Off-the-shelf technology.",
                            "ADEQUATE": "Available with minor adaptation.",
                            "WEAK": "Requires significant R&D.",
                            "FAIL": "Violates known physical laws.",
                        },
                        axiomatic=True,
                    ),
                    SubQuestion(
                        key="feasibility_time",
                        question="Can it be done in a reasonable timeframe?",
                        tier_anchors={
                            "STRONG": "Under 6 months.",
                            "ADEQUATE": "6-18 months.",
                            "WEAK": "2-5 years.",
                            "FAIL": "Over a decade.",
                        },
                    ),
                    SubQuestion(
                        key="feasibility_skill",
                        question="Are the required skills available?",
                        tier_anchors={
                            "STRONG": "Standard engineering skills.",
                            "ADEQUATE": "Specialized but available.",
                            "WEAK": "Very rare expertise needed.",
                            "FAIL": "No known expertise exists.",
                        },
                    ),
                ],
            ),
            Criterion(
                key="cost",
                name="Cost",
                description="Is the cost reasonable?",
                weight=1.5,
                sub_questions=[
                    SubQuestion(
                        key="cost_capital",
                        question="Is the capital expenditure reasonable?",
                        tier_anchors={
                            "STRONG": "Under $1000.",
                            "ADEQUATE": "$1000-$10000.",
                            "WEAK": "$10000-$100000.",
                            "FAIL": "Over $100000.",
                        },
                    ),
                    SubQuestion(
                        key="cost_operating",
                        question="Are ongoing costs manageable?",
                        tier_anchors={
                            "STRONG": "Minimal ongoing costs.",
                            "ADEQUATE": "Moderate recurring costs.",
                            "WEAK": "High recurring costs.",
                            "FAIL": "Unsustainable costs.",
                        },
                    ),
                ],
            ),
            Criterion(
                key="alignment",
                name="Alignment",
                description="Does it solve the original problem?",
                weight=2.0,
                sub_questions=[
                    SubQuestion(
                        key="alignment_direct",
                        question="Does this directly address the problem?",
                        tier_anchors={
                            "STRONG": "Directly solves the core problem.",
                            "ADEQUATE": "Partially addresses the problem.",
                            "WEAK": "Tangentially related.",
                            "FAIL": "Unrelated to the problem.",
                        },
                        axiomatic=True,
                    ),
                    SubQuestion(
                        key="alignment_scope",
                        question="Is the scope appropriate?",
                        tier_anchors={
                            "STRONG": "Perfect scope.",
                            "ADEQUATE": "Slightly over/under-scoped.",
                            "WEAK": "Significantly mis-scoped.",
                            "FAIL": "Completely wrong scope.",
                        },
                    ),
                    SubQuestion(
                        key="alignment_user",
                        question="Does it serve the target users?",
                        tier_anchors={
                            "STRONG": "Ideal for target users.",
                            "ADEQUATE": "Usable by target users.",
                            "WEAK": "Difficult for target users.",
                            "FAIL": "Unusable by target users.",
                        },
                    ),
                ],
            ),
        ]
    )


@pytest.fixture
def fake_client_factory() -> Callable[[Callable[[dict[str, Any]], dict[str, Any]]], FakeLLMClient]:
    def _make(responder: Callable[[dict[str, Any]], dict[str, Any]]) -> FakeLLMClient:
        return FakeLLMClient(responder=responder)

    return _make


@pytest.fixture
def idea_counter() -> Callable[[], int]:
    """An incrementing counter so generator tests can build unique ideas."""
    counter = itertools.count(1)
    return lambda: next(counter)
