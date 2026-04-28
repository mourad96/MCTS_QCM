"""Tests for the IdeaGenerator (deduplication + parsing)."""

from __future__ import annotations

from mcts_qcm.generator import IdeaGenerator, jaccard
from mcts_qcm.node import Node


def test_jaccard_identical() -> None:
    assert jaccard("solar power", "solar power") == 1.0


def test_jaccard_disjoint() -> None:
    assert jaccard("solar panels", "wind turbines") == 0.0


def test_generator_dedupes_near_duplicates(base_config, fake_client_factory) -> None:
    payload = {
        "ideas": [
            {"idea": "Use solar panels to power the desalination plant"},
            {"idea": "Use solar panels to power the desalination plant"},  # exact duplicate
            {"idea": "Build a wind-powered reverse osmosis system"},
        ]
    }
    client = fake_client_factory(lambda _p: payload)
    gen = IdeaGenerator(base_config, client=client)

    root = Node(idea="Design a low-cost desalination process")
    ideas = gen.generate(problem=root.idea, node=root, k=3)
    assert len(ideas) == 2
    assert ideas[0].startswith("Use solar")
    assert ideas[1].startswith("Build a wind")


def test_generator_respects_k_cap(base_config, fake_client_factory) -> None:
    payload = {
        "ideas": [
            {"idea": "Strategy alpha: solar thermal distillation"},
            {"idea": "Strategy beta: graphene oxide membranes"},
            {"idea": "Strategy gamma: capacitive deionization"},
            {"idea": "Strategy delta: forward osmosis with recoverable draw"},
        ]
    }
    client = fake_client_factory(lambda _p: payload)
    gen = IdeaGenerator(base_config, client=client)

    ideas = gen.generate(problem="x", node=Node(idea="x"), k=2)
    assert len(ideas) == 2


def test_generator_handles_string_only_payload(base_config, fake_client_factory) -> None:
    payload = {"ideas": ["raw string idea one", "raw string idea two"]}
    client = fake_client_factory(lambda _p: payload)
    gen = IdeaGenerator(base_config, client=client)
    ideas = gen.generate(problem="x", node=Node(idea="x"), k=2)
    assert ideas == ["raw string idea one", "raw string idea two"]


def test_generator_empty_payload_returns_nothing(base_config, fake_client_factory) -> None:
    client = fake_client_factory(lambda _p: {"ideas": []})
    gen = IdeaGenerator(base_config, client=client)
    assert gen.generate(problem="x", node=Node(idea="x"), k=3) == []
