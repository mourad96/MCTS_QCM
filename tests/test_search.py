"""End-to-end MCTS run with a fake LLM client and tiered rubric."""

from __future__ import annotations

import itertools
from typing import Any

from mcts_qcm.config import MCTSConfig
from mcts_qcm.search import MCTS
from mcts_qcm.scoring import greedy_best_path
from mcts_qcm.visualize import to_json, to_markdown


def _make_responder(sample_rubric) -> Any:
    """Build a responder that returns generator vs auditor payloads correctly.

    We tell the two apart by inspecting the system prompt, since both go through
    the same fake client.
    """
    idea_id = itertools.count(1)
    audit_id = itertools.count(1)

    # Get all sub-question keys from the rubric
    sq_keys = [sq.key for sq in sample_rubric.all_sub_questions()]
    tiers = ["STRONG", "ADEQUATE", "WEAK", "FAIL"]

    def responder(payload: dict[str, Any]) -> dict[str, Any]:
        system = payload["system"]
        # Detect auditor by checking for tier-related keywords
        if "STRONG" in system and "ADEQUATE" in system and "SUB-QUESTIONS" in system:
            i = next(audit_id)
            # Cycle through tier patterns to give the search interesting structure
            result = {}
            for j, key in enumerate(sq_keys):
                tier_idx = (i + j) % 4
                result[key] = {"tier": tiers[tier_idx], "reason": "test"}
            return result
        # Generator
        n = next(idea_id)
        return {
            "ideas": [
                {"idea": f"Distinct idea number {n}-A about strategy alpha"},
                {"idea": f"Different idea number {n}-B about strategy beta"},
                {"idea": f"Separate idea number {n}-C about strategy gamma"},
                {"idea": f"Unique idea number {n}-D about strategy delta"},
            ]
        }

    return responder


def test_full_search_grows_tree_and_records_visits(
    base_config, fake_client_factory, sample_rubric,
) -> None:
    client = fake_client_factory(_make_responder(sample_rubric))
    engine = MCTS.with_client(base_config, sample_rubric, client)
    root = engine.run("Test problem")

    assert root.children, "Root should have been expanded at least once."
    assert root.visits > 0, "Root should accumulate visits via backprop."
    descendants = root.iter_descendants()
    audited = [n for n in descendants if n.audit is not None]
    assert len(audited) >= 1, "At least one node should have an audit."
    assert all(n.audit is not None for n in descendants if n is not root)


def test_full_search_respects_max_nodes(fake_client_factory, sample_rubric) -> None:
    config = MCTSConfig(
        model_gen="fake/g", model_audit="fake/a",
        k_children=4, iterations=10, max_depth=4, max_nodes=8,
        prune_threshold=0.0,
    )
    client = fake_client_factory(_make_responder(sample_rubric))
    engine = MCTS.with_client(config, sample_rubric, client)
    root = engine.run("Capped problem")
    assert len(root.iter_descendants()) <= config.max_nodes


def test_full_search_respects_max_depth(fake_client_factory, sample_rubric) -> None:
    config = MCTSConfig(
        model_gen="fake/g", model_audit="fake/a",
        k_children=2, iterations=20, max_depth=2, max_nodes=200,
        prune_threshold=0.0,
    )
    client = fake_client_factory(_make_responder(sample_rubric))
    engine = MCTS.with_client(config, sample_rubric, client)
    root = engine.run("Depth-capped problem")
    deepest = max(n.depth for n in root.iter_descendants())
    assert deepest <= config.max_depth


def test_axiomatic_pruning(fake_client_factory, sample_rubric) -> None:
    """When an axiomatic sub-question scores FAIL, the node is marked dead."""
    axiomatic_keys = sample_rubric.axiomatic_keys()
    assert len(axiomatic_keys) > 0, "Sample rubric should have axiomatic keys."

    def responder(payload: dict[str, Any]) -> dict[str, Any]:
        system = payload["system"]
        if "SUB-QUESTIONS" in system:
            # Make all axiomatic sub-questions FAIL
            result = {}
            for sq in sample_rubric.all_sub_questions():
                if sq.axiomatic:
                    result[sq.key] = {"tier": "FAIL", "reason": "hard fail"}
                else:
                    result[sq.key] = {"tier": "STRONG", "reason": "fine"}
            return result
        return {"ideas": [{"idea": "alpha branch"}, {"idea": "beta branch"}]}

    config = MCTSConfig(
        model_gen="fake/g", model_audit="fake/a",
        k_children=2, iterations=2, max_depth=3, max_nodes=50,
        prune_threshold=0.0,  # only axiomatic pruning
    )
    client = fake_client_factory(responder)
    engine = MCTS.with_client(config, sample_rubric, client)
    root = engine.run("Axiomatic-fail problem")

    audited = [n for n in root.iter_descendants() if n.audit is not None]
    assert audited, "Expected at least one audited child."
    assert all(n.dead for n in audited), "Every node with axiomatic FAIL should be dead."


def test_threshold_pruning(fake_client_factory, sample_rubric) -> None:
    """Nodes with low weighted scores get pruned by threshold."""
    def responder(payload: dict[str, Any]) -> dict[str, Any]:
        system = payload["system"]
        if "SUB-QUESTIONS" in system:
            # All FAIL → score = 0.0, below any positive threshold
            return {sq.key: {"tier": "FAIL", "reason": "bad"} for sq in sample_rubric.all_sub_questions()}
        return {"ideas": [{"idea": "idea one"}, {"idea": "idea two"}]}

    config = MCTSConfig(
        model_gen="fake/g", model_audit="fake/a",
        k_children=2, iterations=2, max_depth=3, max_nodes=50,
        prune_threshold=0.25,
    )
    client = fake_client_factory(responder)
    engine = MCTS.with_client(config, sample_rubric, client)
    root = engine.run("Low-score problem")

    audited = [n for n in root.iter_descendants() if n.audit is not None]
    assert audited, "Expected at least one audited child."
    assert all(n.dead for n in audited), "All low-score nodes should be pruned."


def test_serialization_helpers_dont_crash(
    fake_client_factory, base_config, sample_rubric,
) -> None:
    client = fake_client_factory(_make_responder(sample_rubric))
    engine = MCTS.with_client(base_config, sample_rubric, client)
    root = engine.run("Serialization problem")

    js = to_json(root)
    assert js.startswith("{")
    md = to_markdown(root)
    assert "Greedy best path" in md
    path = greedy_best_path(root)
    assert path[0] is root
