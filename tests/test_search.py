"""End-to-end MCTS run with a fake LLM client."""

from __future__ import annotations

import itertools
from typing import Any

from mcts_qcm.config import MCTSConfig
from mcts_qcm.search import MCTS
from mcts_qcm.scoring import greedy_best_path
from mcts_qcm.visualize import to_json, to_markdown


def _make_responder() -> Any:
    """Build a responder that returns generator vs auditor payloads correctly.

    We tell the two apart by inspecting the system prompt, since both go through
    the same fake client.
    """
    idea_id = itertools.count(1)
    audit_id = itertools.count(1)

    def responder(payload: dict[str, Any]) -> dict[str, Any]:
        system = payload["system"]
        if "MULTIPLE-CHOICE" in system.upper() or "AUDITOR" in system.upper() or "QCM" in system.upper():
            i = next(audit_id)
            # Cycle through 4/4, 3/4, 2/4, 1/4 to give the search interesting structure.
            level = (i - 1) % 4
            flags = [True, True, True, True]
            for k in range(level):
                flags[3 - k] = False
            return {
                "novelty":     {"pass": flags[0], "reason": "test"},
                "resource":    {"pass": flags[1], "reason": "test"},
                "feasibility": {"pass": flags[2], "reason": "test"},
                "alignment":   {"pass": flags[3], "reason": "test"},
            }
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


def test_full_search_grows_tree_and_records_visits(base_config, fake_client_factory) -> None:
    client = fake_client_factory(_make_responder())
    engine = MCTS.with_client(base_config, client)
    root = engine.run("Test problem")

    assert root.children, "Root should have been expanded at least once."
    assert root.visits > 0, "Root should accumulate visits via backprop."
    descendants = root.iter_descendants()
    audited = [n for n in descendants if n.audit is not None]
    assert len(audited) >= 1, "At least one node should have an audit."
    assert all(n.audit is not None for n in descendants if n is not root)


def test_full_search_respects_max_nodes(fake_client_factory) -> None:
    config = MCTSConfig(
        model_gen="fake/g", model_audit="fake/a",
        k_children=4, iterations=10, max_depth=4, max_nodes=8,
        prune_on_failed_resource=False,
    )
    client = fake_client_factory(_make_responder())
    engine = MCTS.with_client(config, client)
    root = engine.run("Capped problem")
    assert len(root.iter_descendants()) <= config.max_nodes


def test_full_search_respects_max_depth(fake_client_factory) -> None:
    config = MCTSConfig(
        model_gen="fake/g", model_audit="fake/a",
        k_children=2, iterations=20, max_depth=2, max_nodes=200,
        prune_on_failed_resource=False,
    )
    client = fake_client_factory(_make_responder())
    engine = MCTS.with_client(config, client)
    root = engine.run("Depth-capped problem")
    deepest = max(n.depth for n in root.iter_descendants())
    assert deepest <= config.max_depth


def test_pruning_marks_resource_failures_dead(fake_client_factory) -> None:
    """When ``prune_on_failed_resource`` is True, nodes failing resource are dead."""
    def responder(payload: dict[str, Any]) -> dict[str, Any]:
        if "QCM" in payload["system"].upper() or "AUDITOR" in payload["system"].upper():
            return {
                "novelty":     {"pass": True,  "reason": "x"},
                "resource":    {"pass": False, "reason": "too expensive"},
                "feasibility": {"pass": True,  "reason": "x"},
                "alignment":   {"pass": True,  "reason": "x"},
            }
        return {"ideas": [{"idea": "alpha branch"}, {"idea": "beta branch"}]}

    config = MCTSConfig(
        model_gen="fake/g", model_audit="fake/a",
        k_children=2, iterations=2, max_depth=3, max_nodes=50,
        prune_on_failed_resource=True,
    )
    client = fake_client_factory(responder)
    engine = MCTS.with_client(config, client)
    root = engine.run("Resource-fail problem")

    audited = [n for n in root.iter_descendants() if n.audit is not None]
    assert audited, "Expected at least one audited child."
    assert all(n.dead for n in audited), "Every resource-failing node should be dead."


def test_serialization_helpers_dont_crash(fake_client_factory, base_config) -> None:
    client = fake_client_factory(_make_responder())
    engine = MCTS.with_client(base_config, client)
    root = engine.run("Serialization problem")

    js = to_json(root)
    assert js.startswith("{")
    md = to_markdown(root)
    assert "Greedy best path" in md
    path = greedy_best_path(root)
    assert path[0] is root
