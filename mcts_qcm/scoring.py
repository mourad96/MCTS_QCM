"""Scoring helpers: UCB1 selection and QCM pass-rate computation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcts_qcm.auditor import QCMResult
    from mcts_qcm.node import Node


def pass_rate(audit: QCMResult, weights: dict[str, float] | None = None) -> float:
    """Return the weighted pass-rate of a QCM audit, in [0, 1].

    With default unit weights this is simply the fraction of checks that passed.
    Weights are normalized internally so the result always lives in [0, 1].
    """
    if weights is None:
        weights = {"novelty": 1.0, "resource": 1.0, "feasibility": 1.0, "alignment": 1.0}
    total_w = sum(weights.values())
    if total_w <= 0:
        raise ValueError("Weights must sum to a positive value.")
    score = 0.0
    score += weights.get("novelty", 0.0) * (1.0 if audit.novelty else 0.0)
    score += weights.get("resource", 0.0) * (1.0 if audit.resource else 0.0)
    score += weights.get("feasibility", 0.0) * (1.0 if audit.feasibility else 0.0)
    score += weights.get("alignment", 0.0) * (1.0 if audit.alignment else 0.0)
    return score / total_w


def ucb1(node: Node, parent_visits: int, c: float) -> float:
    """Standard UCB1 / UCT score for `node` under a parent with `parent_visits`.

    Unvisited nodes return +infinity so they get explored first.
    """
    if node.dead:
        return -math.inf
    if node.visits == 0:
        return math.inf
    exploitation = node.mean_value
    exploration = c * math.sqrt(math.log(max(parent_visits, 1)) / node.visits)
    return exploitation + exploration


def select_best_child(parent: Node, c: float) -> Node | None:
    """Pick the child of `parent` maximizing UCB1. None if no live children."""
    live = [child for child in parent.children if not child.dead]
    if not live:
        return None
    return max(live, key=lambda ch: ucb1(ch, parent.visits, c))


def greedy_best_path(root: Node) -> list[Node]:
    """Walk the tree picking the highest-mean-value child at each step.

    Tie-broken by visit count (more visits = more confidence). Used to extract
    the engine's final "answer" after the search completes.
    """
    path = [root]
    cur = root
    while cur.children:
        live = [c for c in cur.children if not c.dead]
        candidates = live if live else cur.children
        if not candidates:
            break
        cur = max(candidates, key=lambda ch: (ch.mean_value, ch.visits))
        path.append(cur)
    return path
