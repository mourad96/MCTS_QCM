"""Scoring helpers: UCB1 selection, tiered score computation, and best-path extraction."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcts_qcm.auditor import AuditResult
    from mcts_qcm.node import Node
    from mcts_qcm.rubric import Rubric


def compute_score(audit: AuditResult, rubric: Rubric) -> float:
    """Weighted average of tier values across all sub-questions, in [0, 1].

    For each criterion in the rubric:
      criterion_score = mean(tier_value for each of its sub-questions)

    Overall = sum(criterion_score * criterion.weight) / sum(weights)

    If no results match, returns 0.0.
    """
    tier_values = rubric.tier_values
    total_weight = 0.0
    weighted_sum = 0.0

    for criterion in rubric.criteria:
        sq_keys = {sq.key for sq in criterion.sub_questions}
        matching = [r for r in audit.results if r.key in sq_keys]
        if not matching:
            continue

        criterion_score = sum(
            tier_values.get(r.tier.upper(), 0.0) for r in matching
        ) / len(matching)

        weighted_sum += criterion_score * criterion.weight
        total_weight += criterion.weight

    if total_weight <= 0:
        return 0.0
    return weighted_sum / total_weight


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
