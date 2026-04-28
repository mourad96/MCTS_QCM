"""Tests for UCB1 and pass-rate math."""

from __future__ import annotations

import math

from mcts_qcm.auditor import QCMResult
from mcts_qcm.node import Node
from mcts_qcm.scoring import greedy_best_path, pass_rate, select_best_child, ucb1


def _audit(n: int) -> QCMResult:
    """Build a QCMResult with `n` of 4 checks passing (in declaration order)."""
    flags = [True] * n + [False] * (4 - n)
    return QCMResult(
        novelty=flags[0], resource=flags[1], feasibility=flags[2], alignment=flags[3]
    )


def test_pass_rate_equal_weights() -> None:
    assert pass_rate(_audit(4)) == 1.0
    assert pass_rate(_audit(0)) == 0.0
    assert pass_rate(_audit(3)) == 0.75
    assert pass_rate(_audit(2)) == 0.5


def test_pass_rate_custom_weights() -> None:
    audit = _audit(2)  # novelty=T, resource=T, feasibility=F, alignment=F
    weights = {"novelty": 1.0, "resource": 1.0, "feasibility": 0.0, "alignment": 2.0}
    # Score = (1*1 + 1*1 + 0*0 + 2*0) / (1+1+0+2) = 2/4 = 0.5
    assert pass_rate(audit, weights) == 0.5


def test_ucb1_unvisited_returns_inf() -> None:
    n = Node(idea="x")
    assert ucb1(n, parent_visits=10, c=1.0) == math.inf


def test_ucb1_dead_returns_minus_inf() -> None:
    n = Node(idea="x", visits=3, value_sum=1.5)
    n.dead = True
    assert ucb1(n, parent_visits=10, c=1.0) == -math.inf


def test_ucb1_balances_exploit_and_explore() -> None:
    parent = Node(idea="root")
    parent.visits = 100
    a = Node(idea="A", parent=parent, visits=10, value_sum=8.0)  # mean=0.8
    b = Node(idea="B", parent=parent, visits=1, value_sum=0.4)   # mean=0.4 but rarely visited
    parent.children = [a, b]

    score_a = ucb1(a, parent_visits=parent.visits, c=1.41)
    score_b = ucb1(b, parent_visits=parent.visits, c=1.41)
    assert score_b > score_a, "Rarely visited child should win on exploration term"


def test_select_best_child_skips_dead() -> None:
    parent = Node(idea="root", visits=10)
    good = Node(idea="good", parent=parent, visits=2, value_sum=1.5)
    dead = Node(idea="dead", parent=parent, visits=1, value_sum=1.0, dead=True)
    parent.children = [dead, good]
    chosen = select_best_child(parent, c=1.41)
    assert chosen is good


def test_select_best_child_returns_none_when_all_dead() -> None:
    parent = Node(idea="root")
    d1 = Node(idea="d1", parent=parent, dead=True)
    d2 = Node(idea="d2", parent=parent, dead=True)
    parent.children = [d1, d2]
    assert select_best_child(parent, c=1.41) is None


def test_greedy_best_path_picks_highest_mean() -> None:
    root = Node(idea="root")
    a = root.add_child("A")
    b = root.add_child("B")
    a.visits, a.value_sum = 5, 5.0      # mean = 1.0
    b.visits, b.value_sum = 5, 2.0      # mean = 0.4
    a1 = a.add_child("A1")
    a2 = a.add_child("A2")
    a1.visits, a1.value_sum = 2, 1.0    # mean = 0.5
    a2.visits, a2.value_sum = 2, 1.8    # mean = 0.9

    path = greedy_best_path(root)
    assert [n.idea for n in path] == ["root", "A", "A2"]
