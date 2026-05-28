"""Tests for UCB1, compute_score, and best-path extraction."""

from __future__ import annotations

import math

from mcts_qcm.auditor import AuditResult, SubQuestionResult
from mcts_qcm.node import Node
from mcts_qcm.rubric import Criterion, Rubric, SubQuestion
from mcts_qcm.scoring import compute_score, greedy_best_path, select_best_child, ucb1


def _make_audit(**tiers: str) -> AuditResult:
    """Build an AuditResult from key=tier kwargs."""
    return AuditResult(results=[
        SubQuestionResult(key=k, tier=v) for k, v in tiers.items()
    ])


# ---------------------------------------------------------------------------
# compute_score
# ---------------------------------------------------------------------------

def test_compute_score_all_strong(sample_rubric) -> None:
    audit = _make_audit(
        feasibility_tech="STRONG", feasibility_time="STRONG", feasibility_skill="STRONG",
        cost_capital="STRONG", cost_operating="STRONG",
        alignment_direct="STRONG", alignment_scope="STRONG", alignment_user="STRONG",
    )
    assert compute_score(audit, sample_rubric) == 1.0


def test_compute_score_all_fail(sample_rubric) -> None:
    audit = _make_audit(
        feasibility_tech="FAIL", feasibility_time="FAIL", feasibility_skill="FAIL",
        cost_capital="FAIL", cost_operating="FAIL",
        alignment_direct="FAIL", alignment_scope="FAIL", alignment_user="FAIL",
    )
    assert compute_score(audit, sample_rubric) == 0.0


def test_compute_score_mixed(sample_rubric) -> None:
    audit = _make_audit(
        feasibility_tech="STRONG", feasibility_time="ADEQUATE", feasibility_skill="WEAK",
        cost_capital="STRONG", cost_operating="ADEQUATE",
        alignment_direct="STRONG", alignment_scope="STRONG", alignment_user="STRONG",
    )
    score = compute_score(audit, sample_rubric)
    # feasibility: (1.0 + 0.66 + 0.33) / 3 = 0.6633
    # cost: (1.0 + 0.66) / 2 = 0.83 (weight 1.5)
    # alignment: (1.0 + 1.0 + 1.0) / 3 = 1.0 (weight 2.0)
    # overall: (0.6633*1.0 + 0.83*1.5 + 1.0*2.0) / (1.0+1.5+2.0) = 3.9083/4.5 ≈ 0.8685
    assert 0.86 < score < 0.88


def test_compute_score_respects_weights() -> None:
    """A criterion with weight=0.0 should not affect the score."""
    rubric = Rubric(criteria=[
        Criterion(key="a", name="A", description="d", weight=0.0, sub_questions=[
            SubQuestion(key="a_1", question="?"),
        ]),
        Criterion(key="b", name="B", description="d", weight=1.0, sub_questions=[
            SubQuestion(key="b_1", question="?"),
        ]),
    ])
    audit = _make_audit(a_1="FAIL", b_1="STRONG")
    # weight-0 criterion is excluded from denominator
    score = compute_score(audit, rubric)
    assert score == 1.0


def test_compute_score_empty_audit() -> None:
    rubric = Rubric(criteria=[
        Criterion(key="a", name="A", description="d", weight=1.0, sub_questions=[
            SubQuestion(key="a_1", question="?"),
        ]),
    ])
    audit = AuditResult(results=[])
    assert compute_score(audit, rubric) == 0.0


# ---------------------------------------------------------------------------
# UCB1 (unchanged)
# ---------------------------------------------------------------------------

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
