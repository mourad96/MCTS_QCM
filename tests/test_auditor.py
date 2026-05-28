"""Tests for tiered QCM payload parsing and the auditor's robustness."""

from __future__ import annotations

import pytest

from mcts_qcm.auditor import (
    AuditResult,
    QCMAuditor,
    SubQuestionResult,
    _coerce_tier,
    parse_tiered_payload,
)
from mcts_qcm.node import Node


# ---------------------------------------------------------------------------
# _coerce_tier
# ---------------------------------------------------------------------------

def test_coerce_tier_canonical_names() -> None:
    assert _coerce_tier("STRONG") == "STRONG"
    assert _coerce_tier("ADEQUATE") == "ADEQUATE"
    assert _coerce_tier("WEAK") == "WEAK"
    assert _coerce_tier("FAIL") == "FAIL"


def test_coerce_tier_case_insensitive() -> None:
    assert _coerce_tier("strong") == "STRONG"
    assert _coerce_tier("Adequate") == "ADEQUATE"
    assert _coerce_tier("weak") == "WEAK"
    assert _coerce_tier("fail") == "FAIL"


def test_coerce_tier_common_synonyms() -> None:
    assert _coerce_tier("PASS") == "STRONG"
    assert _coerce_tier("GOOD") == "STRONG"
    assert _coerce_tier("OK") == "ADEQUATE"
    assert _coerce_tier("POOR") == "WEAK"
    assert _coerce_tier("IMPOSSIBLE") == "FAIL"
    assert _coerce_tier("NO") == "FAIL"


def test_coerce_tier_booleans() -> None:
    assert _coerce_tier(True) == "STRONG"
    assert _coerce_tier(False) == "FAIL"


def test_coerce_tier_unknown_defaults_fail() -> None:
    assert _coerce_tier("UNKNOWN_VALUE") == "FAIL"
    assert _coerce_tier(None) == "FAIL"


# ---------------------------------------------------------------------------
# parse_tiered_payload
# ---------------------------------------------------------------------------

def test_parse_tiered_payload_canonical(sample_rubric) -> None:
    payload = {
        "feasibility_tech": {"tier": "STRONG", "reason": "Off-the-shelf."},
        "feasibility_time": {"tier": "ADEQUATE", "reason": "About a year."},
        "feasibility_skill": {"tier": "STRONG", "reason": "Standard."},
        "cost_capital": {"tier": "WEAK", "reason": "Expensive."},
        "cost_operating": {"tier": "ADEQUATE", "reason": "Moderate."},
        "alignment_direct": {"tier": "STRONG", "reason": "Direct hit."},
        "alignment_scope": {"tier": "ADEQUATE", "reason": "Slightly wide."},
        "alignment_user": {"tier": "STRONG", "reason": "Perfect match."},
    }
    result = parse_tiered_payload(payload, sample_rubric)
    assert result.num_results == 8
    counts = result.tier_counts()
    assert counts["STRONG"] == 4
    assert counts["ADEQUATE"] == 3
    assert counts["WEAK"] == 1
    assert counts["FAIL"] == 0


def test_parse_tiered_payload_flat_strings(sample_rubric) -> None:
    payload = {
        "feasibility_tech": "STRONG",
        "feasibility_time": "ADEQUATE",
        "feasibility_skill": "WEAK",
        "cost_capital": "FAIL",
        "cost_operating": "STRONG",
        "alignment_direct": "STRONG",
        "alignment_scope": "ADEQUATE",
        "alignment_user": "WEAK",
    }
    result = parse_tiered_payload(payload, sample_rubric)
    assert result.num_results == 8
    assert result.tier_counts()["FAIL"] == 1


def test_parse_tiered_payload_missing_keys_get_fail(sample_rubric) -> None:
    payload = {
        "feasibility_tech": "STRONG",
        # all other keys missing
    }
    result = parse_tiered_payload(payload, sample_rubric)
    assert result.num_results == 8
    # 7 missing → 7 FAILs + 1 STRONG
    assert result.tier_counts()["FAIL"] == 7
    assert result.tier_counts()["STRONG"] == 1


def test_parse_tiered_payload_tolerant_coercion(sample_rubric) -> None:
    payload = {
        "feasibility_tech": {"tier": "good", "reason": "x"},
        "feasibility_time": {"tier": "ok", "reason": "x"},
        "feasibility_skill": {"tier": "poor", "reason": "x"},
        "cost_capital": {"tier": "impossible", "reason": "x"},
        "cost_operating": True,
        "alignment_direct": False,
        "alignment_scope": "pass",
        "alignment_user": "moderate",
    }
    result = parse_tiered_payload(payload, sample_rubric)
    tiers = {r.key: r.tier for r in result.results}
    assert tiers["feasibility_tech"] == "STRONG"     # good -> STRONG
    assert tiers["feasibility_time"] == "ADEQUATE"    # ok -> ADEQUATE
    assert tiers["feasibility_skill"] == "WEAK"       # poor -> WEAK
    assert tiers["cost_capital"] == "FAIL"             # impossible -> FAIL
    assert tiers["cost_operating"] == "STRONG"         # True -> STRONG
    assert tiers["alignment_direct"] == "FAIL"         # False -> FAIL
    assert tiers["alignment_scope"] == "STRONG"        # pass -> STRONG
    assert tiers["alignment_user"] == "ADEQUATE"       # moderate -> ADEQUATE


# ---------------------------------------------------------------------------
# AuditResult
# ---------------------------------------------------------------------------

def test_audit_result_summary() -> None:
    result = AuditResult(results=[
        SubQuestionResult(key="a", tier="STRONG"),
        SubQuestionResult(key="b", tier="STRONG"),
        SubQuestionResult(key="c", tier="ADEQUATE"),
        SubQuestionResult(key="d", tier="WEAK"),
        SubQuestionResult(key="e", tier="FAIL"),
    ])
    assert result.summary() == "2S 1A 1W 1F"


def test_audit_result_has_axiomatic_failure() -> None:
    result = AuditResult(results=[
        SubQuestionResult(key="safe", tier="FAIL"),
        SubQuestionResult(key="optional", tier="STRONG"),
    ])
    assert result.has_axiomatic_failure({"safe"}) is True
    assert result.has_axiomatic_failure({"optional"}) is False
    assert result.has_axiomatic_failure({"nonexistent"}) is False


def test_audit_result_no_axiomatic_failure_when_not_fail() -> None:
    result = AuditResult(results=[
        SubQuestionResult(key="safe", tier="WEAK"),
    ])
    assert result.has_axiomatic_failure({"safe"}) is False


# ---------------------------------------------------------------------------
# QCMAuditor integration
# ---------------------------------------------------------------------------

def test_auditor_returns_all_fail_on_malformed_payload(
    base_config, fake_client_factory, sample_rubric,
) -> None:
    # Return a completely invalid payload
    bad = fake_client_factory(lambda _p: {"totally": "wrong"})
    auditor = QCMAuditor(base_config, client=bad)
    node = Node(idea="some idea", depth=1)
    result = auditor.audit(problem="root problem", node=node, rubric=sample_rubric)
    # All sub-questions should be FAIL (missing keys → FAIL)
    assert result.tier_counts()["FAIL"] == sample_rubric.sub_question_count()
