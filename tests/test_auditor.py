"""Tests for QCM payload parsing and the auditor's robustness."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mcts_qcm.auditor import QCMAuditor, parse_qcm_payload
from mcts_qcm.node import Node


def test_parse_qcm_payload_canonical() -> None:
    payload = {
        "novelty":     {"pass": True,  "reason": "fresh"},
        "resource":    {"pass": False, "reason": "expensive"},
        "feasibility": {"pass": True,  "reason": "doable"},
        "alignment":   {"pass": True,  "reason": "on target"},
    }
    r = parse_qcm_payload(payload)
    assert r.num_passed == 3
    assert r.fraction_passed == 0.75
    assert r.reasons["resource"] == "expensive"


def test_parse_qcm_payload_flat_booleans() -> None:
    payload = {"novelty": True, "resource": False, "feasibility": True, "alignment": True}
    r = parse_qcm_payload(payload)
    assert r.num_passed == 3


def test_parse_qcm_payload_string_pass_values() -> None:
    payload = {
        "novelty":     {"pass": "yes",  "reason": "x"},
        "resource":    {"pass": "no",   "reason": "x"},
        "feasibility": {"pass": "true", "reason": "x"},
        "alignment":   {"pass": "0",    "reason": "x"},
    }
    r = parse_qcm_payload(payload)
    assert (r.novelty, r.resource, r.feasibility, r.alignment) == (True, False, True, False)


def test_parse_qcm_payload_passed_alias() -> None:
    payload = {
        "novelty":     {"passed": True, "reason": "x"},
        "resource":    {"passed": True, "reason": "x"},
        "feasibility": {"passed": True, "reason": "x"},
        "alignment":   {"passed": True, "reason": "x"},
    }
    r = parse_qcm_payload(payload)
    assert r.num_passed == 4


def test_parse_qcm_payload_missing_field_raises() -> None:
    payload = {"novelty": True, "resource": True, "feasibility": True}  # alignment missing
    with pytest.raises(ValidationError):
        parse_qcm_payload(payload)


def test_qcmresult_summary_format() -> None:
    payload = {"novelty": True, "resource": False, "feasibility": True, "alignment": False}
    r = parse_qcm_payload(payload)
    assert r.summary() == "2/4 [NY RN FY AN]"


def test_auditor_returns_zero_on_malformed_payload(base_config, fake_client_factory) -> None:
    bad = fake_client_factory(lambda _p: {"novelty": True})  # missing fields
    auditor = QCMAuditor(base_config, client=bad)
    node = Node(idea="some idea", depth=1)
    result = auditor.audit(problem="root problem", node=node)
    assert result.num_passed == 0
    assert "error" in result.reasons
