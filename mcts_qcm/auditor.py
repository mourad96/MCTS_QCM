"""QCM Auditor — the LLM-driven value function of the MCTS engine.

Performs a tiered evaluation of an idea against a dynamic Rubric.  Each
sub-question is classified into one of four tiers (STRONG / ADEQUATE /
WEAK / FAIL) by the LLM.  The tiers are mapped to numerical values
deterministically in Python.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from mcts_qcm.config import MCTSConfig
from mcts_qcm.generator import format_path
from mcts_qcm.llm import LLMClient, LLMError, LiteLLMClient
from mcts_qcm.node import Node
from mcts_qcm.prompts import TIERED_AUDITOR_USER, build_tiered_auditor_system
from mcts_qcm.rubric import Rubric, TIER_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic result models
# ---------------------------------------------------------------------------

class SubQuestionResult(BaseModel):
    """Result for one atomic sub-question."""

    key: str
    tier: str = "FAIL"  # "STRONG" | "ADEQUATE" | "WEAK" | "FAIL"
    reason: str = ""


class AuditResult(BaseModel):
    """Complete tiered audit result for a single idea."""

    results: list[SubQuestionResult] = Field(default_factory=list)

    def tier_counts(self) -> dict[str, int]:
        """Count occurrences of each tier, e.g. {"STRONG": 7, ...}."""
        counts = {t: 0 for t in TIER_NAMES}
        for r in self.results:
            tier = r.tier.upper()
            if tier in counts:
                counts[tier] += 1
        return counts

    def summary(self) -> str:
        """Compact summary, e.g. '7S 3A 1W 1F'."""
        c = self.tier_counts()
        return f"{c['STRONG']}S {c['ADEQUATE']}A {c['WEAK']}W {c['FAIL']}F"

    def has_axiomatic_failure(self, axiomatic_keys: set[str]) -> bool:
        """True if any axiomatic sub-question scored FAIL."""
        for r in self.results:
            if r.key in axiomatic_keys and r.tier.upper() == "FAIL":
                return True
        return False

    @property
    def num_results(self) -> int:
        return len(self.results)


# ---------------------------------------------------------------------------
# Tolerant parsing helpers
# ---------------------------------------------------------------------------

# Common LLM-generated synonyms mapped to canonical tier names.
_TIER_ALIASES: dict[str, str] = {
    "STRONG": "STRONG",
    "PASS": "STRONG",
    "GOOD": "STRONG",
    "EXCELLENT": "STRONG",
    "HIGH": "STRONG",
    "YES": "STRONG",
    "ADEQUATE": "ADEQUATE",
    "OK": "ADEQUATE",
    "OKAY": "ADEQUATE",
    "MODERATE": "ADEQUATE",
    "MEDIUM": "ADEQUATE",
    "PARTIAL": "ADEQUATE",
    "WEAK": "WEAK",
    "LOW": "WEAK",
    "POOR": "WEAK",
    "MARGINAL": "WEAK",
    "FAIL": "FAIL",
    "FAILURE": "FAIL",
    "NO": "FAIL",
    "IMPOSSIBLE": "FAIL",
    "NONE": "FAIL",
}


def _coerce_tier(value: Any) -> str:
    """Tolerantly coerce an LLM-emitted tier value to a canonical tier name."""
    if isinstance(value, bool):
        return "STRONG" if value else "FAIL"
    if isinstance(value, str):
        normed = value.strip().upper()
        if normed in TIER_NAMES:
            return normed
        return _TIER_ALIASES.get(normed, "FAIL")
    if isinstance(value, (int, float)):
        return "STRONG" if value else "FAIL"
    return "FAIL"


def parse_tiered_payload(data: dict[str, Any], rubric: Rubric) -> AuditResult:
    """Build an ``AuditResult`` from the raw JSON payload returned by the LLM.

    Accepts the canonical schema (``{sq_key: {tier, reason}}``) and tolerates
    common deviations like flat ``{sq_key: "STRONG"}`` or missing sub-questions
    (defaulted to FAIL).
    """
    results: list[SubQuestionResult] = []

    for sq in rubric.all_sub_questions():
        entry = data.get(sq.key)
        if entry is None:
            results.append(
                SubQuestionResult(
                    key=sq.key,
                    tier="FAIL",
                    reason="Not evaluated by auditor.",
                )
            )
            continue

        if isinstance(entry, dict):
            raw_tier = entry.get("tier", entry.get("classification", "FAIL"))
            tier = _coerce_tier(raw_tier)
            reason = str(entry.get("reason", ""))
        else:
            tier = _coerce_tier(entry)
            reason = ""

        results.append(SubQuestionResult(key=sq.key, tier=tier, reason=reason))

    return AuditResult(results=results)


# ---------------------------------------------------------------------------
# QCMAuditor
# ---------------------------------------------------------------------------

@dataclass
class QCMAuditor:
    """Audits a single idea against a tiered rubric."""

    config: MCTSConfig
    client: LLMClient | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = LiteLLMClient(max_retries=self.config.max_retries)

    def audit(self, *, problem: str, node: Node, rubric: Rubric) -> AuditResult:
        """Audit ``node.idea`` against the ``rubric`` for the given ``problem``."""
        parent_path = node.parent.path_from_root() if node.parent else [node]
        path_str = format_path(parent_path)

        system = build_tiered_auditor_system(rubric)
        user = TIERED_AUDITOR_USER.format(
            problem=problem, path=path_str, idea=node.idea,
        )

        assert self.client is not None
        try:
            data = self.client.chat_json(
                model=self.config.model_audit,
                system=system,
                user=user,
                temperature=self.config.temperature_audit,
                seed=self.config.seed,
                timeout=self.config.request_timeout,
            )
        except LLMError:
            return _fail_all(rubric, reason="LLM call failed; treated as all-FAIL.")

        try:
            return parse_tiered_payload(data, rubric)
        except (ValidationError, TypeError, KeyError):
            return _fail_all(rubric, reason="Auditor returned a malformed payload; treated as all-FAIL.")


def _fail_all(rubric: Rubric, *, reason: str) -> AuditResult:
    """Build an AuditResult with every sub-question set to FAIL."""
    return AuditResult(
        results=[
            SubQuestionResult(key=sq.key, tier="FAIL", reason=reason)
            for sq in rubric.all_sub_questions()
        ]
    )
