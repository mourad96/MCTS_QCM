"""QCM Auditor — the LLM-driven value function of the MCTS engine.

Performs a strict 4-question audit of an idea (Novelty, Resource, Feasibility,
Alignment) and returns a Pydantic-validated ``QCMResult``. The pass-rate over
those four checks is the value used by UCB1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from mcts_qcm.config import MCTSConfig
from mcts_qcm.generator import format_path
from mcts_qcm.llm import LLMClient, LiteLLMClient, LLMError
from mcts_qcm.node import Node
from mcts_qcm.prompts import AUDITOR_SYSTEM, AUDITOR_USER


class QCMResult(BaseModel):
    """The 4-question multiple-choice audit result for a single idea."""

    novelty: bool
    resource: bool
    feasibility: bool
    alignment: bool
    reasons: dict[str, str] = Field(default_factory=dict)

    @property
    def num_passed(self) -> int:
        return int(self.novelty) + int(self.resource) + int(self.feasibility) + int(self.alignment)

    @property
    def fraction_passed(self) -> float:
        return self.num_passed / 4.0

    def summary(self) -> str:
        flags = "".join(
            "Y" if v else "N"
            for v in (self.novelty, self.resource, self.feasibility, self.alignment)
        )
        return f"{self.num_passed}/4 [N{flags[0]} R{flags[1]} F{flags[2]} A{flags[3]}]"


def _coerce_pass(value: Any) -> bool:
    """Tolerant coercion for ``pass`` fields the LLM might return as strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1", "pass", "passed"}
    return False


def parse_qcm_payload(data: dict[str, Any]) -> QCMResult:
    """Build a ``QCMResult`` from the raw JSON payload returned by the LLM.

    Accepts the canonical schema (``{check: {pass, reason}}``) and tolerates
    common deviations like flat ``{check: bool}`` or ``passed`` instead of
    ``pass``. Raises ``ValidationError`` if a required check is missing.
    """
    required = ("novelty", "resource", "feasibility", "alignment")
    flags: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    for key in required:
        if key not in data:
            raise ValidationError.from_exception_data(
                title="QCMResult",
                line_errors=[
                    {
                        "type": "missing",
                        "loc": (key,),
                        "input": data,
                    }
                ],
            )
        entry = data[key]
        if isinstance(entry, dict):
            raw_pass = entry.get("pass", entry.get("passed"))
            flags[key] = _coerce_pass(raw_pass)
            reason = entry.get("reason", "")
            if reason:
                reasons[key] = str(reason)
        else:
            flags[key] = _coerce_pass(entry)
    return QCMResult(**flags, reasons=reasons)


@dataclass
class QCMAuditor:
    """Audits a single idea against the 4-question QCM checklist."""

    config: MCTSConfig
    client: LLMClient | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = LiteLLMClient(max_retries=self.config.max_retries)

    def audit(self, *, problem: str, node: Node) -> QCMResult:
        """Audit ``node.idea`` against the original ``problem``."""
        parent_path = node.parent.path_from_root() if node.parent else [node]
        path_str = format_path(parent_path)
        user = AUDITOR_USER.format(problem=problem, path=path_str, idea=node.idea)
        assert self.client is not None
        try:
            data = self.client.chat_json(
                model=self.config.model_audit,
                system=AUDITOR_SYSTEM,
                user=user,
                temperature=self.config.temperature_audit,
                seed=self.config.seed,
                timeout=self.config.request_timeout,
            )
        except LLMError:
            return QCMResult(
                novelty=False,
                resource=False,
                feasibility=False,
                alignment=False,
                reasons={"error": "LLM call failed; treated as 0/4."},
            )
        try:
            return parse_qcm_payload(data)
        except (ValidationError, TypeError, KeyError):
            return QCMResult(
                novelty=False,
                resource=False,
                feasibility=False,
                alignment=False,
                reasons={"error": "Auditor returned a malformed payload; treated as 0/4."},
            )
