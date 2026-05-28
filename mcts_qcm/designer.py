"""QCM Designer — LLM-driven rubric proposer.

Analyzes the problem domain and proposes an evaluation framework with
4–6 criteria, each decomposed into 2–3 atomic sub-questions with
tiered anchor descriptions.  The user reviews and confirms the rubric
before the MCTS search begins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from mcts_qcm.config import MCTSConfig
from mcts_qcm.llm import LLMClient, LLMError, LiteLLMClient
from mcts_qcm.prompts import DESIGNER_SYSTEM, DESIGNER_USER
from mcts_qcm.rubric import Rubric

logger = logging.getLogger(__name__)


@dataclass
class QCMDesigner:
    """Proposes a domain-specific evaluation rubric for a given problem."""

    config: MCTSConfig
    client: LLMClient | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = LiteLLMClient(max_retries=self.config.max_retries)

    def propose(self, problem: str) -> Rubric:
        """Propose a tiered evaluation rubric for ``problem``.

        Returns a ``Rubric`` with 4–6 criteria, each containing 2–3
        sub-questions with tier anchors and axiomatic flags.

        Raises:
            LLMError: If the LLM call itself fails.
            ValueError: If the LLM response cannot be parsed into a valid Rubric.
        """
        user = DESIGNER_USER.format(problem=problem)
        assert self.client is not None
        try:
            data = self.client.chat_json(
                model=self.config.model_audit,
                system=DESIGNER_SYSTEM,
                user=user,
                temperature=self.config.temperature_audit,
                seed=self.config.seed,
                timeout=self.config.request_timeout,
            )
        except LLMError:
            logger.exception("Rubric designer LLM call failed.")
            raise

        try:
            rubric = Rubric.from_dict(data)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Failed to parse rubric from LLM response: {exc}") from exc

        self._validate(rubric)
        return rubric

    @staticmethod
    def _validate(rubric: Rubric) -> None:
        """Sanity-check that the proposed rubric is usable."""
        if len(rubric.criteria) < 2:
            raise ValueError(
                f"Rubric must have at least 2 criteria, got {len(rubric.criteria)}."
            )
        for criterion in rubric.criteria:
            if not criterion.sub_questions:
                raise ValueError(
                    f"Criterion '{criterion.key}' has no sub-questions."
                )
            for sq in criterion.sub_questions:
                if not sq.question.strip():
                    raise ValueError(
                        f"Sub-question '{sq.key}' has an empty question text."
                    )
