"""Central configuration for an MCTS run.

All knobs live here so the search algorithm and CLI can stay focused.
"""

from __future__ import annotations

from dataclasses import dataclass

# Stable Google Gemini 2.5 Flash ID on the Gemini API via LiteLLM (prefix `gemini/`).
DEFAULT_GEMINI_FLASH = "gemini/gemini-2.5-flash"


@dataclass
class MCTSConfig:
    """Configuration for a single MCTS search.

    Attributes:
        model_gen: LiteLLM model string for the Idea Generator (the "policy").
        model_audit: LiteLLM model string for the QCM Auditor (the "value").
        k_children: Number of children proposed at each expansion step.
        c_explore: UCB1 exploration constant. sqrt(2) ≈ 1.41 is the textbook default.
        iterations: Number of full select/expand/evaluate/backprop iterations.
        max_depth: Maximum tree depth (root has depth 0).
        max_nodes: Hard cap on total nodes in the tree (safety + cost guard).
        auto_qcm: If True, the rubric designer LLM generates a problem-specific
            rubric at the start of the search instead of using a file-based one.
        qcm_file: Optional path to a JSON file containing a pre-built Rubric
            (serialized via Rubric.to_dict). Ignored when auto_qcm is True.
        prune_threshold: Minimum weighted score (0.0–1.0) below which a node is
            marked dead and not expanded further. Also applies to axiomatic FAIL
            results (which set score to 0.0, always below any positive threshold).
        temperature_gen: Sampling temperature for the Idea Generator (higher =
            more diverse children).
        temperature_audit: Sampling temperature for the QCM Auditor (lower =
            more deterministic audits).
        request_timeout: Per-LLM-call timeout in seconds.
        max_retries: Per-LLM-call retry budget.
        seed: Optional seed forwarded to LiteLLM for reproducibility.
    """

    model_gen: str = DEFAULT_GEMINI_FLASH
    model_audit: str = DEFAULT_GEMINI_FLASH

    k_children: int = 4
    c_explore: float = 1.41
    iterations: int = 20
    max_depth: int = 4
    max_nodes: int = 200

    auto_qcm: bool = False
    qcm_file: str | None = None
    prune_threshold: float = 0.25

    temperature_gen: float = 0.9
    temperature_audit: float = 0.1
    request_timeout: float = 60.0
    max_retries: int = 2
    seed: int | None = None
