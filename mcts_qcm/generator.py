"""Idea Generator — the LLM-driven policy of the MCTS engine.

Given a root problem and the current path, asks an LLM for K distinct child
ideas, deduplicates near-identical ones via Jaccard similarity over tokens,
and returns plain strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mcts_qcm.config import MCTSConfig
from mcts_qcm.llm import LLMClient, LiteLLMClient
from mcts_qcm.node import Node
from mcts_qcm.prompts import GENERATOR_SYSTEM, GENERATOR_USER, ROOT_GENERATOR_SYSTEM, ROOT_GENERATOR_USER

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 2}


def jaccard(a: str, b: str) -> float:
    """Jaccard similarity over lowercase alphanumeric tokens of length > 2."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def format_path(path_nodes: list[Node]) -> str:
    """Render a node path for the prompt: numbered list of parent ideas.

    `path_nodes` should be the chain from root to (and including) the node we
    are expanding. The root's idea is the original problem; we skip it and
    list intermediate parent ideas only.
    """
    if len(path_nodes) <= 1:
        return "(none — we are at the root)"
    parts = []
    for i, n in enumerate(path_nodes[1:], start=1):
        parts.append(f"  {i}. {n.idea}")
    return "\n".join(parts)


@dataclass
class IdeaGenerator:
    """Branches a node into K distinct, deduplicated child ideas."""

    config: MCTSConfig
    client: LLMClient | None = None
    similarity_threshold: float = 0.75

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = LiteLLMClient(max_retries=self.config.max_retries)

    def generate(self, *, problem: str, node: Node, k: int | None = None) -> list[str]:
        """Return up to ``k`` deduplicated, distinct child ideas for ``node``."""
        k = k or self.config.k_children
        if node.depth == 0:
            system = ROOT_GENERATOR_SYSTEM
            user = ROOT_GENERATOR_USER.format(problem=problem, k=k)
        else:
            path_str = format_path(node.path_from_root())
            system = GENERATOR_SYSTEM
            user = GENERATOR_USER.format(problem=problem, path=path_str, k=k)

        assert self.client is not None
        data = self.client.chat_json(
            model=self.config.model_gen,
            system=system,
            user=user,
            temperature=self.config.temperature_gen,
            seed=self.config.seed,
            timeout=self.config.request_timeout,
        )
        raw_ideas = data.get("ideas", []) if isinstance(data, dict) else []
        ideas: list[str] = []
        for entry in raw_ideas:
            if isinstance(entry, dict):
                text = str(entry.get("idea", "")).strip()
            else:
                text = str(entry).strip()
            if text:
                ideas.append(text)

        deduped = self._dedupe(ideas, existing=[c.idea for c in node.children])
        return deduped[:k]

    def _dedupe(self, ideas: list[str], existing: list[str]) -> list[str]:
        """Remove ideas too similar to each other or to existing siblings."""
        kept: list[str] = []
        all_seen = list(existing)
        for idea in ideas:
            if any(jaccard(idea, prev) >= self.similarity_threshold for prev in all_seen):
                continue
            kept.append(idea)
            all_seen.append(idea)
        return kept
