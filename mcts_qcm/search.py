"""MCTS orchestrator: select / expand / evaluate / backprop.

This is the heart of the engine. The four phases follow classical UCT, but the
"simulation/rollout" step is replaced by a single LLM-driven QCM audit (the
value head of the system, AlphaZero-style).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from mcts_qcm.auditor import QCMAuditor, QCMResult
from mcts_qcm.config import MCTSConfig
from mcts_qcm.generator import IdeaGenerator
from mcts_qcm.llm import LLMClient
from mcts_qcm.node import Node
from mcts_qcm.scoring import pass_rate, select_best_child

logger = logging.getLogger(__name__)


@dataclass
class IterationResult:
    """A summary of one MCTS iteration, useful for logging and visualization."""

    iteration: int
    selected_idea: str
    new_children: list[Node] = field(default_factory=list)


@dataclass
class MCTS:
    """Run an LLM-driven MCTS search over ideas."""

    config: MCTSConfig
    generator: IdeaGenerator | None = None
    auditor: QCMAuditor | None = None
    on_iteration: Callable[[IterationResult, Node], None] | None = None

    def __post_init__(self) -> None:
        if self.generator is None:
            self.generator = IdeaGenerator(self.config)
        if self.auditor is None:
            self.auditor = QCMAuditor(self.config)

    @classmethod
    def with_client(
        cls,
        config: MCTSConfig,
        client: LLMClient,
        *,
        on_iteration: Callable[[IterationResult, Node], None] | None = None,
    ) -> MCTS:
        """Convenience constructor that shares one LLM client between gen & audit."""
        return cls(
            config=config,
            generator=IdeaGenerator(config, client=client),
            auditor=QCMAuditor(config, client=client),
            on_iteration=on_iteration,
        )

    def run(self, problem: str) -> Node:
        """Run the full MCTS search and return the root node of the populated tree."""
        root = Node(idea=problem, depth=0)
        weights = self.config.weights

        for i in range(1, self.config.iterations + 1):
            if self._node_count(root) >= self.config.max_nodes:
                logger.info("Reached max_nodes=%d, stopping.", self.config.max_nodes)
                break

            leaf = self._select(root)
            new_children = self._expand_and_evaluate(problem, leaf, weights)

            if not new_children:
                self._backprop(leaf, leaf.mean_value if leaf.visits else 0.0)
            else:
                for child in new_children:
                    self._backprop(child, pass_rate(child.audit, weights))  # type: ignore[arg-type]

            result = IterationResult(
                iteration=i, selected_idea=leaf.idea, new_children=new_children
            )
            logger.debug("Iter %d: selected=%r → %d new children", i, leaf.idea, len(new_children))
            if self.on_iteration is not None:
                self.on_iteration(result, root)

        return root

    def _select(self, root: Node) -> Node:
        """Walk down using UCB1 until we hit an expandable leaf."""
        node = root
        while True:
            if node.dead:
                return node
            if node.depth >= self.config.max_depth:
                return node
            if node.is_leaf:
                return node
            chosen = select_best_child(node, self.config.c_explore)
            if chosen is None:
                return node
            node = chosen

    def _expand_and_evaluate(
        self,
        problem: str,
        leaf: Node,
        weights: dict[str, float],
    ) -> list[Node]:
        """Generate K children, audit each, attach as live/dead nodes."""
        if leaf.dead or leaf.depth >= self.config.max_depth:
            return []

        budget = max(0, self.config.max_nodes - self._node_count(self._root_of(leaf)))
        if budget == 0:
            return []
        k = min(self.config.k_children, budget)

        assert self.generator is not None and self.auditor is not None
        try:
            ideas = self.generator.generate(problem=problem, node=leaf, k=k)
        except Exception:  # noqa: BLE001
            logger.exception("Idea generation failed at depth=%d", leaf.depth)
            return []

        new_children: list[Node] = []
        for idea in ideas:
            child = leaf.add_child(idea)
            try:
                audit = self.auditor.audit(problem=problem, node=child)
            except Exception:  # noqa: BLE001
                logger.exception("Auditor failed for idea: %s", idea[:80])
                audit = QCMResult(
                    novelty=False, resource=False, feasibility=False, alignment=False,
                    reasons={"error": "Audit raised an unexpected exception."},
                )
            child.audit = audit

            if self.config.prune_on_failed_resource and not audit.resource:
                child.dead = True
            if self.config.prune_on_failed_novelty and not audit.novelty:
                child.dead = True

            new_children.append(child)
        return new_children

    @staticmethod
    def _backprop(node: Node, value: float) -> None:
        """Propagate `value` (a pass-rate in [0,1]) up to the root."""
        cur: Node | None = node
        while cur is not None:
            cur.visits += 1
            cur.value_sum += value
            cur = cur.parent

    @staticmethod
    def _node_count(root: Node) -> int:
        return len(root.iter_descendants())

    @staticmethod
    def _root_of(node: Node) -> Node:
        cur = node
        while cur.parent is not None:
            cur = cur.parent
        return cur
