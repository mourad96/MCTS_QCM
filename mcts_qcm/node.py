"""Tree node for the MCTS QCM engine."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcts_qcm.auditor import AuditResult


_id_counter = itertools.count()


def _next_id() -> int:
    return next(_id_counter)


@dataclass
class Node:
    """A single idea in the MCTS tree.

    Attributes:
        idea: The natural-language idea this node represents. The root holds
            the user's original problem.
        parent: The parent node, or None for the root.
        children: List of child Nodes.
        visits: Number of times this node has been on a backpropagated path.
        value_sum: Sum of pass-rates seen through this node. The mean value
            is `value_sum / visits`.
        audit: The local QCM result for this idea (None for the unaudited root).
        depth: Depth in the tree (root = 0).
        dead: If True, this node has been pruned and will not be expanded.
        node_id: Process-unique numeric id, useful for visualizations and dumps.
    """

    idea: str
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    audit: AuditResult | None = None
    depth: int = 0
    dead: bool = False
    node_id: int = field(default_factory=_next_id)

    def add_child(self, idea: str) -> Node:
        """Create and attach a child node, return it."""
        child = Node(idea=idea, parent=self, depth=self.depth + 1)
        self.children.append(child)
        return child

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def mean_value(self) -> float:
        """Average pass-rate seen through this node, in [0, 1]."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def path_from_root(self) -> list[Node]:
        """Return [root, ..., self]."""
        chain: list[Node] = []
        cur: Node | None = self
        while cur is not None:
            chain.append(cur)
            cur = cur.parent
        chain.reverse()
        return chain

    def iter_descendants(self) -> list[Node]:
        """DFS over self and all descendants."""
        out = [self]
        stack = list(self.children)
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(n.children)
        return out

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation of this subtree."""
        return {
            "id": self.node_id,
            "idea": self.idea,
            "depth": self.depth,
            "visits": self.visits,
            "value_sum": round(self.value_sum, 4),
            "mean_value": round(self.mean_value, 4),
            "dead": self.dead,
            "audit": self.audit.model_dump() if self.audit is not None else None,
            "children": [c.to_dict() for c in self.children],
        }
