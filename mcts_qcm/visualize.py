"""Rendering helpers: rich-tree printing, JSON dump, Markdown export."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from mcts_qcm.node import Node
from mcts_qcm.scoring import greedy_best_path


def _truncate(text: str, n: int = 110) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _node_label(node: Node) -> Text:
    """Format a single node line for the rich tree."""
    label = Text()
    if node.parent is None:
        label.append("ROOT ", style="bold")
        label.append(_truncate(node.idea, 140))
        return label

    audit = node.audit
    if audit is not None:
        passed = audit.num_passed
        color = {0: "red", 1: "red", 2: "yellow", 3: "green", 4: "bright_green"}[passed]
        label.append(f"[{audit.summary()}] ", style=f"bold {color}")
    else:
        label.append("[unaudited] ", style="dim")

    if node.dead:
        label.append("(dead) ", style="bold red")

    label.append(f"v={node.visits} q={node.mean_value:.2f}  ", style="cyan")
    label.append(_truncate(node.idea, 130))
    return label


def render_tree(root: Node) -> Tree:
    """Build a `rich.tree.Tree` from the MCTS tree rooted at ``root``."""
    tree = Tree(_node_label(root))

    def _add(parent_widget: Tree, n: Node) -> None:
        for child in n.children:
            sub = parent_widget.add(_node_label(child))
            _add(sub, child)

    _add(tree, root)
    return tree


def print_summary(root: Node, console: Console | None = None) -> None:
    """Print the tree, the greedy best path, and a small stats block."""
    console = console or Console()
    console.print(Panel.fit("MCTS QCM Search Tree", style="bold magenta"))
    console.print(render_tree(root))

    path = greedy_best_path(root)
    console.print()
    console.print(Panel.fit("Greedy best path (root → leaf)", style="bold green"))
    for i, node in enumerate(path):
        prefix = "ROOT" if i == 0 else f"  {i}."
        audit_str = f" [{node.audit.summary()}]" if node.audit is not None else ""
        console.print(f"{prefix}{audit_str} {_truncate(node.idea, 200)}")

    descendants = root.iter_descendants()
    audited = [n for n in descendants if n.audit is not None]
    pruned = [n for n in descendants if n.dead]
    console.print()
    console.print(
        f"[bold]Stats:[/bold] {len(descendants)} nodes  •  "
        f"{len(audited)} audited  •  {len(pruned)} pruned  •  "
        f"root visits={root.visits}"
    )


def to_json(root: Node) -> str:
    """JSON dump of the entire tree (pretty-printed)."""
    return json.dumps(root.to_dict(), indent=2, ensure_ascii=False)


def write_json(root: Node, path: str | Path) -> Path:
    p = Path(path)
    p.write_text(to_json(root), encoding="utf-8")
    return p


def to_markdown(root: Node) -> str:
    """Markdown export of the tree (for sharing in PRs / docs)."""
    lines: list[str] = ["# MCTS QCM Search Tree", ""]

    def _walk(n: Node, depth: int) -> None:
        indent = "  " * depth
        if n.parent is None:
            lines.append(f"{indent}- **ROOT**: {n.idea}")
        else:
            audit_str = ""
            if n.audit is not None:
                audit_str = f" [{n.audit.summary()}]"
            dead_str = " _(pruned)_" if n.dead else ""
            lines.append(
                f"{indent}- {audit_str} v={n.visits} q={n.mean_value:.2f}{dead_str} — {n.idea}"
            )
        for child in n.children:
            _walk(child, depth + 1)

    _walk(root, 0)

    lines.append("")
    lines.append("## Greedy best path")
    for i, node in enumerate(greedy_best_path(root)):
        if i == 0:
            lines.append(f"- **ROOT**: {node.idea}")
        else:
            audit_str = f" [{node.audit.summary()}]" if node.audit is not None else ""
            lines.append(f"  {i}.{audit_str} {node.idea}")

    return "\n".join(lines)
