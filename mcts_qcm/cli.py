"""Typer-based CLI for the MCTS QCM engine.

Usage:

    mcts run "Design a low-cost desalination process" --iters 20 --k 4
"""

from __future__ import annotations

import json
import logging
import sys
import webbrowser
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from mcts_qcm.config import DEFAULT_GEMINI_FLASH, MCTSConfig
from mcts_qcm.search import IterationResult, MCTS
from mcts_qcm.node import Node
from mcts_qcm.visualize import print_summary, write_json, to_markdown, generate_canvas, generate_html, _canvas_default_out

app = typer.Typer(
    add_completion=False,
    help="MCTS reasoning engine with LLM-driven QCM auditing (AlphaGo-style).",
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("litellm").setLevel(logging.WARNING)


@app.command()
def run(
    problem: str = typer.Argument(..., help="The problem to reason about."),
    iters: int = typer.Option(20, "--iters", "-n", help="Number of MCTS iterations."),
    k: int = typer.Option(4, "--k", "-k", help="Children per expansion."),
    c: float = typer.Option(1.41, "--c", help="UCB1 exploration constant."),
    max_depth: int = typer.Option(4, "--max-depth", help="Maximum tree depth."),
    max_nodes: int = typer.Option(200, "--max-nodes", help="Hard cap on total nodes."),
    model_gen: str = typer.Option(
        DEFAULT_GEMINI_FLASH, "--model-gen", help="LiteLLM model for the Idea Generator."
    ),
    model_audit: str = typer.Option(
        DEFAULT_GEMINI_FLASH, "--model-audit", help="LiteLLM model for the QCM Auditor."
    ),
    temperature_gen: float = typer.Option(0.9, "--temp-gen", help="Generator temperature."),
    temperature_audit: float = typer.Option(0.1, "--temp-audit", help="Auditor temperature."),
    out: Path = typer.Option(Path("tree.json"), "--out", "-o", help="Where to write the tree dump."),
    md_out: Path | None = typer.Option(
        None, "--md-out", help="Optional Markdown export of the tree."
    ),
    seed: int | None = typer.Option(None, "--seed", help="LLM seed for reproducibility."),
    no_prune_resource: bool = typer.Option(
        False, "--no-prune-resource", help="Disable auto-pruning on failed Resource check."
    ),
    prune_novelty: bool = typer.Option(
        False, "--prune-novelty", help="Also auto-prune on failed Novelty check."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    """Run the MCTS QCM engine on PROBLEM."""
    load_dotenv(override=False)
    _setup_logging(verbose)

    config = MCTSConfig(
        model_gen=model_gen,
        model_audit=model_audit,
        k_children=k,
        c_explore=c,
        iterations=iters,
        max_depth=max_depth,
        max_nodes=max_nodes,
        temperature_gen=temperature_gen,
        temperature_audit=temperature_audit,
        seed=seed,
        prune_on_failed_resource=not no_prune_resource,
        prune_on_failed_novelty=prune_novelty,
    )

    console.print(
        Panel.fit(
            f"[bold]Problem:[/bold] {problem}\n"
            f"[dim]model_gen={config.model_gen}  model_audit={config.model_audit}  "
            f"iters={config.iterations}  k={config.k_children}  "
            f"c={config.c_explore}  max_depth={config.max_depth}[/dim]",
            title="MCTS QCM Engine",
            style="bold magenta",
        )
    )

    def _on_iter(result: IterationResult, root: Node) -> None:
        new_audits = ", ".join(
            child.audit.summary() if child.audit else "?" for child in result.new_children
        )
        console.print(
            f"[cyan]iter {result.iteration:>3}[/cyan]  "
            f"selected: [white]{result.selected_idea[:80]}[/white]  "
            f"new: [{new_audits}]"
        )

    engine = MCTS(config=config, on_iteration=_on_iter)
    try:
        root = engine.run(problem)
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted by user — printing partial tree…[/yellow]")
        sys.exit(130)

    console.print()
    print_summary(root, console=console)

    out_path = write_json(root, out)
    console.print(f"\n[bold green]Tree written to:[/bold green] {out_path.resolve()}")
    if md_out is not None:
        md_path = Path(md_out)
        md_path.write_text(to_markdown(root), encoding="utf-8")
        console.print(f"[bold green]Markdown written to:[/bold green] {md_path.resolve()}")


@app.command()
def visualize(
    json_path: Path = typer.Argument(
        Path("tree.json"),
        help="Path to a tree.json produced by `mcts run`.",
    ),
    html_out: Path | None = typer.Option(
        None,
        "--html-out",
        "-o",
        help=(
            "Where to write the self-contained HTML explorer. "
            "Defaults to <stem>-explorer.html next to the input file."
        ),
    ),
    no_open: bool = typer.Option(
        False,
        "--no-open",
        help="Do not automatically open the HTML file in the browser.",
    ),
    canvas_out: Path | None = typer.Option(
        None,
        "--canvas-out",
        "-c",
        help=(
            "Where to write the .canvas.tsx file. "
            "Defaults to the Cursor-managed canvases directory for this workspace."
        ),
    ),
) -> None:
    """Generate an interactive HTML tree explorer from a tree.json file.

    Run this after `mcts run` to get an interactive DAG tree visualization with
    a click-to-inspect detail panel that opens in your browser.

    Example workflow:

        mcts run "Your problem" --out tree.json

        mcts visualize tree.json
    """
    json_path = Path(json_path)
    if not json_path.exists():
        console.print(f"[red]File not found:[/red] {json_path}")
        raise typer.Exit(1)

    try:
        tree_data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"[red]Failed to read {json_path}:[/red] {exc}")
        raise typer.Exit(1)

    # ── HTML explorer (primary output) ────────────────────────────────────────
    if html_out is None:
        stem = json_path.stem
        html_out = json_path.parent / f"{stem}-explorer.html"

    html_path = generate_html(tree_data, html_out)
    console.print(f"\n[bold green]HTML explorer written to:[/bold green] {html_path}")

    if not no_open:
        webbrowser.open(html_path.as_uri())
        console.print("[dim]Opening in your default browser…[/dim]")
    else:
        console.print("[dim]Open the file above in any browser to view the interactive tree.[/dim]")

    # ── Cursor Canvas (best-effort; requires a supported Cursor build) ─────────
    if canvas_out is None:
        canvas_dir = _canvas_default_out(Path.cwd())
        canvas_out = canvas_dir / "mcts-tree-explorer.canvas.tsx"

    try:
        canvas_path = generate_canvas(tree_data, canvas_out)
        console.print(f"[dim]Canvas also written to: {canvas_path}[/dim]")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[dim yellow]Canvas generation skipped: {exc}[/dim yellow]")


@app.command()
def version() -> None:
    """Print the package version."""
    from mcts_qcm import __version__

    console.print(__version__)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
