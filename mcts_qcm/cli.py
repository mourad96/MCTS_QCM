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
    rich_markup_mode="rich",
    help=(
        "AlphaGo-style tree search where an LLM generates ideas and a 4-question "
        "QCM audit replaces the neural value head.\n\n"
        "[bold]Quick start[/bold]\n\n"
        "  [cyan]mcts run[/cyan] [green]\"Your problem\"[/green] --out tree.json\n\n"
        "  [cyan]mcts visualize[/cyan] tree.json\n\n"
        "Run any sub-command with [bold]--help[/bold] for full flag details."
    ),
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
    problem: str = typer.Argument(..., help="The problem statement to reason about."),
    iters: int = typer.Option(
        20, "--iters", "-n",
        help="MCTS iterations to run. More iterations = wider and deeper tree. 20–40 is a good starting range.",
    ),
    k: int = typer.Option(
        4, "--k", "-k",
        help="Child ideas generated per expansion. Higher values explore more breadth per step.",
    ),
    c: float = typer.Option(
        1.41, "--c",
        help="UCB1 exploration constant. Higher values favour less-visited nodes; lower values exploit high-scoring ones.",
    ),
    max_depth: int = typer.Option(
        4, "--max-depth",
        help="Hard cap on tree depth. Nodes at this depth are never expanded further.",
    ),
    max_nodes: int = typer.Option(
        200, "--max-nodes",
        help="Hard cap on total node count. Acts as a safety guard against runaway expansion.",
    ),
    model_gen: str = typer.Option(
        DEFAULT_GEMINI_FLASH, "--model-gen",
        help=(
            "LiteLLM model string for the Idea Generator. "
            "Use the provider prefix, e.g. openai/gpt-4o-mini, anthropic/claude-3-5-sonnet-latest, "
            "groq/llama-3.1-70b-versatile, ollama/llama3."
        ),
    ),
    model_audit: str = typer.Option(
        DEFAULT_GEMINI_FLASH, "--model-audit",
        help=(
            "LiteLLM model string for the QCM Auditor. "
            "Can differ from --model-gen; a cheaper/faster model often works well here."
        ),
    ),
    temperature_gen: float = typer.Option(
        0.9, "--temp-gen",
        help="Sampling temperature for the Idea Generator. Higher = more creative/diverse ideas.",
    ),
    temperature_audit: float = typer.Option(
        0.1, "--temp-audit",
        help="Sampling temperature for the QCM Auditor. Keep low for consistent, deterministic scoring.",
    ),
    out: Path = typer.Option(
        Path("tree.json"), "--out", "-o",
        help="Path to write the JSON tree dump. Pass this file to 'mcts visualize' afterwards.",
    ),
    md_out: Path | None = typer.Option(
        None, "--md-out",
        help="If set, also write a Markdown export of the full tree to this path.",
    ),
    seed: int | None = typer.Option(
        None, "--seed",
        help="Integer seed forwarded to the LLM API for reproducible runs (not all providers honour it).",
    ),
    no_prune_resource: bool = typer.Option(
        False, "--no-prune-resource",
        help="Disable auto-pruning of nodes that fail the Resource check. By default such nodes are marked dead.",
    ),
    prune_novelty: bool = typer.Option(
        False, "--prune-novelty",
        help="Also prune nodes that fail the Novelty check (off by default — novelty failures are informational).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG-level logging."),
) -> None:
    """Run the MCTS QCM engine on [bold]PROBLEM[/bold] and write the search tree to a JSON file.

    [bold]Basic usage[/bold]

      mcts run "Design a low-cost desalination process for off-grid villages"

    [bold]More iterations, custom output, then visualize[/bold]

      mcts run "Your problem" --iters 40 --k 4 --out tree.json

      mcts visualize tree.json

    [bold]Use a different model provider[/bold]

      mcts run "Your problem" --model-gen openai/gpt-4o-mini --model-audit openai/gpt-4o-mini

    [bold]Keep a fast auditor, use a stronger generator[/bold]

      mcts run "Your problem" --model-gen anthropic/claude-3-5-sonnet-latest

    [dim]Supported providers (via LiteLLM): Google (gemini/...), OpenAI (openai/...),
    Anthropic (anthropic/...), Groq (groq/...), Ollama (ollama/...).
    Set the matching API key in .env before running.[/dim]
    """
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
        help="Path to a tree.json produced by 'mcts run'.",
    ),
    html_out: Path | None = typer.Option(
        None,
        "--html-out",
        "-o",
        help=(
            "Path for the self-contained HTML explorer. "
            "Defaults to <stem>-explorer.html in the same directory as the input file."
        ),
    ),
    no_open: bool = typer.Option(
        False,
        "--no-open",
        help="Write the HTML file but do not open it in the browser automatically.",
    ),
    canvas_out: Path | None = typer.Option(
        None,
        "--canvas-out",
        "-c",
        help=(
            "Path for the Cursor .canvas.tsx file (best-effort; requires a supported Cursor build). "
            "Defaults to the Cursor-managed canvases directory for this workspace."
        ),
    ),
) -> None:
    """Generate an interactive tree explorer from a [bold]tree.json[/bold] file.

    [bold]Basic usage[/bold]

      mcts visualize tree.json

    Opens a self-contained HTML file in your default browser — scrollable DAG,
    colour-coded audit badges, click-to-inspect detail panel. Re-run after every
    'mcts run' to refresh the view with fresh data.

    [bold]Generate without opening the browser[/bold]

      mcts visualize tree.json --no-open

    [bold]Custom output path[/bold]

      mcts visualize tree.json --html-out reports/my-run.html

    [dim]The HTML file has zero dependencies — no server, no network calls.
    Open it in any browser at any time.[/dim]
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
    """Print the installed package version."""
    from mcts_qcm import __version__

    console.print(__version__)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
