"""Typer-based CLI for the MCTS QCM engine.

Usage:

    mcts run "Design a low-cost desalination process" --iters 20 --k 4
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from mcts_qcm.config import DEFAULT_GEMINI_FLASH, MCTSConfig
from mcts_qcm.designer import QCMDesigner
from mcts_qcm.node import Node
from mcts_qcm.rubric import Rubric
from mcts_qcm.search import IterationResult, MCTS
from mcts_qcm.visualize import (
    generate_canvas,
    generate_html,
    print_summary,
    to_markdown,
    write_json,
    _canvas_default_out,
)

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    help=(
        "AlphaGo-style tree search where an LLM generates ideas and a tiered "
        "QCM audit replaces the neural value head.\n\n"
        "[bold]Quick start[/bold]\n\n"
        "  [cyan]mcts run[/cyan] [green]\"Your problem\"[/green] --auto-qcm --out tree.json\n\n"
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


# ---------------------------------------------------------------------------
# Rubric design helpers
# ---------------------------------------------------------------------------

def _display_rubric(rubric: Rubric, con: Console) -> None:
    """Pretty-print a Rubric as a Rich table for interactive review."""
    table = Table(
        title="Proposed Evaluation Rubric",
        show_lines=True,
        expand=True,
    )
    table.add_column("Criterion", style="bold cyan", no_wrap=True)
    table.add_column("Weight", justify="center", width=8)
    table.add_column("Sub-Question", style="white")
    table.add_column("Axiomatic", justify="center", width=10)

    for criterion in rubric.criteria:
        first = True
        for sq in criterion.sub_questions:
            crit_display = f"{criterion.name}\n[dim]{criterion.description}[/dim]" if first else ""
            weight_display = str(criterion.weight) if first else ""
            axio = "[bold red]YES[/bold red]" if sq.axiomatic else "[dim]no[/dim]"
            table.add_row(
                crit_display,
                weight_display,
                f"[bold]{sq.key}[/bold]\n{sq.question}",
                axio,
            )
            first = False

    con.print()
    con.print(table)
    con.print(
        f"\n[dim]Total criteria: {len(rubric.criteria)} | "
        f"Total sub-questions: {rubric.sub_question_count()} | "
        f"Axiomatic keys: {rubric.axiomatic_keys() or 'none'}[/dim]\n"
    )


def _edit_rubric_in_editor(rubric: Rubric) -> Rubric:
    """Dump rubric to a temp JSON file, open in $EDITOR, reload after close."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="mcts_rubric_", delete=False, encoding="utf-8",
    ) as f:
        json.dump(rubric.to_dict(), f, indent=2, ensure_ascii=False)
        tmp_path = f.name

    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "notepad"))
    try:
        subprocess.run([editor, tmp_path], check=True)
    except FileNotFoundError:
        console.print(f"[red]Editor '{editor}' not found. Set $EDITOR.[/red]")
        return rubric
    except subprocess.CalledProcessError:
        console.print("[yellow]Editor exited with an error; keeping original rubric.[/yellow]")
        return rubric

    try:
        data = json.loads(Path(tmp_path).read_text(encoding="utf-8"))
        edited = Rubric.from_dict(data)
        console.print("[green]Rubric updated from editor.[/green]")
        return edited
    except Exception as exc:
        console.print(f"[red]Failed to parse edited rubric: {exc}. Keeping original.[/red]")
        return rubric
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _design_rubric(problem: str, config: MCTSConfig, con: Console) -> Rubric:
    """Phase 0: design the evaluation rubric before MCTS starts."""
    # Load from file if provided
    if config.qcm_file:
        qcm_path = Path(config.qcm_file)
        if not qcm_path.exists():
            con.print(f"[red]QCM file not found:[/red] {qcm_path}")
            raise typer.Exit(1)
        try:
            data = json.loads(qcm_path.read_text(encoding="utf-8"))
            rubric = Rubric.from_dict(data)
            con.print(f"[bold green]Loaded rubric from:[/bold green] {qcm_path}")
            return rubric
        except Exception as exc:
            con.print(f"[red]Failed to parse rubric file: {exc}[/red]")
            raise typer.Exit(1) from exc

    # LLM proposes the rubric
    designer = QCMDesigner(config)
    con.print("[dim]Designing evaluation rubric for this problem…[/dim]")
    try:
        rubric = designer.propose(problem)
    except Exception as exc:
        con.print(f"[red]Rubric design failed: {exc}[/red]")
        raise typer.Exit(1) from exc

    if config.auto_qcm:
        con.print("[dim]Auto-QCM: using LLM-proposed rubric without review.[/dim]")
        _display_rubric(rubric, con)
        return rubric

    # Interactive review loop
    while True:
        _display_rubric(rubric, con)
        choice = Prompt.ask(
            "[bold]Accept this rubric?[/bold]",
            choices=["a", "e", "r"],
            default="a",
        )
        if choice == "a":
            break
        elif choice == "e":
            rubric = _edit_rubric_in_editor(rubric)
        elif choice == "r":
            con.print("[dim]Regenerating rubric…[/dim]")
            try:
                rubric = designer.propose(problem)
            except Exception as exc:
                con.print(f"[yellow]Regeneration failed: {exc}. Keeping current rubric.[/yellow]")

    return rubric


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

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
    auto_qcm: bool = typer.Option(
        False, "--auto-qcm",
        help="Skip interactive rubric approval and use the LLM-proposed rubric as-is.",
    ),
    qcm_file: Path | None = typer.Option(
        None, "--qcm-file",
        help="Load a pre-authored rubric from a JSON file (skips designer + interactive prompts).",
    ),
    prune_threshold: float = typer.Option(
        0.25, "--prune-threshold",
        help="Weighted score below this value triggers node pruning. Default 0.25.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG-level logging."),
) -> None:
    """Run the MCTS QCM engine on [bold]PROBLEM[/bold] and write the search tree to a JSON file.

    [bold]Basic usage (interactive rubric design)[/bold]

      mcts run "Design a low-cost desalination process for off-grid villages"

    [bold]Auto-QCM (skip rubric review)[/bold]

      mcts run "Your problem" --auto-qcm --iters 40 --out tree.json

    [bold]Reuse a saved rubric[/bold]

      mcts run "Your problem" --qcm-file .mcts_rubric.json

    [bold]Then visualize[/bold]

      mcts visualize tree.json

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
        auto_qcm=auto_qcm,
        qcm_file=str(qcm_file) if qcm_file else None,
        prune_threshold=prune_threshold,
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

    # Phase 0: Design the evaluation rubric
    rubric = _design_rubric(problem, config, console)

    def _on_iter(result: IterationResult, root: Node) -> None:
        new_audits = ", ".join(
            child.audit.summary() if child.audit else "?" for child in result.new_children
        )
        console.print(
            f"[cyan]iter {result.iteration:>3}[/cyan]  "
            f"selected: [white]{result.selected_idea[:80]}[/white]  "
            f"new: [{new_audits}]"
        )

    engine = MCTS(config=config, rubric=rubric, on_iteration=_on_iter)
    try:
        root = engine.run(problem)
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted by user — printing partial tree…[/yellow]")
        sys.exit(130)

    console.print()
    print_summary(root, console=console)

    # Write tree.json with embedded rubric
    out_data = {"rubric": rubric.to_dict(), "tree": root.to_dict()}
    out_path = Path(out)
    out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[bold green]Tree written to:[/bold green] {out_path.resolve()}")

    if md_out is not None:
        md_path = Path(md_out)
        md_path.write_text(to_markdown(root), encoding="utf-8")
        console.print(f"[bold green]Markdown written to:[/bold green] {md_path.resolve()}")

    # Save rubric for reuse
    rubric_path = Path(".mcts_rubric.json")
    rubric_path.write_text(
        json.dumps(rubric.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8",
    )
    console.print(
        f"\n[dim]Your custom QCM rubric has been saved to:[/dim] {rubric_path}\n"
        f"[dim]To bypass prompts and re-run this exact rubric later, use:[/dim]\n"
        f"  mcts run \"{problem}\" --qcm-file {rubric_path}"
    )


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
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"[red]Failed to read {json_path}:[/red] {exc}")
        raise typer.Exit(1)

    # Support both new format {"rubric": ..., "tree": ...} and raw tree dict
    if isinstance(raw, dict) and "tree" in raw:
        tree_data = raw["tree"]
    else:
        tree_data = raw

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
