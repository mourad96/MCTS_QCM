# MCTS QCM Reasoning Engine

An autonomous reasoning engine that uses **Monte Carlo Tree Search** (AlphaGo-style) where:

- The **policy network** is replaced by an LLM-driven **Idea Generator** equipped with **Root Paradigm Brainstorming** (depth 0 switches to a high-creativity brainstorming mode to explore diverse, distinct domains) and **Anti-Analysis Hard Constraints** (forbidding purely academic planning/paper-writing steps in favor of concrete, physical engineering designs).
- The **value network** is replaced by a **Hybrid Tiered QCM Auditor** where:
  1. A **QCM Designer** proposes 4–6 domain-specific criteria (each with 2–3 atomic sub-questions) customized for your exact problem.
  2. A **QCMAuditor** strictly classifies each sub-question into categorical tiers: `STRONG`, `ADEQUATE`, `WEAK`, or `FAIL`.
  3. Python maps these tiers deterministically to values: `STRONG=1.0`, `ADEQUATE=0.66`, `WEAK=0.33`, `FAIL=0.0`.
  4. The overall score is a weighted average of criteria scores.
- **Selection** is classical UCB1 / UCT.
- **Pruning** is dual-mode: **Axiomatic** (immediate pruning if any non-negotiable sub-question scores `FAIL`) and **Threshold-based** (pruning if overall score is below a minimum, e.g., `0.25`).
- **Backpropagation** averages the weighted QCM score up the path.

The result is a *transparent*, audit-driven tree search instead of an opaque neural value head.

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # fill in at least one provider key
```

## Run

The CLI defaults both generator and auditor to **Gemini 2.5 Flash** (`gemini/gemini-2.5-flash`). Set `GEMINI_API_KEY` in `.env`.

When running from scratch, the system will enter **Interactive Rubric Design** (proposing 4–6 custom criteria with sub-questions, allowing you to `[a]ccept`, `[e]dit`, or `[r]egenerate` it):

```bash
mcts run "Design a low-cost desalination process for off-grid villages" \
    --iters 20 --k 4 \
    --out tree.json
```

To run instantly in **non-interactive mode**, skipping review and accepting the LLM's proposed rubric as-is:

```bash
mcts run "Your problem" --auto-qcm --iters 20
```

To **reuse a saved rubric** (skips interactive prompts and design phase entirely):

```bash
mcts run "Your problem" --qcm-file .mcts_rubric.json --iters 20
```

LiteLLM auto-detects providers via the model prefix:

| Provider  | Example model string                       |
| --------- | ------------------------------------------ |
| Google    | **`gemini/gemini-2.5-flash`** (default), `gemini/gemini-flash-latest` |
| OpenAI    | `openai/gpt-4o-mini`, `openai/gpt-4o`      |
| Anthropic | `anthropic/claude-3-5-sonnet-latest`       |
| Groq      | `groq/llama-3.1-70b-versatile`             |
| Ollama    | `ollama/llama3`, `ollama/qwen2.5:14b`      |

## CLI help

Built-in help includes copy-paste examples, shorthand flags, and long descriptions for every option:

```bash
mcts --help
mcts run --help
mcts visualize --help
mcts version --help
```

## How it works

```
Root: user problem
  ├─ Idea A   [7S 3A 1W 1F]
  │   ├─ A.1  [10S 2A 0W 0F]   <-- best path digs here
  │   └─ A.2  [4S 4A 4W 0F]
  ├─ Idea B   [2S 2A 2W 2F]    <-- pruned (failed axiomatic sub-question)
  └─ Idea C   [6S 4A 2W 0F]
```

Each iteration:

1. **Select** a leaf via UCB1 over the weighted average QCM score.
2. **Expand** with `K` LLM-proposed children (deduplicated by Jaccard):
   - At the root (depth 0), it performs **Root Paradigm Brainstorming** to ensure diverse starter domains.
   - Forbids purely analytical or academic planning steps (e.g. financial modeling) via **Anti-Analysis Hard Constraints**.
3. **Evaluate** each child by auditing it against the dynamic rubric (classifies each sub-question into `STRONG`, `ADEQUATE`, `WEAK`, or `FAIL` tiers).
4. **Backprop** overall weighted score to the root.

After all iterations, the engine prints the **greedy best path** plus the full tree, and writes a JSON dump of the tree.

## Visualize

After a run, generate an interactive tree explorer and open it in your browser:

```bash
mcts visualize tree.json
```

The HTML file is fully self-contained (no server, no dependencies). Re-run it after every `mcts run` to refresh the view. The explorer shows:

- Scrollable DAG with nodes coloured by audit result and best path highlighted
- Click any node to see its full idea text, QCM audit breakdown, and stats

```bash
mcts visualize tree.json --no-open
mcts visualize tree.json --html-out out.html   # or: -o out.html
```

The command also writes a **Cursor Canvas** `.canvas.tsx` under the IDE-managed canvases folder for the current working directory when generation succeeds (live preview requires a Cursor build with Canvas support). Override with `--canvas-out` / `-c`.

## CLI flags

Summary below; for full detail see `mcts run --help` and `mcts visualize --help`.

### `mcts run`

| Flag                  | Shorthand | Default                   | Description |
| --------------------- | --------- | ------------------------- | ----------- |
| `PROBLEM`             | —         | —                         | Problem statement (required positional argument). |
| `--iters`             | `-n`      | `20`                      | MCTS iterations. |
| `--k`                 | `-k`      | `4`                       | Child ideas per expansion. |
| `--c`                 | —         | `1.41`                    | UCB1 exploration constant. |
| `--max-depth`         | —         | `4`                       | Maximum tree depth. |
| `--max-nodes`         | —         | `200`                     | Maximum total nodes. |
| `--model-gen`         | —         | `gemini/gemini-2.5-flash` | LiteLLM model for the Idea Generator. |
| `--model-audit`       | —         | `gemini/gemini-2.5-flash` | LiteLLM model for the QCM Auditor. |
| `--temp-gen`          | —         | `0.9`                     | Generator sampling temperature. |
| `--temp-audit`        | —         | `0.1`                     | Auditor sampling temperature. |
| `--out`               | `-o`      | `tree.json`               | JSON tree output path. |
| `--md-out`            | —         | —                         | Optional Markdown export path. |
| `--seed`              | —         | —                         | Optional LLM seed (provider-dependent). |
| `--auto-qcm`          | —         | off                       | Skip interactive rubric design and accept LLM proposal as-is. |
| `--qcm-file`          | —         | —                         | Load a pre-authored rubric from a JSON file (skips design phase). |
| `--prune-threshold`   | —         | `0.25`                    | Weighted score below this → prune the node (0.0 to 1.0). |
| `--verbose`           | `-v`      | off                       | DEBUG logging. |

### `mcts visualize`

| Flag           | Shorthand | Default                              | Description |
| -------------- | --------- | ------------------------------------ | ----------- |
| `JSON_PATH`    | —         | `tree.json`                          | Input tree from `mcts run`. |
| `--html-out`   | `-o`      | `<stem>-explorer.html` beside input  | Self-contained HTML explorer path. |
| `--no-open`    | —         | off                                  | Write HTML only; do not open a browser tab. |
| `--canvas-out` | `-c`      | Cursor `canvases/` for cwd           | Optional `.canvas.tsx` path (best-effort). |

### `mcts version`

Prints the installed package version.

## Tests

```bash
pytest
```

The mocked-LLM tests cover UCB1, pass-rate, JSON parsing, and a full end-to-end run with a fake LLM client.

## Project layout

```
mcts_qcm/
  rubric.py       Rubric, Criterion, SubQuestion data models
  designer.py     QCMDesigner rubric proposer
  node.py         Node dataclass + tree helpers
  scoring.py      ucb1(), compute_score(), best_child()
  config.py       MCTSConfig (all knobs in one place)
  llm.py          Thin LiteLLM wrapper with retry + JSON parsing
  prompts.py      Prompt templates for generator, designer, and auditor
  generator.py    IdeaGenerator with Root Paradigm Brainstorming
  auditor.py      QCMAuditor with Pydantic-validated tiered output
  search.py       MCTS orchestrator (select / expand / evaluate / backprop)
  visualize.py    Rich tree print + JSON / Markdown / HTML export
  cli.py          Typer entrypoint (`mcts`, `mcts run`, `mcts visualize`, …)
tests/            Pytest suite
examples/         Sample problems
```
