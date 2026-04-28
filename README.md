# MCTS QCM Reasoning Engine

An autonomous reasoning engine that uses **Monte Carlo Tree Search** (AlphaGo-style) where:

- The **policy network** is replaced by an LLM-driven **Idea Generator** that branches into `K` distinct child ideas.
- The **value network** is replaced by an LLM-driven **QCM Auditor** that answers a fixed 4-question checklist (Novelty, Resource, Feasibility, Alignment). The pass-rate `[0, 1]` is the value.
- **Selection** is classical UCB1 / UCT.
- **Backpropagation** averages pass-rate up the path.

The result is a *transparent*, audit-driven tree search instead of an opaque neural value head.

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # fill in at least one provider key
```

## Run

The CLI defaults both generator and auditor to **Gemini 2.5 Flash** (`gemini/gemini-2.5-flash`). Set `GEMINI_API_KEY` in `.env`, then:

```bash
mcts run "Design a low-cost desalination process for off-grid villages" \
    --iters 20 --k 4 \
    --out tree.json
```

To use OpenAI instead, pass explicit models:

```bash
mcts run "Your problem" --model-gen openai/gpt-4o-mini --model-audit openai/gpt-4o-mini
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
  ├─ Idea A   [3/4 ✓]
  │   ├─ A.1  [4/4 ✓✓]   <-- best path digs here
  │   └─ A.2  [2/4]
  ├─ Idea B   [1/4]      <-- pruned (failed Resource)
  └─ Idea C   [3/4 ✓]
```

Each iteration:

1. **Select** a leaf via UCB1 over pass-rate.
2. **Expand** with `K` LLM-proposed children (deduplicated by Jaccard).
3. **Evaluate** each child via the QCM Auditor → pass-rate.
4. **Backprop** pass-rate to the root.

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
| `--no-prune-resource` | —         | off                       | Do not auto-prune on failed Resource check. |
| `--prune-novelty`     | —         | off                       | Also prune on failed Novelty check. |
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
  node.py         Node dataclass + tree helpers
  scoring.py      ucb1(), pass_rate(), best_child()
  config.py       MCTSConfig (all knobs in one place)
  llm.py          Thin LiteLLM wrapper with retry + JSON parsing
  prompts.py      Prompt templates for generator + auditor
  generator.py    IdeaGenerator
  auditor.py      QCMAuditor with pydantic-validated output
  search.py       MCTS orchestrator (select / expand / evaluate / backprop)
  visualize.py    Rich tree print + JSON / Markdown / HTML export
  cli.py          Typer entrypoint (`mcts`, `mcts run`, `mcts visualize`, …)
tests/            Pytest suite
examples/         Sample problems
```
