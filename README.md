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
mcts visualize tree.json --no-open          # generate but don't open browser
mcts visualize tree.json --html-out out.html  # custom output path
```

## CLI flags

### `mcts run`

| Flag                  | Default                   | Description                                  |
| --------------------- | ------------------------- | -------------------------------------------- |
| `--iters`             | `20`                      | Number of MCTS iterations.                   |
| `--k`                 | `4`                       | Children per expansion.                      |
| `--c`                 | `1.41`                    | UCB1 exploration constant.                   |
| `--max-depth`         | `4`                       | Max tree depth.                              |
| `--max-nodes`         | `200`                     | Hard cap on total nodes.                     |
| `--model-gen`         | `gemini/gemini-2.5-flash` | Model for the Idea Generator.                |
| `--model-audit`       | `gemini/gemini-2.5-flash` | Model for the QCM Auditor.                   |
| `--out`               | `tree.json`               | Where to write the tree dump.                |
| `--md-out`            | —                         | Optional Markdown export of the tree.        |
| `--no-prune-resource` | off                       | Disable auto-pruning on failed Resource.     |
| `--prune-novelty`     | off                       | Also prune on failed Novelty.                |

### `mcts visualize`

| Flag           | Default                                | Description                             |
| -------------- | -------------------------------------- | --------------------------------------- |
| `--html-out`   | `<stem>-explorer.html` next to input   | Path for the self-contained HTML file.  |
| `--no-open`    | off                                    | Skip auto-opening the browser.          |
| `--canvas-out` | Cursor canvases dir                    | Path for the `.canvas.tsx` (best-effort). |

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
  cli.py          Typer entrypoint (mcts run ...)
tests/            Pytest suite
examples/         Sample problems
```
