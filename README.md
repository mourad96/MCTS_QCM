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

```bash
mcts run "Design a low-cost desalination process for off-grid villages" \
    --iters 20 --k 4 \
    --model-gen openai/gpt-4o-mini \
    --model-audit openai/gpt-4o-mini \
    --out tree.json
```

LiteLLM auto-detects providers via the model prefix:

| Provider  | Example model string                       |
| --------- | ------------------------------------------ |
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

## CLI flags

| Flag             | Default                  | Description                                  |
| ---------------- | ------------------------ | -------------------------------------------- |
| `--iters`        | `20`                     | Number of MCTS iterations.                   |
| `--k`            | `4`                      | Children per expansion.                      |
| `--c`            | `1.41`                   | UCB1 exploration constant.                   |
| `--max-depth`    | `4`                      | Max tree depth.                              |
| `--max-nodes`    | `200`                    | Hard cap on total nodes.                     |
| `--model-gen`    | `openai/gpt-4o-mini`     | Model for the Idea Generator.                |
| `--model-audit`  | `openai/gpt-4o-mini`     | Model for the QCM Auditor.                   |
| `--out`          | `tree.json`              | Where to write the tree dump.                |

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
  visualize.py    Rich tree print + JSON / Markdown export
  cli.py          Typer entrypoint (mcts run ...)
tests/            Pytest suite
examples/         Sample problems
```
