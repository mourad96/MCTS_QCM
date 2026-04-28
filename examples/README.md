# Example problems

Each file in this folder is a short prompt suitable for `mcts run`. Pick one,
copy the prompt, and run:

```bash
mcts run "$(cat examples/desalination.txt)" --iters 15 --k 3 --max-depth 3
```

(Default models are **Gemini 2.5 Flash** via `gemini/gemini-2.5-flash`; ensure `GEMINI_API_KEY`
is set in `.env`.)

```bash
mcts run "Design a low-cost desalination process for off-grid villages" \
    --iters 15 --k 3 --max-depth 3 \
    --md-out runs/desalination.md \
    --out runs/desalination.json
```

Explicit OpenAI fallback example:

```bash
mcts run "Your problem here" \
    --model-gen openai/gpt-4o-mini --model-audit openai/gpt-4o-mini \
    --md-out runs/out.md --out runs/out.json
```

## Tuning suggestions

- **Cheap exploration**: `--iters 10 --k 3 --max-depth 3` (~30–40 LLM calls).
- **Wider search**: `--iters 30 --k 5 --max-depth 4` (~150+ LLM calls).
- `--model-gen` can use a heavier model (`openai/gpt-4o`, `gemini/gemini-2.5-pro`) and
  `--model-audit` stay on Flash or `openai/gpt-4o-mini` to separate cost tiers.
