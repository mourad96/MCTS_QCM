# Example problems

Each file in this folder is a short prompt suitable for `mcts run`. Pick one,
copy the prompt, and run:

```bash
mcts run "$(cat examples/desalination.txt)" --iters 15 --k 3 --max-depth 3
```

Or paste the problem directly:

```bash
mcts run "Design a low-cost desalination process for off-grid villages" \
    --iters 15 --k 3 --max-depth 3 \
    --model-gen openai/gpt-4o-mini \
    --model-audit openai/gpt-4o-mini \
    --md-out runs/desalination.md \
    --out runs/desalination.json
```

## Tuning suggestions

- **Cheap exploration**: `--iters 10 --k 3 --max-depth 3` (~30–40 LLM calls).
- **Wider search**: `--iters 30 --k 5 --max-depth 4` (~150+ LLM calls).
- Use a stronger model for `--model-gen` and a cheaper one for `--model-audit`
  to control cost without losing audit signal.
