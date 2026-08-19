#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SESSION="eval"

EVALUATOR_SCRIPT="${1:-evaluate.py}"

# ─── Start dependencies ───────────────────────────────────────────────────────
bash "$SCRIPT_DIR/dependencies.sh"

# ─── Run Evaluator ────────────────────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Tmux session '$SESSION' already exists."
    echo "Kill it first with:"
    echo "  tmux kill-session -t $SESSION"
    exit 1
fi

echo "Starting Evaluator in tmux session: $SESSION"

tmux new-session -d -s "$SESSION" \
    "cd '$RUNNER_DIR' && python '$EVALUATOR_SCRIPT'; read"

echo ""
echo "All sessions launched."
echo ""
echo "Useful commands:"
echo "  tmux ls"
echo "  tmux attach -t qdrant"
echo "  tmux attach -t ollama"
echo "  tmux attach -t $SESSION"
echo "  make qdrant"
echo "  make ollama"
echo "  make ceval"