#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SESSION="chunk"

CHUNKER_SCRIPT="${1:-chunk_processor.py}"

# ─── Run Chunker ────────────────────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Tmux session '$SESSION' already exists."
    echo "Kill it first with:"
    echo "  tmux kill-session -t $SESSION"
    exit 1
fi

echo "Starting PDF Chunker in tmux session: $SESSION"

tmux new-session -d -s "$SESSION" \
    "cd '$RUNNER_DIR' && python '$CHUNKER_SCRIPT'; read"

echo ""
echo "All sessions launched."
echo ""
echo "Useful commands:"
echo "  tmux ls"
echo "  tmux attach -t $SESSION"
echo "  make ccs"