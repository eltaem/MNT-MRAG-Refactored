#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SESSION="index-src"

INDEXER_SCRIPT="${1:-src_indexer.py}"

# ─── Start dependencies ───────────────────────────────────────────────────────
bash "$SCRIPT_DIR/qdrant.sh"

# ─── Run Source Indexer ───────────────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Tmux session '$SESSION' already exists."
    echo "Kill it first with:"
    echo "  tmux kill-session -t $SESSION"
    exit 1
fi

echo "Starting Source Indexer in tmux session: $SESSION"

tmux new-session -d -s "$SESSION" \
    "cd '$RUNNER_DIR' && python '$INDEXER_SCRIPT'; read"

echo ""
echo "All sessions launched."
echo ""
echo "Useful commands:"
echo "  tmux ls"
echo "  tmux attach -t qdrant"
echo "  tmux attach -t $SESSION"
echo "  make qdrant"
echo "  make cis"