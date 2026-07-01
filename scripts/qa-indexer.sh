#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SESSION="index-qa"

INDEXER_MODULE="${1:-src.qa_indexer}"

# ─── Start dependencies ───────────────────────────────────────────────────────
bash "$SCRIPT_DIR/qdrant.sh"

# ─── Run QA Indexer ───────────────────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Tmux session '$SESSION' already exists."
    echo "Kill it first with:"
    echo "  tmux kill-session -t $SESSION"
    exit 1
fi

echo "Starting QA Indexer in tmux session: $SESSION"

tmux new-session -d -s "$SESSION" \
    "cd '$RUNNER_DIR' && python -m $INDEXER_MODULE; read"

echo ""
echo "All sessions launched."
echo ""
echo "Useful commands:"
echo "  tmux ls"
echo "  tmux attach -t qdrant"
echo "  tmux attach -t $SESSION"