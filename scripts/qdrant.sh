#!/bin/bash

set -e

SESSION="qdrant"
QDRANT_BIN="${QDRANT_BIN:-./qdrant}"
RUNNER_DIR="$(cd "$(dirname "$0")/.." && pwd)"

wait_for_qdrant() {
    echo "Waiting for Qdrant to be ready..."

    for i in $(seq 1 20); do
        if curl -fs http://localhost:6333/healthz >/dev/null 2>&1; then
            echo "Qdrant is ready."
            return 0
        fi
        sleep 2
    done

    echo "Qdrant did not respond after 40 seconds."

    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo ""
        echo "Qdrant logs:"
        tmux capture-pane -pt "$SESSION"
    fi

    exit 1
}

# ──────────────────────────────────────────────────────────────
# Start Qdrant if not already running
# ──────────────────────────────────────────────────────────────

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Qdrant tmux session already exists."
else
    echo "Starting Qdrant..."

    tmux new-session -d -s "$SESSION" \
        "cd '$RUNNER_DIR' && exec '$QDRANT_BIN'"
fi

wait_for_qdrant