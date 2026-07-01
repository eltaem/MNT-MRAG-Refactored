#!/bin/bash

SESSION="ollama"
OLLAMA_BIN="${OLLAMA_BIN:-~/ollama/bin/ollama}"
# Set your desired context length variable here (e.g., 16384 or 32768)
CONTEXT_LENGTH=""

# ─── Context Length Logic ────────────────────────────────────────────────────
# If CONTEXT_LENGTH is set and non-empty, use it
# Otherwise, leave it unset so Ollama uses model default (usually 2048 or model-specific)
if [ -n "$CONTEXT_LENGTH" ]; then
  echo "Using custom context length: $CONTEXT_LENGTH"
  OLLAMA_ENV="OLLAMA_CONTEXT_LENGTH=$CONTEXT_LENGTH"
else
  echo "Using Ollama default context length"
  OLLAMA_ENV=""  # Empty = no override, Ollama uses default
fi

wait_for_ollama() {
  echo "Waiting for Ollama to be ready..."
  for i in $(seq 1 20); do
    if curl -s http://localhost:11434 >/dev/null 2>&1; then
      echo "Ollama is ready."
      return 0
    fi
    sleep 2
  done
  echo "Ollama did not respond after 40s. Aborting."
  exit 1
}

# ─── Kill stray Ollama processes outside tmux ─────────────────────────────────
echo "Checking for Ollama processes running outside tmux..."

PIDS=$(pgrep -f "ollama serve" 2>/dev/null)

if [ -n "$PIDS" ]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "  Ollama already running inside tmux session '$SESSION'. Leaving it."
  else
    echo "  Found Ollama outside tmux (PIDs: $PIDS). Killing..."
    kill -TERM $PIDS
    sleep 2
    kill -KILL $PIDS 2>/dev/null
    sleep 2
    echo "  Killed stray Ollama process(es)."
  fi
fi

# ─── Start Ollama in tmux if not already running ──────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "  Tmux session '$SESSION' already exists. Skipping."
else
  if [ -n "$OLLAMA_ENV" ]; then
    echo "Starting Ollama in tmux session: $SESSION with context limit: $CONTEXT_LENGTH"
    tmux new-session -d -s "$SESSION" "env $OLLAMA_ENV $OLLAMA_BIN serve"
  else
    echo "Starting Ollama in tmux session: $SESSION with default context"
    tmux new-session -d -s "$SESSION" "$OLLAMA_BIN serve"
  fi
  wait_for_ollama
fi