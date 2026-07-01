#!/bin/bash
# Kill only eval-* index-* and qdrant sessions
for session in $(tmux ls | grep -E 'pdf-|ollama|index|qdrant|eval' | cut -d: -f1); do
  echo "Killing $session ..."
  tmux kill-session -t "$session"
done