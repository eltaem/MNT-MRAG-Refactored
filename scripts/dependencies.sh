SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─── Start dependencies ───────────────────────────────────────────────────────
bash "$SCRIPT_DIR/qdrant.sh"
bash "$SCRIPT_DIR/ollama.sh"