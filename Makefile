SCRIPTS := bash-scripts

.PHONY: kill load cload ollama users qdrant iqa is

ollama:
	tmux attach -t ollama

qdrant:
	tmux attach -t qdrant

users:
	nvidia-smi --query-compute-apps=pid --format=csv,noheader | \
	xargs -I{} ps -o user= -p {} | sort | uniq