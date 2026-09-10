SCRIPTS := scripts

.PHONY: kill load cload ollama users qdrant iqa ciqa is cis eval ceval parse dpd gen cgen

kill:
	bash $(SCRIPTS)/kill.sh

ollama:
	tmux attach -t ollama

qdrant:
	tmux attach -t qdrant

users:
	nvidia-smi --query-compute-apps=pid --format=csv,noheader | \
	xargs -I{} ps -o user= -p {} | sort | uniq

parse:
	python -m src.qa_parser

iqa:
	bash $(SCRIPTS)/qa-indexer.sh

ciqa:
	tmux attach -t index-qa

is:
	bash $(SCRIPTS)/src-indexer.sh

cis:
	tmux attach -t index-src

eval:
	bash $(SCRIPTS)/evaluator.sh

ceval:
	tmux attach -t eval

gen:
	bash $(SCRIPTS)/generator.sh

cgen:
	tmux attach -t gen

dpd:
	bash $(SCRIPTS)/dependencies.sh
