SCRIPTS := scripts

.PHONY: kill load cload ollama users qdrant iqa ciqa is cis eval ceval pqa ps cps cs ccs dpd gen cgen

kill:
	bash $(SCRIPTS)/kill.sh

ollama:
	tmux attach -t ollama

qdrant:
	tmux attach -t qdrant

users:
	nvidia-smi --query-compute-apps=pid --format=csv,noheader | \
	xargs -I{} ps -o user= -p {} | sort | uniq

pqa:
	python -m qa_parser

ps:
	bash $(SCRIPTS)/processor.sh

cps:
	tmux attach -t processor

cs:
	bash $(SCRIPTS)/src-chunker.sh

ccs:
	tmux attach -t chunk

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
