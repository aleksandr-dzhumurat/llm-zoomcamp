CURRENT_DIR = $(shell pwd)
NETWORK_NAME = service_network
include .env
export


prepare-dirs:
	mkdir -p ${CURRENT_DIR}/ollama_files|| true

run:
	docker run -it \
		--rm \
		-v ./ollama_files:/root/.ollama \
		-p 11434:11434 \
		--name ollama \
		ollama/ollama

run-gemma:
	docker run -it --rm -p 11434:11434 ollama-gemma2b