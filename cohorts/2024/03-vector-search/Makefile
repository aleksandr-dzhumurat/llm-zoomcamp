CURRENT_DIR = $(shell pwd)
DEV_ENV = $(shell docker ps --filter "name=dev-run" --format "{{.ID}}")
include .env
export

prepare-dirs:
	mkdir -p ${CURRENT_DIR}/models|| true

build:
	docker build -f Dockerfile -t vectoriser:latest .

train:
	docker run -it \
		--rm \
		-v ./models:/srv/models \
		-v ./src:/srv/src \
		--name vector_search \
		vectoriser:latest python3 src/train_embeds.py /srv

run:
	docker-compose run -d dev

eval:
	docker exec -it ${DEV_ENV} python3 src/train_embeds.py /srv

down:
	docker rm -f ${DEV_ENV}; docker-compose down

bash:
	docker exec -it ${DEV_ENV} bash