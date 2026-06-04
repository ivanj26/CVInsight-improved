docker-build:
	docker build --no-cache -t cvparser-service .

docker-run:
	docker run -d --name docker-cvparser -p 9001:9001 cvparser-service

docker-update:
	docker stop docker-cvparser && docker rm docker-cvparser && docker run -d --name docker-cvparser -p 9001:9001 cvparser-service

run-worker:
	python -m workers.docs_checker_worker
