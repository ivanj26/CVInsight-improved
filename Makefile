docker-build:
	docker build --no-cache -t cvparser-service .

docker-run:
	docker run -d --name docker-cvparser -p 9001:9001 --network host cvparser-service

docker-update:
	docker stop docker-cvparser && docker rm docker-cvparser && docker run -d --name docker-cvparser -p 9001:9001 --add-host host.docker.internal:host-gateway cvparser-service

docker-build-worker:
	docker build --no-cache -f Dockerfile.worker -t docs-checker-worker .

docker-run-worker:
	mkdir -p logs/docs-checker
	docker run -d \
		--name docs-checker \
		--restart on-failure:5 \
		--network host \
		--user $$(id -u):$$(id -g) \
		--env-file .env \
		-v $(PWD)/logs:/app/logs \
		docs-checker-worker

docker-stop-worker:
	docker stop docs-checker && docker rm docs-checker

run-worker:
	python -m workers.docs_checker_worker
