docker-build:
	docker build --no-cache -t cvparser-service .

docker-run:
	docker run -d --name docker-cvparser -p 9001:9001 cvparser-service