install_requirements:
	python -m pip install --upgrade pip
	pip install pip-tools --upgrade
	pip install -r requirements.txt

add_requirements:
	pip-compile requirements.in

docker_build:
	docker-compose build

docker_up:
	docker-compose up -d

docker_down:
	docker-compose down

enter_docker:
	docker-compose exec -it age-detection bash
