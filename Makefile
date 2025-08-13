.PHONY: format lint test type fix install-dev

install-dev:
	python -m pip install -U pip
	pip install -r codex-env/requirements.txt

format:
	ruff check --fix .
	black .

lint:
	ruff check .
	black --check .

test:
	pytest -q

type:
	@echo "typing checks are optional for now"

fix: format
