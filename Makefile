.PHONY: format lint test type fix install-dev datasets

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

DATE ?= $(shell date +%Y%m%d)

datasets:
	@echo "Generating datasets for $(DATE)"
	@for suite in episodic semantic spatial; do \
	  for size in 50 200 1000; do \
	    for seed in 1337 2025 4242; do \
	      python scripts/build_datasets.py --suite $$suite --size $$size --seed $$seed --out data/$$suite/$$size\_$$seed.jsonl; \
	    done; \
	  done; \
         done

eval-baselines:
	python scripts/run_baselines.py --date $(DATE)

