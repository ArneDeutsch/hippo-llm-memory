.PHONY: format lint test slow-test type fix install-dev datasets

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

slow-test:
	pytest -q -m slow --runslow


type:
	@echo "typing checks are optional for now"

fix: format

DATE ?= $(shell date +%Y%m%d)

datasets:
	@echo "Generating datasets for $(DATE)"
	@for suite in episodic semantic spatial episodic_multi episodic_cross episodic_capacity; do \
	  for size in 50 200 1000; do \
	    for seed in 1337 2025 4242; do \
	      python scripts/build_datasets.py --suite $$suite --size $$size --seed $$seed --out data/$$suite/$$size\_$$seed.jsonl; \
	    done; \
	  done; \
	done
	python scripts/audit_datasets.py

eval-baselines:
	python scripts/run_baselines.py --date $(DATE) \
	  --suites episodic semantic spatial episodic_multi episodic_cross episodic_capacity

smoke:
	bash scripts/smoke_eval.sh


gate-sweep:
	@for thr in 0.5 0.6 0.7; do \
	  python scripts/eval_model.py suite=semantic preset=memory/sgc_rss n=5 seed=1337 \
	    relational.gate.threshold=$$thr outdir=runs/$(DATE)/gate_sweep/rel_$$thr \
	    dry_run=true; \
	done; \
	for thr in 0.8 1.0 1.2; do \
	  python scripts/eval_model.py suite=spatial preset=memory/smpd n=5 seed=1337 \
	    spatial.gate.block_threshold=$$thr outdir=runs/$(DATE)/gate_sweep/spat_$$thr \
	    dry_run=true; \
	done
