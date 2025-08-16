#!/usr/bin/env bash
set -e

# Unit tests and coverage
pytest -q --maxfail=1
pytest --cov=hippo_mem --cov=scripts --cov-report=term-missing --cov-report=html:coverage/html --cov-report=json:coverage/coverage.json

# Static analysis
python -m flake8 .
python -m bandit -q -r . > static/bandit.txt
python -m radon cc -s -a hippo_mem > static/radon_cc.txt
python -m radon mi hippo_mem > static/radon_mi.txt

# Dataset generation (small synthetic)
mkdir -p data
python scripts/build_datasets.py --suite episodic --n 50 --seed 1 --out data/episodic.jsonl

# Dry-run training and evaluation
PYTHONPATH=. python scripts/train_lora.py dry_run=true max_steps=5
PYTHONPATH=. python scripts/eval_bench.py suite=episodic preset=core
PYTHONPATH=. python scripts/eval_bench.py suite=episodic preset=longctx
PYTHONPATH=. python scripts/eval_bench.py suite=episodic preset=rag
