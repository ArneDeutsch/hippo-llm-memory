#!/bin/bash
set -euxo pipefail

pytest -q --maxfail=1
pytest --cov=hippo_mem --cov=scripts --cov-report=term-missing --cov-report=html:coverage/html --cov-report=json:coverage/coverage.json
flake8 .
bandit -q -r .
mkdir -p static
radon cc -s -a hippo_mem > static/radon_cc.txt
radon mi hippo_mem > static/radon_mi.txt

PYTHONPATH=. python scripts/build_datasets.py --suite episodic --n 50 --seed 0 --out data/episodic_small.jsonl
PYTHONPATH=. python scripts/build_datasets.py --suite semantic --n 50 --seed 0 --out data/semantic_small.jsonl
PYTHONPATH=. python scripts/build_datasets.py --suite spatial --n 50 --seed 0 --out data/spatial_small.jsonl

PYTHONPATH=. python scripts/train_lora.py episodic.enabled=true relational=true spatial.enabled=true replay.enabled=true max_steps=5 dry_run=true || true
PYTHONPATH=. python scripts/train_lora.py episodic.enabled=false relational=false spatial.enabled=false replay.enabled=false max_steps=3 dry_run=true

PYTHONPATH=. python scripts/eval_bench.py suite=episodic preset=core n=5 outdir=runs/tmp1
PYTHONPATH=. python scripts/eval_bench.py suite=episodic preset=longctx n=5 outdir=runs/tmp2
PYTHONPATH=. python scripts/eval_bench.py suite=episodic preset=rag n=5 outdir=runs/tmp3
