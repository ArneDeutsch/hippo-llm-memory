#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
pip install -r codex-env/requirements.txt
echo "[codex-env] Dependencies installed. Running lint/tests will be possible."
