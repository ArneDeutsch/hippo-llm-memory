# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
import subprocess
from pathlib import Path


def test_env_script_exports_run_id():
    """Ensure sourcing _env.sh exports RUN_ID for child processes."""

    script = Path("scripts/_env.sh").resolve()
    cmd = (
        f"RUN_ID=test_run; source {script}; "
        "python - <<'PY'\nimport os; print(os.environ.get('RUN_ID', ''))\nPY"
    )
    result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "test_run"
