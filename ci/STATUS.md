# CI Status

- Workflow: `.github/workflows/ci.yml` (runs on PRs)
- Steps: checkout → setup-python 3.10 → install deps → `make lint` → `make test`
- Badges: *none committed*
- Last run: unknown (local environment)
- Required checks: flake8, pytest
