Thanks for contributing! This project is designed to work smoothly with **ChatGPT Codex (web)** creating PRs. Please follow these guidelines.

# Workflow

1. Create an issue using **Issues → New issue → “Codex task”** template, or prepare a clear task prompt.
2. In ChatGPT, open **Codex → Code**, connect this repo, and paste the task prompt.
3. Codex runs in a CPU container (no GPU), executes `codex-env/setup.sh`, and runs `make lint` + `make test`.
4. Codex proposes diffs and opens a PR. You **review the PR**, request changes if needed, and merge.
5. **Training runs happen locally** on your 12 GB GPU.

# Branching & PRs

- Branch names: `feat/<area>-<slug>`, `fix/<area>-<slug>`, `exp/<experiment>-<slug>`.
- All PRs must:
  - Pass `make lint` and `make test`.
  - Update relevant docs (`RUN.md`, `README.md`) if behavior changes.
  - Fill the PR checklist in `.github/PULL_REQUEST_TEMPLATE.md`.

# Local dev environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r codex-env/requirements.txt
pre-commit install  # optional if you add pre-commit
```

# Testing

- Put unit tests under `tests/`. Use pytest, small synthetic fixtures, and temporary dirs.
- Keep tests CPU‑only and fast (< 30s total) so Codex/CI can run them.

# Style & typing

- Follow **CODING\_STANDARDS.md**. Run `make format` locally before committing.

# Security & provenance

- Do not commit API keys.
- Memory stores (episodic/KG) must record provenance (source text span, timestamp, confidence) and offer rollback/delete.
