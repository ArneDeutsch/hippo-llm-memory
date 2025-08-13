# Python & packaging

- Python **3.10+**. Package code under `hippo_mem/` with clear module boundaries.
- Prefer **type hints** and **dataclasses**.
- Docstrings: short module/class/function summary + args/returns.

# Linting & formatting

- **ruff** for linting; **black** for formatting. Run:
  - `make format` to auto‑fix
  - `make lint` to check only

# Imports & structure

- Absolute imports from `hippo_mem` in app code; relative imports allowed inside subpackages.
- Keep functions ≤ 60 lines where reasonable; refactor into helpers.

# Tests

- One test module per feature: `tests/test_<feature>.py`.
- Use small, deterministic fixtures; avoid network and GPU.

# Logging & errors

- Use `logging` (no prints in library code). Raise specific exceptions; include actionable messages.

# Performance notes

- Retrieval paths must be **CPU‑friendly** (FAISS‑CPU). Keep vector dims modest (e.g., 384–1024).
- Adapters: default LoRA r=16, α=32, dropout=0.05; configurable in Hydra.
