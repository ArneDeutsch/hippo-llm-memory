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

# Comments
- Use NumPy-style triple-quoted docstrings for public modules, classes, functions, and dataclasses.
- Omit sections for returns, raises, side effects, or complexity when they match the defaults:
  - returns nothing
  - raises nothing
  - no side effects
  - complexity ``O(1)``
- Describe tensor shapes and units for times or scores.
- Keep comments short and intentional; avoid narrating code.
- Precede non-obvious logic with a rationale using `# why:`.
- Mark conditions or guarantees with `# pre:`, `# post:`, and `# invariant:`.
- Prefer logger calls for rationale; if comments are needed, prefix with `# log:`.
- Limit line length to 100 characters and avoid trailing spaces.
