# Milestone 8 — Execution Plan

> Status target: **Start of Milestone 8** (8a is complete).  
> Goal of this milestone: establish **baseline** metrics (no learned memory) with full reproducibility.

---

## 1) Summary

We will:
- deterministically generate datasets for all suites (`episodic`, `semantic`, `spatial`) at sizes **50 / 200 / 1000** with seeds **1337 / 2025 / 4242**,
- run the baseline presets **core / rag / longctx** over the full matrix,
- store metrics under `runs/20250822/baselines/...`, and
- aggregate per‑suite reports in `reports/20250822/`.

No GPU is required in this milestone; baselines use the `mock` model and run on CPU.

**Definition of Done** is the Gate in the updated `PROJECT_PLAN.md` (copied below for convenience).

---

## 2) Work packages → concrete tasks

### 2.1 Dataset generation & registry

**Codex‑executable tasks**

1. **Implement checksums & dataset cards**
   - **Files:** `scripts/build_datasets.py`
   - **Change:** After writing each JSONL, compute SHA256 and append to `data/<suite>/checksums.json`. Also write/update `data/<suite>/dataset_card.json` with:
     ```json
     {"suite": "<suite>", "sizes": [50,200,1000], "seeds": [1337,2025,4242],
       "generator_version": "<code_sha_or_semver>",
       "files": {"<size>_<seed>.jsonl": "<sha256>"},
       "created_utc": "<ISO8601>",
       "cli_example": "python scripts/build_datasets.py suite=<suite> size=<size> seed=<seed> out=data/<suite>/<size>_<seed>.jsonl"}
     ```
   - **Acceptance:** Running for `suite=episodic size=50 seed=1337` creates the JSONL and updates both metadata files with valid SHA256.

2. **Add a Make target to build all datasets**
   - **Files:** `Makefile`
   - **Target:** `make datasets DATE=20250822`
   - **Action:** loops suites×sizes×seeds and calls `python scripts/build_datasets.py ...`.
   - **Acceptance:** `make datasets` produces 27 JSONLs and updates checksums.

**Human task (no GPU)**
- Run: `make datasets DATE=20250822`

---

### 2.2 Run‑matrix driver

**Codex‑executable tasks**

3. **Create a simple driver**
   - **Files:** `scripts/run_baselines.py` (new)
   - **Functionality:** Iterate presets = `baselines/core, baselines/rag, baselines/longctx`; suites = `episodic, semantic, spatial`; sizes = `50,200,1000`; seeds = `1337,2025,4242`. For each combo:
     - Validate dataset checksum before run.
     - Call `python scripts/eval_bench.py suite=<suite> preset=<preset> n=<size> seed=<seed> date=20250822`
     - Ensure outputs land under `runs/20250822/<preset>/<suite>/<size>_<seed>/`.
   - **Meta:** Extend `eval_bench.py` to include `python`, `platform`, optional CUDA info, and `pip_hash` in `meta.json`.

4. **Makefile target**
   - **Files:** `Makefile`
   - **Target:** `make eval-baselines DATE=20250822`
   - **Action:** `python scripts/run_baselines.py --date 20250822`

5. **Smoke script for CI**
   - **Files:** `scripts/smoke_8.sh` (new)
   - **Action:** Build size=50 datasets, run only `baselines/core` with seed=1337 for all suites, then `python scripts/report.py --date 20250822`. Exit non‑zero on any missing artifact.

**Human task (no GPU)**
- Run:
  ```bash
  make eval-baselines DATE=20250822
  python scripts/report.py --date 20250822
  ```

---

### 2.3 Environment capture & determinism

**Codex‑executable tasks**

6. **Augment meta.json**
   - **Files:** `scripts/eval_bench.py`
   - **Change:** Record:
     - `python`: `sys.version`
     - `platform`: `platform.platform()`
     - `pip_hash`: SHA256 of `subprocess.check_output(["pip","freeze"])`
     - `cuda`: if `torch.cuda.is_available()`, record `torch.version.cuda` and driver
   - **Acceptance:** `meta.json` shows these fields.

7. **Checksum gate**
   - **Files:** `scripts/eval_bench.py`
   - **Change:** Before running, if `data/<suite>/<size>_<seed>.jsonl` exists, recompute SHA256 and compare with `checksums.json`; abort if mismatch.

**Human task**
- None.

---

### 2.4 Aggregation & reports

**Codex‑executable tasks**

8. **Robust aggregation**
   - **Files:** `scripts/report.py`
   - **Change:** Compute mean ± std across seeds per (suite, preset, size). Write Markdown tables to `reports/20250822/<suite>/summary.md`. If matplotlib available, save bar charts as PNG next to the Markdown.
   - **Acceptance:** Running on provided `runs/20250819/...` produces summaries without error.

9. **Tests**
   - **Files:** `tests/test_report.py`
   - **Change:** Add assertions for:
     - latest date discovery,
     - presence of the per‑suite summaries,
     - expected table headers for all presets,
     - graceful handling when optional fields (retrieval/gates) are absent.

**Human task**
- None.

---

### 2.5 CI & smoke wiring

**Codex‑executable tasks**

10. **Add CI job (optional if you use local runs only)**
    - **Files:** `.github/workflows/ci.yaml`
    - **Change:** Add a job `milestone8-smoke` that executes `scripts/smoke_8.sh`.

**Human task**
- None.

---

### 2.6 Documentation updates

**Codex‑executable tasks**

11. **EVAL_PLAN.md — Baselines section**
    - Pin exact presets, suites, sizes, seeds, and output layout.
    - Add a small “How to reproduce Milestone 8” guide (two commands).

12. **DESIGN.md — Baselines note**
    - Clarify that adapters, retrieval, and gates are disabled in baseline presets and must not affect latency/metrics (no accidental code paths).

**Human task**
- Review and approve wording.

---

## 3) Gate (Definition of Done)

- Datasets with checksums and dataset cards exist for all suites × sizes × seeds.
- For each preset × suite × size × seed, `metrics.json`, `metrics.csv`, `meta.json` are present and non‑empty under `runs/20250822/...`.
- `reports/20250822/<suite>/summary.md` exists for all suites with aggregated tables (mean ± std across seeds).
- `smoke_8.sh` runs end‑to‑end locally on CPU and returns success.
- Re‑running baselines on the same commit reproduces identical metrics.

---

## 4) Commands (for the human)

```bash
# From repo root
make datasets DATE=20250822
make eval-baselines DATE=20250822
python scripts/report.py --date 20250822

# (optional) quick smoke
bash scripts/smoke_8.sh
```

---

## 5) Impact on DESIGN.md / EVAL_PLAN.md

- **EVAL_PLAN.md:** update “Baselines” with the explicit matrix and artifact layout; include a short reproduction recipe.
- **DESIGN.md:** add one paragraph stating that Milestone 8 evaluates *no‑memory* configurations; any memory‑side telemetry must be disabled to avoid confounds.

