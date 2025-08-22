# Codex prompts — Milestone 9 readiness fixes (from Review §4)

## Prompt 1

CODex Task: Add compute telemetry (time and memory) to metrics
Context:
- We need to record `compute.time_ms_per_100` and `compute.rss_mb` in `metrics.json` for ALL runs (baselines and memory).
- Files involved: `scripts/eval_bench.py`, `scripts/eval_model.py`, CSV writer code paths, and `scripts/report.py` (to read/aggregate new fields).
- Dependencies: Add `psutil` to `codex-env/requirements.txt` for cross-platform RSS collection. Fallback to `resource` on Unix if `psutil` not available.

Edits:
1) `codex-env/requirements.txt`: append a new line: `psutil>=5.9`.
2) In both `scripts/eval_bench.py` and `scripts/eval_model.py`, wrap the evaluation loop with wall-clock timing using `time.perf_counter()`.
   - Compute `elapsed = (t1 - t0)` seconds for the processed `n_items`.
   - Write `metrics["compute"]["time_ms_per_100"] = 100000 * elapsed / max(1, n_items)`.
3) Add RSS measurement:
   - Try: `import psutil; rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)`.
   - Except ImportError: on POSIX, use `import resource; rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; if sys.platform == "darwin": rss_mb /= 1024`.
   - Set `metrics["compute"]["rss_mb"] = float(rss_mb)`.
4) Ensure CSV output includes compute columns:
   - When writing per-row or aggregate CSV, include `time_ms_per_100` and `rss_mb` in the compute columns if present.
5) `scripts/report.py`: when summarising, include the new compute fields in the table(s) and plot(s) if present.

Definition of Done:
- Running `make install-dev && DATE=20250822 make eval-baselines && python scripts/report.py --date 20250822` completes without error.
- Each `runs/20250822/.../metrics.json` has the nested keys: `metrics.compute.time_ms_per_100` and `metrics.compute.rss_mb` with non-null numeric values.
- The generated `reports/20250822/<suite>/summary.md` shows these columns in the compute section.


## Prompt 2

CODex Task: Make `meta.json` self-contained (duplicate suite/preset/n)
Context:
- `meta.json` currently lacks `suite`, `preset`, and `n`, which forces consumers to also read `metrics.json`. We want standalone provenance.
- Files involved: `scripts/eval_bench.py`, `scripts/eval_model.py` (where meta is written).

Edits:
- When writing `meta.json`, inject the following top-level fields:
  - `"suite": <episodic|semantic|spatial>`
  - `"preset": <e.g., baselines/core or memory/hei_nw>`
  - `"n": <int>`
- Keep existing fields (`git_sha`, `pip_hash`, `model`, `seed`, ablation flags, replay cycles, etc.).

Definition of Done:
- In every `runs/20250822/.../meta.json`, the keys `suite`, `preset`, and `n` exist and match the corresponding `metrics.json` entries.
- `scripts/report.py` can summarise using only `metrics.json` and `meta.json` if needed.


## Prompt 3

CODex Task: Replace zeroed latency with measured per-item latency
Context:
- In baseline harness the per-item `latency_ms` column in `metrics.csv` is currently zero. We need realistic timings even for mock inference to catch regressions.
- Files: `scripts/eval_bench.py` (and optionally `scripts/eval_model.py` if it writes per-item CSV).

Edits:
- Around the mock "inference" step (where predictions are produced), time each iteration using `perf_counter()` and record `latency_ms = (t_after - t_before) * 1000.0` for that item.
- Write this per-item field to `metrics.csv` (and any in-memory aggregations that compute mean latency into `metrics.json` as `metrics.compute.latency_ms_mean`).
- Ensure that even for very fast iterations the timings are non-zero by taking the total loop time as a fallback mean if all per-item timings are too coarse.

Definition of Done:
- `metrics.csv` per-run has a non-zero `latency_ms` column.
- `metrics.json` has `metrics.compute.latency_ms_mean` (> 0 for non-empty runs).


## Prompt 4

CODex Task: Generate top-level roll-up report `reports/<date>/index.md`
Context:
- We currently produce per-suite files under `reports/<date>/<suite>/summary.md`. Add a roll-up `reports/<date>/index.md` with:
  - A matrix table across suites × presets (EM mean, tokens, compute fields).
  - Links to each per-suite summary.
  - If matplotlib available, embed a simple overall bar chart PNG comparing presets.

Edits:
- Extend `scripts/report.py`:
  - After building per-suite summaries, aggregate across suites and write `reports/<date>/index.md`.
  - Create `reports/<date>/assets/` for any images saved by the roll-up.
- Update CLI help to mention the new output.

Definition of Done:
- Running `python scripts/report.py --date 20250822` creates `reports/20250822/index.md` with a top-level table and links to the three suite summaries.
- If matplotlib is available, a PNG `reports/20250822/assets/overall_em.png` is produced and referenced from the index.


## Prompt 5

CODex Task: Emit a unified dataset MANIFEST
Context:
- We have per-suite `checksums.json`. Add a unified `data/MANIFEST.json` covering all suites with file list, sha256, and item counts.
- Files: `scripts/audit_datasets.py` (extend) OR create `scripts/build_manifest.py` and call it from the `datasets` Makefile target.

Edits (option A – extend audit script):
- In `scripts/audit_datasets.py`, after verifying, build a dict:
  `{suite: [{"file": "<path>", "sha256": "<hex>", "items": <int>}, ...], "sizes": [50,200,1000], "seeds": [1337,2025,4242]}`
- Count items by reading JSONL lines for each file.
- Write to `data/MANIFEST.json` with stable ordering.

Makefile:
- Append a step to the `datasets` target to run `python scripts/audit_datasets.py` so the manifest is refreshed after dataset generation.

Definition of Done:
- `data/MANIFEST.json` exists with all dataset entries and non-empty hashes and item counts.
- `make datasets DATE=20250822` regenerates datasets and refreshes the manifest without errors.


## Prompt 6

CODex Task: Add a tiny human-readable smoke report with raw rows
Context:
- We want `reports/<date>/smoke.md` showing 2–3 raw rows per suite for a quick human integrity check (prompt/answer format, no PII).

Edits:
- Add `scripts/report_smoke.py` (or extend `scripts/report.py` with a `--smoke` flag) to:
  - For each suite, open the first JSONL in `data/<suite>/` (prefer the smallest, e.g., `50_1337.jsonl`), read 3 lines.
  - Render a Markdown table with selected fields, e.g., `id`, `question`, `answer` (or task-appropriate names present in the JSONL).
  - Write to `reports/<date>/smoke.md`.
- Link `smoke.md` from `reports/<date>/index.md` if present.

Definition of Done:
- `python scripts/report.py --date 20250822` (or `python scripts/report_smoke.py --date 20250822`) creates `reports/20250822/smoke.md` with sample rows from each suite.
- The `index.md` contains a link: `See also: smoke.md`.


