# EVAL_PROTOCOL.md — Full Validation Run (Milestones 9 & 9.5)

_Updated: 2025-08-28 (by ChatGPT)_

This protocol executes a **complete, reproducible** validation run across **baselines** and **memory-enabled** algorithms (HEI‑NW, SGC‑RSS, SMPD), plus **ablations** and **consolidation** checks (Milestone 9.5).

> All commands are copy‑pasteable in **bash**. Lines with `#` are comments.

---

### Choosing a run identifier

Set a stable session id once before running the protocol:

```bash
# Example: timestamp + short git sha
export RUN_ID="$(date -u +%Y%m%d_%H%M)-$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
# or a human label, e.g.
export RUN_ID="ablation-aug29"
```

All outputs (runs, reports, stores, adapters) and `meta.json.date` will use this `RUN_ID`.
For backward compatibility, `DATE` is set equal to `RUN_ID` inside the protocol.

## 0) Shell prelude — environment & variables

```bash
set -euo pipefail

# >>> RUN_ID prelude (stable session identifier)
# Prefer RUN_ID if provided; else use DATE; else use UTC timestamp.
: "${RUN_ID:=${DATE:-$(date -u +%Y%m%d_%H%M)}}"
# Back-compat: keep DATE equal to RUN_ID so existing $DATE usages still work.
DATE="$RUN_ID"

# Derived paths
RUNS="runs/$RUN_ID"
REPORTS="reports/$RUN_ID"
STORES="$RUNS/stores"
ADAPTERS="adapters/$RUN_ID"

# Defaults (caller can override before sourcing)
: "${MODEL:=Qwen/Qwen2.5-1.5B-Instruct}"
if [[ -z ${SIZES+x} ]]; then SIZES=(50 200 1000); fi
if [[ -z ${SEEDS+x} ]]; then SEEDS=(1337 2025 4242); fi
# <<< RUN_ID prelude

mkdir -p "$RUNS" "$REPORTS" "$STORES" "$ADAPTERS"

echo "RUN_ID=$RUN_ID"
echo "MODEL=$MODEL"
```

---

## 1) Environment & sanity checks

```bash
# Install dev deps (if not already)
make install-dev

# Basic quality gates (optional but recommended)
make lint
make test

# Quick smoke (tiny sizes, tiny model) — does not validate quality
bash scripts/smoke_eval.sh

# Gate overhead micro-benchmark (target ≤10% overhead)
python scripts/bench_gating_overhead.py
```

---

## 2) Build & audit datasets

```bash
# Build standard JSONL datasets for all suites/sizes/seeds
make datasets DATE="$DATE"

# In case you prefer explicit calls (equivalent to the Makefile target), e.g.:
# python scripts/build_datasets.py --suite episodic --size 50  --seed 1337 --out data/episodic/50_1337.jsonl
# python scripts/build_datasets.py --suite semantic --size 200 --seed 2025 --out data/semantic/200_2025.jsonl
# python scripts/build_datasets.py --suite spatial  --size 1000 --seed 4242 --out data/spatial/1000_4242.jsonl
```

---

## 3) **Baseline grid (fix)**

**Why this matters:** The previous step used `python scripts/run_baselines.py`, which forwards to the **light-weight bench harness** (`hippo_mem.eval.bench`). That harness returns **ground‑truth as predictions** by design (for CI plumbing), so runs are **extremely fast but not meaningful** for baseline quality. For a *real* baseline you must invoke the **model harness**.

**Do this instead** — run the matrix with a **real model** and the baseline presets:

```bash
python scripts/eval_model.py +run_matrix=true date="$DATE" \
  presets="[baselines/core,baselines/span_short,baselines/rag,baselines/longctx]" \
  tasks="[episodic,semantic,spatial,episodic_multi,episodic_cross,episodic_capacity]" \
  n_values="[50,200,1000]" \
  seeds="[1337,2025,4242]" \
  model="$MODEL" outdir="runs/$DATE"
```

Notes:
- Baseline presets **disable** memory and retrieval as configured under `configs/eval/baselines/*.yaml`.
- You can extend `tasks` to include `episodic_multi,episodic_cross,episodic_capacity` if required.


---

## 4) Memory grids with teach → replay → test

We evaluate each algorithm with **pre‑replay** and **post‑replay** phases. Outputs are written under `runs/$DATE/memory/<algo>/<suite>/<n>_<seed>/`.

### 4.1) HEI‑NW (episodic + variants)

```bash
SESS="hei_${DATE}"  # session id for persistent store reuse
for suite in episodic episodic_multi episodic_cross episodic_capacity; do
  for n in "${SIZES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      OUT="$RUNS/memory/hei_nw/$suite/${n}_${seed}"
      # Teach & persist
      python scripts/eval_model.py suite="$suite" preset=memory/hei_nw n="$n" seed="$seed" date="$DATE"         model="$MODEL" mode=teach store_dir="$STORES/hei_nw" session_id="$SESS" outdir="$OUT"
      # Replay (3 cycles)
      python scripts/eval_model.py suite="$suite" preset=memory/hei_nw n="$n" seed="$seed" date="$DATE"         model="$MODEL" mode=replay store_dir="$STORES/hei_nw" session_id="$SESS" replay.cycles=3 outdir="$OUT"
      # Test (post‑replay)
      python scripts/eval_model.py suite="$suite" preset=memory/hei_nw n="$n" seed="$seed" date="$DATE"         model="$MODEL" mode=test store_dir="$STORES/hei_nw" session_id="$SESS" outdir="$OUT"
    done
  done
done
```

### 4.2) SGC‑RSS (semantic)

```bash
SESS="sgc_${DATE}"
suite=semantic
for n in "${SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    OUT="$RUNS/memory/sgc_rss/$suite/${n}_${seed}"
    python scripts/eval_model.py suite="$suite" preset=memory/sgc_rss n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=teach store_dir="$STORES/sgc_rss" session_id="$SESS" outdir="$OUT"
    python scripts/eval_model.py suite="$suite" preset=memory/sgc_rss n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=replay store_dir="$STORES/sgc_rss" session_id="$SESS" replay.cycles=3 outdir="$OUT"
    python scripts/eval_model.py suite="$suite" preset=memory/sgc_rss n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=test store_dir="$STORES/sgc_rss" session_id="$SESS" outdir="$OUT"
  done
done
```

### 4.3) SMPD (spatial)

```bash
SESS="smpd_${DATE}"
suite=spatial
for n in "${SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    OUT="$RUNS/memory/smpd/$suite/${n}_${seed}"
    python scripts/eval_model.py suite="$suite" preset=memory/smpd n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=teach store_dir="$STORES/smpd" session_id="$SESS" outdir="$OUT"
    python scripts/eval_model.py suite="$suite" preset=memory/smpd n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=replay store_dir="$STORES/smpd" session_id="$SESS" replay.cycles=3 outdir="$OUT"
    python scripts/eval_model.py suite="$suite" preset=memory/smpd n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=test store_dir="$STORES/smpd" session_id="$SESS" outdir="$OUT"
  done
done
```

---

## 5) Ablations (optional but recommended)

Example ablations for gate toggles and retrieval:

```bash
# Disable gating to quantify its impact
python scripts/eval_model.py suite=semantic preset=memory/sgc_rss n=200 seed=1337 date="$DATE"   model="$MODEL" mode=test gating_enabled=false outdir="$RUNS/ablate/sgc_rss_no_gate"

# Disable retrieval to isolate long‑context baseline
python scripts/eval_model.py suite=semantic preset=baselines/longctx n=200 seed=1337 date="$DATE"   model="$MODEL" outdir="$RUNS/ablate/longctx_no_retrieval"
```

---

## 6) Gate threshold sweeps (clarified)

These are **small, example sweeps** around the defaults to **check sensitivity**, **not** exhaustive grid searches. Keep them light to control runtime.

```bash
# Choose small, symmetric ranges around the configured defaults:
#   episodic tau (~0.5), relational threshold (=0.6), spatial block_threshold (=1.0)
EPISODIC_TAUS=(0.3 0.5 0.7)
RELATIONAL_THRESHOLDS=(0.4 0.6 0.8)
SPATIAL_BLOCK_THRESHOLDS=(0.5 1.0 2.0)

SWEEP_SIZES=(200)
SWEEP_SEEDS=(1337 2025)

# HEI‑NW taus
for tau in "${EPISODIC_TAUS[@]}"; do
  python scripts/eval_model.py suite=episodic preset=memory/hei_nw n="${SWEEP_SIZES[0]}" seed="${SWEEP_SEEDS[0]}" date="$DATE"     model="$MODEL" mode=test episodic.gate.tau="$tau" outdir="$RUNS/sweeps/hei_nw_tau_${tau}"
done

# SGC‑RSS relational thresholds
for thr in "${RELATIONAL_THRESHOLDS[@]}"; do
  python scripts/eval_model.py suite=semantic preset=memory/sgc_rss n="${SWEEP_SIZES[0]}" seed="${SWEEP_SEEDS[0]}" date="$DATE"     model="$MODEL" mode=test relational.gate.threshold="$thr" outdir="$RUNS/sweeps/sgc_rss_thr_${thr}"
done

# SMPD spatial thresholds
for thr in "${SPATIAL_BLOCK_THRESHOLDS[@]}"; do
  python scripts/eval_model.py suite=spatial preset=memory/smpd n="${SWEEP_SIZES[0]}" seed="${SWEEP_SEEDS[0]}" date="$DATE"     model="$MODEL" mode=test spatial.gate.block_threshold="$thr" outdir="$RUNS/sweeps/smpd_thr_${thr}"
done
```

---

## 7) Summaries & reports

```bash
# Per-preset summaries as CSV/JSON
python scripts/summarize_runs.py "$RUNS" --out "$RUNS/summaries"

# Markdown reports and plots
python scripts/report.py --date "$DATE" --runs-dir runs --out-dir reports --data-dir data --plots
```

---

## 8) Consolidation checks (Milestone 9.5)

Minimal smoke for **cross‑session recall** using the same store directory.

```bash
# After one run finished, do a delayed recall test (post-hoc)
python scripts/replay_consolidate.py --date "$DATE" --store "$STORES/hei_nw" --model "$MODEL"   --out "$RUNS/consolidation/hei_nw_smoke"
python scripts/test_consolidation.py --runs "$RUNS/consolidation/hei_nw_smoke"
```

---

## 9) Final roll‑up

```bash
echo "Done. See:"
echo "  - $RUNS        (raw run outputs)"
echo "  - $RUNS/summaries (per‑preset CSV/JSON)"
echo "  - $REPORTS     (Markdown summaries & plots)"
echo "  - $ADAPTERS    (trained adapters, if any)"
```

---

### Appendix — Deprecated commands

- `python scripts/run_baselines.py`: **CI/plumbing‑only**. It calls the light‑weight bench that returns ground truth as predictions; **do not use** for real baselines.
