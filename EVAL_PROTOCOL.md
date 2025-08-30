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

> **Parameter reference**
> - `--store_dir`: base directory for persistent stores, typically `$STORES`
> - `--session_id`: logical key nested under each algorithm's subfolder
> - `--persist`: write to `store_dir` during `mode=teach` or `mode=replay`
> - `--mode`: `{teach,replay,test}` phase selector

> **Profiles**
> | Suite              | Recommended profile |
> | ------------------ | ------------------ |
> | episodic           | base               |
> | episodic_multi     | base               |
> | episodic_cross     | hard               |
> | episodic_capacity  | hard               |
> | semantic           | base               |
> | spatial            | base               |

## 0) Shell prelude — environment & variables

```bash
source scripts/env_prelude.sh
if [[ -z ${SIZES+x} ]]; then SIZES=(50 200 1000); fi
if [[ -z ${SEEDS+x} ]]; then SEEDS=(1337 2025 4242); fi

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
# python scripts/make_datasets.py --suite episodic --profile base --size 50  --seed 1337 --out data/episodic/50_1337.jsonl
# python scripts/make_datasets.py --suite semantic --profile hard --size 200 --seed 2025 --out data/semantic_hard/200_2025.jsonl
# python scripts/make_datasets.py --suite spatial  --profile base --size 1000 --seed 4242 --out data/spatial/1000_4242.jsonl
```

`--profile` selects a difficulty preset defined in `configs/datasets/`. The `datasets`
Makefile target emits both `base` and `hard` variants; use `dataset_profile=hard`
when running evaluations for suites such as `episodic_cross` or `episodic_capacity`
to avoid saturation.

---

## 3) **Baseline grid (fix)**

**Why this matters:** The previous step used `python scripts/run_baselines_bench.py`, which forwards to the **light-weight bench harness** (`hippo_mem.eval.bench`). That harness returns **ground‑truth as predictions** by design (for CI plumbing), so runs are **extremely fast but not meaningful** for baseline quality. For a *real* baseline you must invoke the **model harness**.

**Do this instead** — run the matrix with a **real model** and the baseline presets:

```bash
# Expand the SIZES/SEEDS arrays from §0 into comma‑separated lists
NV=$(IFS=,; echo "${SIZES[*]}")
SD=$(IFS=,; echo "${SEEDS[*]}")

python scripts/eval_model.py +run_matrix=true date="$DATE" \
  presets="[baselines/core,baselines/span_short,baselines/rag,baselines/longctx]" \
  tasks="[episodic,semantic,spatial,episodic_multi]" \
  n_values="[$NV]" seeds="[$SD]" \
  mode=teach model="$MODEL" outdir="$RUNS"

# Hard profiles for suites prone to saturation
python scripts/eval_model.py +run_matrix=true date="$DATE" \
  presets="[baselines/core,baselines/span_short,baselines/rag,baselines/longctx]" \
  tasks="[episodic_cross,episodic_capacity]" dataset_profile=hard \
  n_values="[$NV]" seeds="[$SD]" \
  mode=teach model="$MODEL" outdir="$RUNS"
```

Notes:
- Baseline presets **disable** memory and retrieval as configured under `configs/eval/baselines/*.yaml`.
- `mode=teach` avoids the `--store_dir/--session_id` requirement of `mode=test` and keeps runs stateless.
- `SIZES` and `SEEDS` default to `(50 200 1000)` and `(1337 2025 4242)`; override them before sourcing `scripts/env_prelude.sh`.
- You can extend `tasks` to include additional suites as needed.


---

## 4) Memory grids with teach → replay → test

We evaluate each algorithm with **pre‑replay** and **post‑replay** phases. Outputs are written under
`runs/$DATE/memory/<algo>/<suite>/<n>_<seed>/`. Pass `--store_dir "$STORES"`; each wrapper appends its
own `<algo>` subfolder and nests traces under the provided `--session_id`.

### 4.1) HEI‑NW (episodic + variants)

```bash
SID="hei_${DATE}"  # session id for persistent store reuse
# Base-profile suites
for suite in episodic episodic_multi; do
  for n in "${SIZES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      OUT="$RUNS/memory/hei_nw/$suite/${n}_${seed}"
      # Teach & persist
      python scripts/eval_cli.py suite="$suite" dataset_profile=base preset=memory/hei_nw n="$n" seed="$seed" date="$DATE" model="$MODEL" mode=teach persist=true store_dir="$STORES" session_id="$SID" outdir="$OUT" --strict-telemetry
      # Validate persisted store layout
      python scripts/validate_store.py --algo hei_nw --kind episodic
      # Replay (3 cycles)
      python scripts/eval_cli.py suite="$suite" dataset_profile=base preset=memory/hei_nw n="$n" seed="$seed" date="$DATE" model="$MODEL" mode=replay persist=true store_dir="$STORES" session_id="$SID" replay.cycles=3 outdir="$OUT" --strict-telemetry
      # Test (post‑replay)
      python scripts/eval_cli.py suite="$suite" dataset_profile=base preset=memory/hei_nw n="$n" seed="$seed" date="$DATE" model="$MODEL" mode=test store_dir="$STORES" session_id="$SID" outdir="$OUT" --strict-telemetry
    done
  done
done

# Hard-profile suites to avoid saturation
for suite in episodic_cross episodic_capacity; do
  for n in "${SIZES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      OUT="$RUNS/memory/hei_nw/$suite/${n}_${seed}"
      # Teach & persist
      python scripts/eval_cli.py suite="$suite" dataset_profile=hard preset=memory/hei_nw n="$n" seed="$seed" date="$DATE" model="$MODEL" mode=teach persist=true store_dir="$STORES" session_id="$SID" outdir="$OUT" --strict-telemetry
      # Validate persisted store layout
      python scripts/validate_store.py --algo hei_nw --kind episodic
      # Replay (3 cycles)
      python scripts/eval_cli.py suite="$suite" dataset_profile=hard preset=memory/hei_nw n="$n" seed="$seed" date="$DATE" model="$MODEL" mode=replay persist=true store_dir="$STORES" session_id="$SID" replay.cycles=3 outdir="$OUT" --strict-telemetry
      # Test (post‑replay)
      python scripts/eval_cli.py suite="$suite" dataset_profile=hard preset=memory/hei_nw n="$n" seed="$seed" date="$DATE" model="$MODEL" mode=test store_dir="$STORES" session_id="$SID" outdir="$OUT" --strict-telemetry
    done
  done
done
```

> When running with `SIZES=(50)` and `SEEDS=(1337)`, the expected store is `runs/$RUN_ID/stores/hei_nw/hei_$RUN_ID/episodic.jsonl`.

```bash
ls -l "runs/$RUN_ID/stores/hei_nw/hei_${RUN_ID}/episodic.jsonl"
```

### 4.2) SGC‑RSS (semantic)

```bash
SID="sgc_${DATE}"
suite=semantic
for n in "${SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    OUT="$RUNS/memory/sgc_rss/$suite/${n}_${seed}"
    python scripts/eval_cli.py suite="$suite" preset=memory/sgc_rss n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=teach persist=true store_dir="$STORES" session_id="$SID" outdir="$OUT" --strict-telemetry
    # Validate persisted store layout
    python scripts/validate_store.py --algo sgc_rss --kind kg
    python scripts/eval_cli.py suite="$suite" preset=memory/sgc_rss n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=replay persist=true store_dir="$STORES" session_id="$SID" replay.cycles=3 outdir="$OUT" --strict-telemetry
    python scripts/eval_cli.py suite="$suite" preset=memory/sgc_rss n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=test store_dir="$STORES" session_id="$SID" outdir="$OUT" --strict-telemetry
  done
done
```

### 4.3) SMPD (spatial)

```bash
SID="smpd_${DATE}"
suite=spatial
for n in "${SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    OUT="$RUNS/memory/smpd/$suite/${n}_${seed}"
    python scripts/eval_cli.py suite="$suite" preset=memory/smpd n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=teach persist=true store_dir="$STORES" session_id="$SID" outdir="$OUT" --strict-telemetry
    # Validate persisted store layout
    python scripts/validate_store.py --algo smpd --kind map
    python scripts/eval_cli.py suite="$suite" preset=memory/smpd n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=replay persist=true store_dir="$STORES" session_id="$SID" replay.cycles=3 outdir="$OUT" --strict-telemetry
    python scripts/eval_cli.py suite="$suite" preset=memory/smpd n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=test store_dir="$STORES" session_id="$SID" outdir="$OUT" --strict-telemetry
  done
done
```

---

## 5) Ablations (optional but recommended)

Example ablations for gate toggles and retrieval. Reuse the session
identifiers from §4 and iterate over the same `SIZES` and `SEEDS` arrays:

```bash
# Disable relational gating to quantify its impact
SID="sgc_${DATE}"
suite=semantic
for n in "${SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    OUT="$RUNS/ablate/sgc_rss_no_gate/$suite/${n}_${seed}"
      python scripts/eval_cli.py suite="$suite" preset=memory/sgc_rss n="$n" seed="$seed" date="$DATE" \
        model="$MODEL" mode=test store_dir="$STORES" session_id="$SID" \
        gating_enabled=false outdir="$OUT" --strict-telemetry
  done
done

# Disable retrieval to isolate the long‑context baseline (stateless; `mode=teach` avoids store requirements)
suite=semantic
for n in "${SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    OUT="$RUNS/ablate/longctx_no_retrieval/$suite/${n}_${seed}"
    python scripts/eval_model.py suite="$suite" preset=baselines/longctx n="$n" seed="$seed" date="$DATE" \
      model="$MODEL" mode=teach outdir="$OUT"
  done
done
```

---

## 6) Gate threshold sweeps (clarified)

These are **small, example sweeps** around the defaults to **check sensitivity**, **not** exhaustive grid searches. Keep them light to control runtime. Reuse the persisted stores from §4; `mode=test` requires `--store_dir` and `--session_id`.

```bash
# Choose small, symmetric ranges around the configured defaults:
#   episodic tau (~0.5), relational threshold (=0.6), spatial block_threshold (=1.0)
EPISODIC_TAUS=(0.3 0.5 0.7)
RELATIONAL_THRESHOLDS=(0.4 0.6 0.8)
SPATIAL_BLOCK_THRESHOLDS=(0.5 1.0 2.0)

SWEEP_SIZES=(200)
SWEEP_SEEDS=(1337 2025)

# Reuse persisted stores from §4
SID_HEI="hei_${DATE}"
SID_SGC="sgc_${DATE}"
SID_SMPD="smpd_${DATE}"

# HEI‑NW taus
for tau in "${EPISODIC_TAUS[@]}"; do
  python scripts/eval_cli.py suite=episodic preset=memory/hei_nw \
    n="${SWEEP_SIZES[0]}" seed="${SWEEP_SEEDS[0]}" date="$DATE" \
    model="$MODEL" mode=test store_dir="$STORES" session_id="$SID_HEI" \
    episodic.gate.tau="$tau" outdir="$RUNS/sweeps/hei_nw_tau_${tau}" --strict-telemetry
done

# SGC‑RSS relational thresholds
for thr in "${RELATIONAL_THRESHOLDS[@]}"; do
  python scripts/eval_cli.py suite=semantic preset=memory/sgc_rss \
    n="${SWEEP_SIZES[0]}" seed="${SWEEP_SEEDS[0]}" date="$DATE" \
    model="$MODEL" mode=test store_dir="$STORES" session_id="$SID_SGC" \
    relational.gate.threshold="$thr" \
    outdir="$RUNS/sweeps/sgc_rss_thr_${thr}" --strict-telemetry
done

# SMPD spatial thresholds
for thr in "${SPATIAL_BLOCK_THRESHOLDS[@]}"; do
  python scripts/eval_cli.py suite=spatial preset=memory/smpd \
    n="${SWEEP_SIZES[0]}" seed="${SWEEP_SEEDS[0]}" date="$DATE" \
    model="$MODEL" mode=test store_dir="$STORES" session_id="$SID_SMPD" \
    spatial.gate.block_threshold="$thr" \
    outdir="$RUNS/sweeps/smpd_thr_${thr}" --strict-telemetry
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

> When running with `SIZES=(50)` and `SEEDS=(1337)`, the expected store is `runs/$RUN_ID/stores/hei_nw/hei_$RUN_ID/episodic.jsonl`.

```bash
# §8 prelude (self-contained)
source scripts/env_prelude.sh

SID="hei_${RUN_ID}"

# Require a persisted HEI-NW store
test -f "$STORES/hei_nw/$SID/episodic.jsonl" || {
  echo "No persisted HEI-NW store for $SID. Run §4.1 (teach+replay with persist=true) first."
  exit 1
}
ls -l "runs/$RUN_ID/stores/hei_nw/hei_${RUN_ID}/episodic.jsonl"

# 1) Pre‑consolidation baseline (memory OFF)
python scripts/test_consolidation.py --phase pre \
  --suite episodic --n 50 --seed 1337 \
  --model "$MODEL" --outdir "$RUNS/consolidation/pre"

# 2) Replay → LoRA training
python scripts/replay_consolidate.py \
  --store_dir "$STORES" --session_id "$SID" \
  --model "$MODEL" --config configs/consolidation/lora_small.yaml \
  --outdir "$RUNS/consolidation/lora"

# 3) Post‑consolidation test (memory OFF with adapter)
python scripts/test_consolidation.py --phase post \
  --suite episodic --n 50 --seed 1337 \
  --model "$MODEL" --adapter "$RUNS/consolidation/lora" \
  --pre_dir "$RUNS/consolidation/pre" --outdir "$RUNS/consolidation/post" \
  --min-uplift 0.05
```

Use `--uplift-mode ci` when multiple seeds are present; the gate then
requires the 95% CI of `(post - pre)` to exclude zero (significance
controlled by `--alpha`, default 0.05).

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

- `python scripts/run_baselines_bench.py`: **CI/plumbing‑only**. It calls the light‑weight bench that returns ground truth as predictions; **do not use** for real baselines.
