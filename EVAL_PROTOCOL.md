# EVAL_PROTOCOL.md — Full Validation Run (Milestones 9 & 9.5)

_Updated: 2025-08-28 05:58_

This protocol executes a **complete, reproducible** validation run across **baselines** and **memory-enabled** presets (HEI‑NW, SGC‑RSS, SMPD), including **teach → replay → test**, **ablations**, and **systems consolidation** (Milestone 9.5).

> All commands are copy‑pasteable in **bash**. Lines with `#` are comments.

---

## 0) Shell prelude — environment & variables

```bash
set -euo pipefail

# 0.1) Date stamp for this run and standard directories
DATE=$(date +%Y%m%d_%H%M)
RUNS="runs/$DATE"
REPORTS="reports/$DATE"
STORES="$RUNS/stores"
ADAPTERS="adapters/$DATE"

mkdir -p "$RUNS" "$REPORTS" "$STORES" "$ADAPTERS"

# 0.2) Model & matrix defaults (override as needed)
MODEL="Qwen/Qwen2.5-1.5B-Instruct"   # or models/tiny-gpt2 for smoke/local
SIZES=(50 200 1000)
SEEDS=(1337 2025 4242)

# 0.3) Session id for cross‑session teach/test
SESS="s1"

# 0.4) (Optional) Install dev deps
# make install-dev
```


---

## 1) Datasets (required)

The Make target builds JSONL datasets for all suites (including episodic variants) at sizes 50/200/1000 and seeds 1337/2025/4242, then audits them.

```bash
make datasets DATE="$DATE"
```


---

## 2) Sanity checks (optional but recommended)

```bash
# Quick smoke (small model / tiny sizes)
bash scripts/smoke_eval.sh

# Gate overhead benchmark (should be ≤ 10% overhead)
python scripts/bench_gating_overhead.py
```


---

## 3) Baseline grid (Milestone 9 — H2)

Runs the full baseline matrix across presets: `baselines/core`, `baselines/span_short`, `baselines/rag`, `baselines/longctx` and suites: `episodic`, `semantic`, `spatial`, plus episodic variants.

```bash
python scripts/run_baselines.py --date "$DATE"
```


---

## 4) Memory grids with teach → replay → test (Milestone 9 — H3/H4/H5)

> We use `scripts/eval_model.py` (Hydra overrides). The harness writes `pre_*` and `post_*` metrics when replay runs. We set `replay.cycles=3` for a measurable Δ.

### 4.1) HEI‑NW (episodic) — suites: episodic, episodic_multi, episodic_cross, episodic_capacity

```bash
for suite in episodic episodic_multi episodic_cross episodic_capacity; do
  for n in "${SIZES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      OUT="$RUNS/memory/hei_nw/$suite/${n}_${seed}"
      # Teach & persist
      python scripts/eval_model.py         suite="$suite" preset=memory/hei_nw n="$n" seed="$seed" date="$DATE"         model="$MODEL" mode=teach persist=true store_dir="$STORES/hei_nw" session_id="$SESS"         outdir="$OUT"
      # Replay (3 cycles)
      python scripts/eval_model.py         suite="$suite" preset=memory/hei_nw n="$n" seed="$seed" date="$DATE"         model="$MODEL" mode=replay store_dir="$STORES/hei_nw" session_id="$SESS"         replay.cycles=3 outdir="$OUT"
      # Test (no ingestion)
      python scripts/eval_model.py         suite="$suite" preset=memory/hei_nw n="$n" seed="$seed" date="$DATE"         model="$MODEL" mode=test store_dir="$STORES/hei_nw" session_id="$SESS"         outdir="$OUT"
    done
  done
done
```

### 4.2) SGC‑RSS (semantic)

```bash
suite=semantic
for n in "${SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    OUT="$RUNS/memory/sgc_rss/$suite/${n}_${seed}"
    python scripts/eval_model.py       suite="$suite" preset=memory/sgc_rss n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=teach persist=true store_dir="$STORES/sgc_rss" session_id="$SESS"       outdir="$OUT"
    python scripts/eval_model.py       suite="$suite" preset=memory/sgc_rss n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=replay store_dir="$STORES/sgc_rss" session_id="$SESS"       replay.cycles=3 outdir="$OUT"
    python scripts/eval_model.py       suite="$suite" preset=memory/sgc_rss n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=test store_dir="$STORES/sgc_rss" session_id="$SESS"       outdir="$OUT"
  done
done
```

### 4.3) SMPD (spatial)

```bash
suite=spatial
for n in "${SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    OUT="$RUNS/memory/smpd/$suite/${n}_${seed}"
    python scripts/eval_model.py       suite="$suite" preset=memory/smpd n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=teach persist=true store_dir="$STORES/smpd" session_id="$SESS"       outdir="$OUT"
    python scripts/eval_model.py       suite="$suite" preset=memory/smpd n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=replay store_dir="$STORES/smpd" session_id="$SESS"       replay.cycles=3 outdir="$OUT"
    python scripts/eval_model.py       suite="$suite" preset=memory/smpd n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=test store_dir="$STORES/smpd" session_id="$SESS"       outdir="$OUT"
  done
done
```


---

## 5) Ablations & sensitivity sweeps (Milestone 9 — H6)

> These runs help attribute effects to specific components. We **re‑teach** for ablations that affect ingestion.

### 5.1) Gate toggles

- **HEI‑NW write‑gate OFF**

```bash
for n in "${SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    OUT="$RUNS/ablations/hei_nw_gate_off/episodic/${n}_${seed}"
    python scripts/eval_model.py       suite=episodic preset=memory/hei_nw n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=teach persist=true store_dir="$STORES/hei_nw_gate_off" session_id="$SESS"       memory.episodic.gate.enabled=false outdir="$OUT"
    python scripts/eval_model.py       suite=episodic preset=memory/hei_nw n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=test store_dir="$STORES/hei_nw_gate_off" session_id="$SESS"       memory.episodic.gate.enabled=false outdir="$OUT"
  done
done
```

- **SGC‑RSS relational gate OFF**

```bash
for n in "${SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    OUT="$RUNS/ablations/sgc_rss_gate_off/semantic/${n}_${seed}"
    python scripts/eval_model.py       suite=semantic preset=memory/sgc_rss n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=teach persist=true store_dir="$STORES/sgc_rss_gate_off" session_id="$SESS"       memory.relational.gate.enabled=false outdir="$OUT"
    python scripts/eval_model.py       suite=semantic preset=memory/sgc_rss n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=test store_dir="$STORES/sgc_rss_gate_off" session_id="$SESS"       memory.relational.gate.enabled=false outdir="$OUT"
  done
done
```

- **SMPD spatial gate OFF**

```bash
for n in "${SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    OUT="$RUNS/ablations/smpd_gate_off/spatial/${n}_${seed}"
    python scripts/eval_model.py       suite=spatial preset=memory/smpd n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=teach persist=true store_dir="$STORES/smpd_gate_off" session_id="$SESS"       memory.spatial.gate.enabled=false outdir="$OUT"
    python scripts/eval_model.py       suite=spatial preset=memory/smpd n="$n" seed="$seed" date="$DATE"       model="$MODEL" mode=test store_dir="$STORES/smpd_gate_off" session_id="$SESS"       memory.spatial.gate.enabled=false outdir="$OUT"
  done
done
```

## 5.2) OPTIONAL — Gate threshold sweeps (set lists or skip)

# Choose small, symmetric ranges around the defaults in configs:
#   - episodic tau default ≈ 0.5
#   - relational threshold default = 0.6  (configs/memory/relational.yaml)
#   - spatial block_threshold default = 1.0 (configs/memory/spatial.yaml)
EPISODIC_TAUS=(0.3 0.5 0.7)
RELATIONAL_THRESHOLDS=(0.4 0.6 0.8)
SPATIAL_BLOCK_THRESHOLDS=(0.5 1.0 2.0)

# Keep the sweep light to control runtime; adjust if you have more compute
SWEEP_SIZES=(200)
SWEEP_SEEDS=(1337 2025)

# --- HEI-NW (episodic) tau sweep ---
for thr in "${EPISODIC_TAUS[@]}"; do
  for n in "${SWEEP_SIZES[@]}"; do
    for seed in "${SWEEP_SEEDS[@]}"; do
      OUT="$RUNS/sweeps/epis_tau_${thr}/episodic/${n}_${seed}"
      python scripts/eval_model.py \
        suite=episodic preset=memory/hei_nw n="$n" seed="$seed" date="$DATE" \
        model="$MODEL" mode=teach persist=true store_dir="$STORES/hei_nw_tau_${thr}" session_id="$SESS" \
        memory.episodic.gate.tau="$thr" outdir="$OUT"
      python scripts/eval_model.py \
        suite=episodic preset=memory/hei_nw n="$n" seed="$seed" date="$DATE" \
        model="$MODEL" mode=test store_dir="$STORES/hei_nw_tau_${thr}" session_id="$SESS" \
        memory.episodic.gate.tau="$thr" outdir="$OUT"
    done
  done
done

# --- SGC-RSS (relational) threshold sweep ---
for thr in "${RELATIONAL_THRESHOLDS[@]}"; do
  for n in "${SWEEP_SIZES[@]}"; do
    for seed in "${SWEEP_SEEDS[@]}"; do
      OUT="$RUNS/sweeps/rel_thr_${thr}/semantic/${n}_${seed}"
      python scripts/eval_model.py \
        suite=semantic preset=memory/sgc_rss n="$n" seed="$seed" date="$DATE" \
        model="$MODEL" mode=teach persist=true store_dir="$STORES/sgc_rss_thr_${thr}" session_id="$SESS" \
        memory.relational.gate.threshold="$thr" outdir="$OUT"
      python scripts/eval_model.py \
        suite=semantic preset=memory/sgc_rss n="$n" seed="$seed" date="$DATE" \
        model="$MODEL" mode=test store_dir="$STORES/sgc_rss_thr_${thr}" session_id="$SESS" \
        memory.relational.gate.threshold="$thr" outdir="$OUT"
    done
  done
done

# --- SMPD (spatial) block_threshold sweep ---
for thr in "${SPATIAL_BLOCK_THRESHOLDS[@]}"; do
  for n in "${SWEEP_SIZES[@]}"; do
    for seed in "${SWEEP_SEEDS[@]}"; do
      OUT="$RUNS/sweeps/spa_block_${thr}/spatial/${n}_${seed}"
      python scripts/eval_model.py \
        suite=spatial preset=memory/smpd n="$n" seed="$seed" date="$DATE" \
        model="$MODEL" mode=teach persist=true store_dir="$STORES/smpd_block_${thr}" session_id="$SESS" \
        memory.spatial.gate.block_threshold="$thr" outdir="$OUT"
      python scripts/eval_model.py \
        suite=spatial preset=memory/smpd n="$n" seed="$seed" date="$DATE" \
        model="$MODEL" mode=test store_dir="$STORES/smpd_block_${thr}" session_id="$SESS" \
        memory.spatial.gate.block_threshold="$thr" outdir="$OUT"
    done
  done
done


---

## 6) Summaries & reports

```bash
# 6.1) Per‑preset summaries (CSV/JSON) under runs/$DATE/summaries
python scripts/summarize_runs.py "$RUNS" --out "$RUNS/summaries"

# 6.2) Markdown reports (+ optional plots) under reports/$DATE
python scripts/report.py --date "$DATE" --runs-dir runs --out-dir reports --data-dir data --plots --smoke
```


---

## 7) Milestone 9.5 — Systems consolidation (LoRA adapters)

### 7.1) Pre‑consolidation tests (memory OFF)

```bash
C_SUITES=(episodic semantic)  # spatial optional; starts with episodic for speed
C_SIZES=(50)                  # spot‑check size; expand if needed

for suite in "${C_SUITES[@]}"; do
  for n in "${C_SIZES[@]}"; do
    for seed in 1337; do
      OUT="$RUNS/consolidation/pre/$suite/${n}_${seed}"
      python scripts/test_consolidation.py         --phase pre --suite "$suite" --n "$n" --seed "$seed"         --model "$MODEL" --outdir "$OUT"
    done
  done
done
```

### 7.2) Train adapters from replayed stores

```bash
# Example: one adapter trained from the HEI‑NW store
python scripts/replay_consolidate.py   --store_dir "$STORES/hei_nw" --session_id "$SESS"   --outdir "$ADAPTERS/hei_nw_lora" --model "$MODEL"   --config configs/consolidation/lora_small.yaml --seed 1337
```

### 7.3) Post‑consolidation tests (memory OFF + adapter ON)

```bash
for suite in "${C_SUITES[@]}"; do
  for n in "${C_SIZES[@]}"; do
    for seed in 1337; do
      PRE="$RUNS/consolidation/pre/$suite/${n}_${seed}"
      OUT="$RUNS/consolidation/post/$suite/${n}_${seed}"
      python scripts/test_consolidation.py         --phase post --suite "$suite" --n "$n" --seed "$seed"         --model "$MODEL" --adapter "$ADAPTERS/hei_nw_lora"         --pre_dir "$PRE" --outdir "$OUT"
    done
  done
done
```

### 7.4) (Optional) Export or merge adapter

```bash
# Save raw adapter tensors or merge into base weights
python scripts/export_adapter.py --base-model "$MODEL" --adapter "$ADAPTERS/hei_nw_lora" --output "$ADAPTERS/hei_nw_lora_export"
python scripts/export_adapter.py --base-model "$MODEL" --adapter "$ADAPTERS/hei_nw_lora" --output "$ADAPTERS/hei_nw_merged" --merge
```


---

## 8) Final artifact roll‑up

```bash
# Re‑generate reports to include consolidation results
python scripts/report.py --date "$DATE" --runs-dir runs --out-dir reports --data-dir data --plots

echo "Done. See:"
echo "  - $RUNS        (raw run outputs)"
echo "  - $RUNS/summaries (per‑preset CSV/JSON)"
echo "  - $REPORTS     (Markdown summaries & plots)"
echo "  - $ADAPTERS    (trained adapters)"
```
