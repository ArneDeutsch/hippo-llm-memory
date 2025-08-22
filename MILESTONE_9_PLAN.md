# Milestone 9 — Execution Plan

> Status target: **Start of Milestone 9** (Milestone 8 complete).  
> Goal: run **memory‑augmented evaluations** (HEI‑NW, SGC‑RSS, SMPD, and ALL) with ablations, record compute & memory telemetry, and produce an aggregate report that compares against **real** baselines.

---

## 1) Summary

We will:
- establish **real (non‑oracle) baselines** with `scripts/eval_model.py` (no memory) over the same matrix used in Milestone 8 (sizes **50 / 200 / 1000** × seeds **1337 / 2025 / 4242**),
- run **memory variants** `memory/hei_nw`, `memory/sgc_rss`, `memory/smpd`, and `memory/all`,
- run **ablations** at `n=200` × 3 seeds per memory (toggle key knobs),
- log **compute telemetry** (`time_ms_per_100`, `rss_mb`, `latency_ms_mean`) and **memory telemetry** (write‑gate accept %, store growth, retrieval hit@k, replay cycles),
- generate a top‑level roll‑up `reports/<DATE>/index.md` that links per‑suite summaries and `smoke.md`.

GPU is recommended but not strictly required for tiny models (e.g., `models/tiny-gpt2`).

---

## 2) Matrix and configs

**Suites:** `episodic`, `semantic`, `spatial`  
**Sizes:** 50 / 200 / 1000 (ablations use **200** only)  
**Seeds:** 1337 / 2025 / 4242  
**Baselines (real):** `baselines/core` (no memory)  
**Memories:** `memory/hei_nw`, `memory/sgc_rss`, `memory/smpd`, `memory/all`  
**Ablations:** per‑memory toggle of key knobs (see task 3 below)

Configs live in `configs/eval/{baselines,memory}/*.yaml`.

---

## 3) Codex‑executable tasks

### 3.1 Readiness fixes (run first)

Use the prompts in `CODEX_PROMPTS_M9_READINESS.md` (Prompts 1–6) to:
- add compute telemetry (time/RSS/latency),
- make `meta.json` self‑contained,
- create the roll‑up `index.md`,
- emit `data/MANIFEST.json`,
- and generate `reports/<DATE>/smoke.md`.

**Gate for readiness:** re-run Milestone 8 sweep to verify new fields and reports:
```bash
make install-dev
DATE=<DATE> make eval-baselines
python scripts/report.py --date <DATE>
```

### 3.2 Real baselines

Run `eval_model.py` without memory to establish non‑oracle baselines (same matrix as M8):
```bash
# core baseline (no memory) across suites/sizes/seeds
python scripts/eval_model.py preset=baselines/core +run_matrix=true date=<DATE>
python scripts/report.py --date <DATE>
```
*Definition of done:* EM/tokens/compute fields are recorded for every `(suite, size, seed)`; reports updated.

### 3.3 Memory variants

Run each memory preset:
```bash
# episodic
python scripts/eval_model.py preset=memory/hei_nw +run_matrix=true date=<DATE>

# relational
python scripts/eval_model.py preset=memory/sgc_rss +run_matrix=true date=<DATE>

# spatial/procedural
python scripts/eval_model.py preset=memory/smpd +run_matrix=true date=<DATE>

# combined (n=200 only to control cost)
python scripts/eval_model.py preset=memory/all suite=all n=200 seeds='[1337,2025,4242]' date=<DATE>
```

**Telemetry requirements (enforced in code):**
- `metrics.memory.write.accept_rate`, `metrics.memory.store.size_items/tokens`, `metrics.memory.read.hit_at_k` (k=1,5), `metrics.memory.replay.cycles`.
- `metrics.compute.time_ms_per_100`, `metrics.compute.rss_mb`, `metrics.compute.latency_ms_mean`, and `metrics.compute.tokens`.

### 3.4 Ablations (n=200, 3 seeds)

Run selective toggles per memory (examples; exact flags depend on adapters):
```bash
# HEI‑NW: gate OFF, Hopfield OFF
python scripts/eval_model.py preset=memory/hei_nw n=200 seeds='[1337,2025,4242]' date=<DATE> gate.enabled=false hopfield.enabled=false

# SGC‑RSS: schema fast‑track OFF
python scripts/eval_model.py preset=memory/sgc_rss n=200 seeds='[1337,2025,4242]' date=<DATE> schema.fast_track=false

# SMPD: macro distillation OFF
python scripts/eval_model.py preset=memory/smpd n=200 seeds='[1337,2025,4242]' date=<DATE> macro.distill=false
```

*Definition of done:* ablation metrics exist and show directional effects; reports include an **Ablations** section comparing ON vs OFF at `n=200`.

---

## 4) CLI cheat‑sheet

```bash
# install deps
make install-dev

# (optional) generate datasets
make datasets DATE=<DATE>

# run real baseline across the full matrix
python scripts/eval_model.py preset=baselines/core +run_matrix=true date=<DATE>

# run memory variants
python scripts/eval_model.py preset=memory/hei_nw +run_matrix=true date=<DATE>
python scripts/eval_model.py preset=memory/sgc_rss +run_matrix=true date=<DATE>
python scripts/eval_model.py preset=memory/smpd +run_matrix=true date=<DATE>

# combined (n=200 only)
python scripts/eval_model.py preset=memory/all suite=all n=200 seeds='[1337,2025,4242]' date=<DATE>

# generate reports (includes index.md and smoke.md)
python scripts/report.py --date <DATE>
```

---

## 5) Artifacts & layout

```
runs/<DATE>/
  baselines/core/<suite>/
    metrics.json, metrics.csv, meta.json
  memory/<preset>/<suite>/
    metrics.json, metrics.csv, meta.json
reports/<DATE>/
  index.md
  smoke.md
  episodic/summary.md
  semantic/summary.md
  spatial/summary.md
  assets/*.png   # optional plots
data/
  MANIFEST.json
```

---

## 6) Gate (Definition of Done)

- Real baseline (`baselines/core`) exists for **all** suites × sizes × seeds with compute telemetry present.
- Each memory variant (`hei_nw`, `sgc_rss`, `smpd`) has runs at **50** and **200** (all seeds) with telemetry + memory stats.
- Combined `memory/all` exists at **n=200** (3 seeds).
- Ablation results exist at **n=200** for each memory showing clear directional effects.
- `reports/<DATE>/index.md` summarises baselines vs memories and links per‑suite summaries + `smoke.md`.
- `data/MANIFEST.json` present.

---

## 7) Risks & mitigations

- **Runtime variance on CPU:** pin `OMP_NUM_THREADS=1` and use the same seeds; collect time per 100 items to smooth variance.
- **Memory growth unbounded:** enforce max store sizes in configs; log prunes explicitly.
- **Plotting deps missing:** roll-up works without matplotlib; plots are optional.

---

## 8) Impact on DESIGN.md / EVAL_PLAN.md

- **EVAL_PLAN.md:** add the compute/memory telemetry fields to the schema section and reference the roll‑up index.
- **DESIGN.md:** document how each memory exposes telemetry (write/read/replay) and note any knobs used in ablations.

---

## 9) Rollback

If a memory run fails, proceed with the others and still generate the roll‑up. The gate requires at least the real baseline + two memory variants at 50/200; file an issue for the missing one.
