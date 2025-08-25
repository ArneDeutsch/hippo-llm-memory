# Milestone 9 — Execution Plan (Reworked)

Status target: **Start of Milestone 9** (Milestone 8 complete).  
Goal: run **memory‑augmented evaluations** (HEI‑NW, SGC‑RSS, SMPD, and ALL) with ablations, record
compute and memory telemetry, and produce an aggregate report that compares against **real**
baselines.

## 1) Summary

Milestone 9 contains work for both Codex (code‑writing agent) and the human collaborator. Codex
prepares scripts, configs and docs so the human only needs to execute deterministic commands and
commit the resulting artifacts.

## 2) Matrix and configs

* **Suites:** `episodic`, `semantic`, `spatial`
* **Sizes:** 50 / 200 / 1000 (ablations use **200**)
* **Seeds:** 1337 / 2025 / 4242
* **Baselines:** `baselines/core` (no memory)
* **Baselines:** `baselines/core` (no memory)
* `baselines/rag` and `baselines/longctx` **deferred** – current datasets fit in prompt so retrieval/long context not required yet
* **Memories:** `memory/hei_nw`, `memory/sgc_rss`, `memory/smpd`, `memory/all`
* **Ablations:** per‑memory toggles of key knobs

Configs live in `configs/eval/{baselines,memory}/*.yaml`.

## 3) Task breakdown

### 3.1 Codex tasks

Codex performs all code and documentation changes:

1. **Readiness patches** – guided by `CODEX_PROMPTS_M9_READINESS.md` (Prompts 1‑6):
   - add compute telemetry (`time_ms_per_100`, `rss_mb`, `latency_ms_mean`, `tokens`) to evaluation outputs,
   - enrich `meta.json` with commit hash, Python/OS info and config hash,
   - emit `data/MANIFEST.json` after dataset generation,
   - generate `reports/<DATE>/index.md` and `reports/<DATE>/smoke.md`,
   - ensure scripts support `+run_matrix=true` and write telemetry fields.
2. **Documentation updates**
   - update `EVAL_PLAN.md` with the telemetry fields and reference the roll‑up index,
   - update `DESIGN.md` to describe telemetry exposure and the ablation knobs.
3. **CLI examples and instructions**
   - provide in this plan a tested CLI cheat‑sheet for the human,
   - document which files are expected after each command.

Codex commits all modified scripts and docs. No large run artifacts are committed by Codex.

### 3.2 Human tasks

After Codex's patches are merged, the human executes the prepared scripts and commits artifacts.

1. **Setup**
   - `make install-dev`
   - optional: `make datasets DATE=<DATE>` (ensures `data/MANIFEST.json` is updated)
2. **Real baselines**
   - `python scripts/eval_model.py preset=baselines/core +run_matrix=true date=<DATE> model=<MODEL_NAME> outdir=runs/<DATE>/baselines/core/<MODEL_SLUG>`
   - commit: `runs/<DATE>/baselines/core/**/metrics.{json,csv}`, matching `meta.json`, and
     updated `reports/<DATE>/*`
3. **Memory variants**
   - `python scripts/eval_model.py preset=memory/hei_nw +run_matrix=true date=<DATE> model=<MODEL_NAME> outdir=runs/<DATE>/memory/hei_nw/<MODEL_SLUG>`
   - `python scripts/eval_model.py preset=memory/sgc_rss +run_matrix=true date=<DATE> model=<MODEL_NAME> outdir=runs/<DATE>/memory/sgc_rss/<MODEL_SLUG>`
   - `python scripts/eval_model.py preset=memory/smpd +run_matrix=true date=<DATE> model=<MODEL_NAME> outdir=runs/<DATE>/memory/smpd/<MODEL_SLUG>`
   - `python scripts/eval_model.py preset=memory/all suite=all n=200 seeds='[1337,2025,4242]' date=<DATE> model=<MODEL_NAME> outdir=runs/<DATE>/memory/all/<MODEL_SLUG>`
   - commit run directories and reports as above
4. **Ablations (n=200, 3 seeds)**
   - `python scripts/eval_model.py preset=memory/hei_nw n=200 seeds='[1337,2025,4242]' date=<DATE> model=<MODEL_NAME> outdir=runs/<DATE>/memory/hei_nw/<MODEL_SLUG> gate.enabled=false hopfield.enabled=false`
   - `python scripts/eval_model.py preset=memory/sgc_rss n=200 seeds='[1337,2025,4242]' date=<DATE> model=<MODEL_NAME> outdir=runs/<DATE>/memory/sgc_rss/<MODEL_SLUG> schema.fast_track=false`
   - `python scripts/eval_model.py preset=memory/smpd n=200 seeds='[1337,2025,4242]' date=<DATE> model=<MODEL_NAME> outdir=runs/<DATE>/memory/smpd/<MODEL_SLUG> macro.distill=false`
   - commit ablation run directories and resulting summaries
5. **Reporting**
   - `python scripts/report.py --date <DATE>`
   - commit: `reports/<DATE>/index.md`, `reports/<DATE>/smoke.md`, and
     `reports/<DATE>/<suite>/summary.md`

Human commits only the small JSON/CSV/MD artifacts; large models or caches remain untracked.

## 4) CLI cheat‑sheet

```
# install deps
make install-dev

# (optional) dataset regeneration
make datasets DATE=<DATE>

# core baseline
python scripts/eval_model.py preset=baselines/core +run_matrix=true date=<DATE> model=meta-llama/Llama-3.2-3B outdir=runs/<DATE>/baselines/core/meta-llama_Llama-3.2-3B
python scripts/eval_model.py preset=baselines/core +run_matrix=true date=<DATE> model=microsoft/Phi-3-mini-4k-instruct outdir=runs/<DATE>/baselines/core/microsoft_Phi-3-mini-4k-instruct
python scripts/eval_model.py preset=baselines/core +run_matrix=true date=<DATE> model=Qwen/Qwen2-1.5B outdir=runs/<DATE>/baselines/core/Qwen_Qwen2-1.5B

# memory variants
python scripts/eval_model.py preset=memory/hei_nw +run_matrix=true date=<DATE> model=meta-llama/Llama-3.2-3B outdir=runs/<DATE>/memory/hei_nw/meta-llama_Llama-3.2-3B
python scripts/eval_model.py preset=memory/hei_nw +run_matrix=true date=<DATE> model=microsoft/Phi-3-mini-4k-instruct outdir=runs/<DATE>/memory/hei_nw/microsoft_Phi-3-mini-4k-instruct
python scripts/eval_model.py preset=memory/hei_nw +run_matrix=true date=<DATE> model=Qwen/Qwen2-1.5B outdir=runs/<DATE>/memory/hei_nw/Qwen_Qwen2-1.5B
python scripts/eval_model.py preset=memory/sgc_rss +run_matrix=true date=<DATE> model=meta-llama/Llama-3.2-3B outdir=runs/<DATE>/memory/sgc_rss/meta-llama_Llama-3.2-3B
python scripts/eval_model.py preset=memory/sgc_rss +run_matrix=true date=<DATE> model=microsoft/Phi-3-mini-4k-instruct outdir=runs/<DATE>/memory/sgc_rss/microsoft_Phi-3-mini-4k-instruct
python scripts/eval_model.py preset=memory/sgc_rss +run_matrix=true date=<DATE> model=Qwen/Qwen2-1.5B outdir=runs/<DATE>/memory/sgc_rss/Qwen_Qwen2-1.5B
python scripts/eval_model.py preset=memory/smpd +run_matrix=true date=<DATE> model=meta-llama/Llama-3.2-3B outdir=runs/<DATE>/memory/smpd/meta-llama_Llama-3.2-3B
python scripts/eval_model.py preset=memory/smpd +run_matrix=true date=<DATE> model=microsoft/Phi-3-mini-4k-instruct outdir=runs/<DATE>/memory/smpd/microsoft_Phi-3-mini-4k-instruct
python scripts/eval_model.py preset=memory/smpd +run_matrix=true date=<DATE> model=Qwen/Qwen2-1.5B outdir=runs/<DATE>/memory/smpd/Qwen_Qwen2-1.5B
python scripts/eval_model.py preset=memory/all suite=all n=200 seeds='[1337,2025,4242]' date=<DATE> model=meta-llama/Llama-3.2-3B outdir=runs/<DATE>/memory/all/meta-llama_Llama-3.2-3B

# ablations (n=200)
python scripts/eval_model.py preset=memory/hei_nw n=200 seeds='[1337,2025,4242]' date=<DATE> model=meta-llama/Llama-3.2-3B outdir=runs/<DATE>/memory/hei_nw/meta-llama_Llama-3.2-3B gate.enabled=false hopfield.enabled=false
python scripts/eval_model.py preset=memory/sgc_rss n=200 seeds='[1337,2025,4242]' date=<DATE> model=meta-llama/Llama-3.2-3B outdir=runs/<DATE>/memory/sgc_rss/meta-llama_Llama-3.2-3B schema.fast_track=false
python scripts/eval_model.py preset=memory/smpd n=200 seeds='[1337,2025,4242]' date=<DATE> model=meta-llama/Llama-3.2-3B outdir=runs/<DATE>/memory/smpd/meta-llama_Llama-3.2-3B macro.distill=false

# aggregate reports
python scripts/report.py --date <DATE>
```

**Model selection:** With Option B, set the base model via `model=...`. Presets no longer fix the model.

**Output layout:** Use `outdir=.../<model_slug>` to run several models on the same `<DATE>` without overwriting.

## 5) Artifacts & layout

```
runs/<DATE>/
  baselines/core/<suite>/<size>_<seed>/
    metrics.json, metrics.csv, meta.json
  memory/<preset>/<suite>/<size>_<seed>/
    metrics.json, metrics.csv, meta.json
reports/<DATE>/
  index.md
  smoke.md
  episodic/summary.md
  semantic/summary.md
  spatial/summary.md
  assets/*.png   # optional plots
data/MANIFEST.json
```

## 6) Gate (Definition of Done)

- **Codex:** telemetry and documentation patches merged; scripts generate `MANIFEST.json` and roll‑up
  reports.
- **Human:** run matrix complete for baselines and each memory (sizes 50 & 200, all seeds), combined
  run at n=200, ablations at n=200, and all metrics/reports committed.
- `reports/<DATE>/index.md` summarises baselines vs memories and links per‑suite summaries and
  `smoke.md`.
- `data/MANIFEST.json` present and tracked.

## 7) Risks & mitigations

- **Runtime variance on CPU:** pin `OMP_NUM_THREADS=1` and fix seeds; collect time per 100 items.
- **Memory growth unbounded:** enforce max store sizes in configs and log prunes.
- **Plotting deps missing:** roll‑up works without matplotlib; plots are optional.

## 8) Impact on DESIGN.md / EVAL_PLAN.md

Codex updates these documents so telemetry fields and ablation knobs are described.

## 9) Rollback

If a memory run fails, proceed with the others and still generate the roll‑up. The gate requires the
real baseline plus at least two memory variants at 50/200; file an issue for missing runs.
