# 0) Purpose

A concrete, repeatable plan to **validate** HEI‑NW, SGC‑RSS, and SMPD on a single 12 GB GPU setup. It defines datasets, baselines, run matrix, metrics, ablations, file formats, and commands so Codex/CI and local runs yield comparable, auditable results.

# 1) Hypotheses (what we expect to improve)

- **HEI‑NW (episodic):** one‑shot episodic recall from partial cues with durability after replay; lower interference than long‑context.
- **SGC‑RSS (semantic/relational):** higher multi‑hop factual accuracy and lower contradiction rate; faster stabilization for schema‑fit items.
- **SMPD (spatial/procedural):** higher path success and lower path suboptimality; fewer steps via macro reuse on repeated tasks.
* **Relational gates:** reduce duplicate churn and hub growth **without degrading** multi‑hop accuracy; faster stabilization via reduced noise.
* **Spatial gates:** reduce graph/map growth (nodes/edges per 1k obs) and planning latency with unchanged success/suboptimality.

# 2) Baselines

Define three presets under `configs/eval/baselines/`:

- `core.yaml` — base model without memory or RAG.
- `rag.yaml` — vector DB over the same corpus used by stores (FAISS‑CPU), retrieved text concatenated to inputs.
- `longctx.yaml` — same base model with the longest feasible context (no memory modules).

Memory variants under `configs/eval/memory/`:

- `hei_nw.yaml`, `sgc_rss.yaml`, `smpd.yaml`, and `all.yaml` (combined).

# 3) Suites & generators

Implemented by `scripts/build_datasets.py`. All generators are **deterministic** with `seed`.

## 3.1 Episodic suite (HEI‑NW)

- **W4 stories:** short, templated stories with tuples (who, what, where, when). Example size: 100–1,000 items.
- **Partial‑cue queries:** (who\@where?), (what\@who?), etc., after N distractors.
- **Delay condition:** evaluate immediately and after one replay cycle (“sleep”).
- **Reward/pin flags:** mark salient or pinned episodes for write‑gate precision/recall tests.

## 3.2 Semantic/relational suite (SGC‑RSS)

- **Schema‑fit/mismatch facts:** generate frames (e.g., `buy(person,item,store,time)`) and off‑schema variants.
- **Multi‑hop questions:** 2–3 hop queries spanning linked tuples.
- **Contradictions:** inject conflicting facts; probe contradiction detection/avoidance.
- **Schema-fit labels over time:** tag each fact with a schema-fit flag and time index to track consolidation speed.

## 3.3 Spatial/procedural suite (SMPD)

- **Grid/graph worlds:** parameterisable N×N grids with random walls/obstacles
  and small graphs with labeled nodes/edges.
- **Planning tasks:** shortest path; “recall by place” questions.
- **Procedural macros:** repeated multi‑step scripts (4–6 steps) to allow macro distillation.
- **Sequential trajectories:** random walk paths for path-integration stress tests.

## 3.4 Generator coverage

`scripts/build_datasets.py` now emits the above features – reward/pin flags,
temporal schema-fit labels and sequential trajectories – providing full
coverage for the validation matrix below.

# 4) Run matrix

For each **suite**:

- Presets: `baselines/{core,rag,longctx}` and `memory/{hei_nw,sgc_rss,smpd}` (or `all` for combined).
- Sizes: {small: 50, medium: 200, large: 1,000} per suite.
- Seeds: {1337, 2025, 4242}.
- Replay: evaluate **pre‑replay** and **post‑replay** (1–3 cycles).

For relational and spatial suites, execute **paired runs** with gates `enabled=true` and `enabled=false` per (size, seed). Report ON→OFF deltas for duplicate rate and map growth alongside primary accuracy metrics.

# 5) Metrics

## 5.1 Primary

- **Episodic:** exact match (EM), F1, recall@{1,3}, robustness vs. distractors (EM drop per 10 distractors), ΔEM after replay.
- **Semantic:** multi‑hop accuracy, contradiction rate, time‑to‑stabilize (# replays to reach 95% of peak accuracy for schema‑fit).
- **Spatial:** success rate, path suboptimality (ratio to optimal), steps‑to‑solve; **Procedural:** steps reduction (%) with macros.

## 5.2 Compute & memory

- Tokens processed, wall‑clock runtime per 100 queries (CPU timing acceptable for comparison), estimated KV‑cache MB, retrieval calls, hit@K, avg K used, trace token count M, write rate %, avg S, +ms/step from retrieval.

## 5.3 Algorithm‑specific checks

| Module & algorithm | Intended enhancement | Verification | Metrics |
| --- | --- | --- | --- |
| **HEI‑NW**: k‑WTA sparsity + Hopfield completion | Partial‑cue recall with low interference | Run episodic suite; ablate `episodic.use_sparsity` and `episodic.use_completion` | ΔEM/F1 vs. baseline, recall@k |
| **HEI‑NW**: neuromodulated write gate | Store only salient/pinned episodes | Generate tasks with pin flags; compare writes with/without `episodic.use_gate` | write precision/recall, store size |
| **HEI‑NW**: CA2 replay scheduler | Reduce interference via prioritized replay | Compare ΔEM after replay vs. `replay.enabled=false` | ΔEM after 1–3 cycles |
| **SGC‑RSS**: schema fast‑track | Faster consolidation for schema‑fit facts | Semantic suite with schema-fit labels; ablate `relational.schema_fasttrack` | time-to-stabilize, multi-hop accuracy |
| **SGC‑RSS**: contradiction detection | Avoid storing conflicting facts | Semantic generator with contradictions | contradiction rate |
| **SMPD**: path integration | Robust localization over long trajectories | Spatial suite with sequential trajectories; ablate path integration | localization error, path success |
| **SMPD**: macro distillation | Reuse learned procedures | Spatial macro tasks; ablate `spatial.macros` | steps reduction %, success rate |

## 5.4 Gate telemetry & KPIs

* **Raw counters** (by memory): `attempts, inserted, aggregated, routed_to_episodic (relational), blocked_new_edges (spatial)`.
* **Derived:**
  * **Duplicate rate (relational):** `aggregated / attempts`.
  * **Map growth per 1k obs (spatial):** `nodes_added/1k`, `edges_added/1k`.
  * **Gating overhead:** Δ runtime/100 queries (gates ON vs OFF).
* **Integrity checks:** gates ON must not reduce **accuracy** by >1pp vs OFF at matched seeds/sizes.

# 6) Ablations (toggles)

Exposed in Hydra and consumed by the harness. Useful flags:

- `episodic.use_sparsity=false`
- `episodic.use_completion=false`
- `episodic.use_gate=false`
- `replay.enabled=false`
- `memory.runtime.enable_writes=false`
- `relational.schema_fasttrack=false`
- `spatial.macros=false`
- `relational.gate.enabled={true,false}`
- `spatial.gate.enabled={true,false}`

*(Optional sensitivity sweeps)*

- `relational.gate.threshold ∈ {0.5, 0.6, 0.7}`
- `spatial.gate.block_threshold ∈ {0.8, 1.0, 1.2}`

Run example:

```bash
python scripts/eval_bench.py suite=episodic +ablate=episodic.use_gate=false
```

When gates are disabled, ingestion reverts to pre‑gate behavior (no aggregation/routing). Use the same seeds and dataset sizes for ON/OFF comparisons.

# 7) Harness behavior (`scripts/eval_bench.py`)

- Loads a suite and a preset; constructs a **runner** object.
- Iterates tasks, obtains model outputs (baseline or memory‑augmented), computes metrics online.
- Writes per‑task logs and aggregates to:
  - `runs/<date>/<exp>/<suite>/metrics.json`
  - `runs/<date>/<exp>/<suite>/metrics.csv`
  - `runs/<date>/<exp>/<suite>/meta.json` (config hash, seeds, git SHA, model ID, preset, ablation flags, replay cycles).
- Supports `dry_run=true` for CI smoke tests (e.g., 5 tasks).
- Milestone 8a adds hooks logging retrieval hit rates, memory token shapes,
  and gate decisions; exercised via `scripts/smoke_8a.sh`.
- Include `metrics["gates"]` and `meta["config"].{relational.gate, spatial.gate}` in outputs. When present, `scripts/report.py` renders §5.4 tables; otherwise it skips.

## 7.1 File schemas

`metrics.json` (aggregate):

```json
{
  "suite": "episodic",
  "n": 200,
  "seed": 1337,
  "preset": "memory/hei_nw",
  "metrics": {
    "episodic": {"em": 0.72, "f1": 0.78, "r_at_1": 0.70, "r_at_3": 0.85, "delta_after_replay": 0.06},
    "compute": {"tokens": 81234, "runtime_s": 123.4, "kv_mem_mb": 512}
  }
}
```

`metrics.csv` (per‑task rows):

```
idx,prompt,answer,pred,correct,latency_ms,flags
0,"who@Cafe?","Alice","Alice",1,130.2,"pre_replay"
```

`meta.json`:

```json
{"git_sha":"...","model":"llama32-3b","config_hash":"...","ablate":{"episodic.use_gate":false}}
```

# 8) Replay protocol (post‑replay measurement)

- **Cycle definition:** sample from ReplayQueue with CA2‑style scheduler; run fine‑tuning on adapters only; freeze base.
- Evaluate the same suite **before** and **after** each cycle; report Δ.
- Suggested cycles: 1 and 3.

# 9) Commands (examples)

## 9.1 Build datasets

```bash
echo "build W4/Schema/Grid fixtures"
python scripts/build_datasets.py --out data/episodic.jsonl --suite episodic --size 1000 --seed 1337
python scripts/build_datasets.py --out data/semantic.jsonl --suite semantic --size 1000 --seed 1337 --hop-depth 3 --contradict
python scripts/build_datasets.py --out data/spatial.jsonl  --suite spatial  \
  --size 1000 --seed 1337 --grid-size 5 --obstacle-density 0.2
```

The semantic command above emits 2–3 hop chains linking people, items, stores
and cities. Using ``--hop-depth 3`` adds an item→store hop, while
``--contradict`` inserts a conflicting store location that the query
disambiguates.

## 9.2 Run baselines

```bash
python scripts/eval_bench.py suite=episodic preset=baselines/core   n_trials=200 seed=1337
python scripts/eval_bench.py suite=semantic preset=baselines/rag    n_trials=200 seed=1337
python scripts/eval_bench.py suite=spatial  preset=baselines/longctx n_trials=200 seed=1337
```

## 9.3 Run memory variants

```bash
python scripts/eval_bench.py suite=episodic preset=memory/hei_nw  n_trials=200 seed=1337
python scripts/eval_bench.py suite=semantic preset=memory/sgc_rss n_trials=200 seed=1337
python scripts/eval_bench.py suite=spatial  preset=memory/smpd    n_trials=200 seed=1337
```

## 9.4 Combined + ablations + replay

```bash
python scripts/eval_bench.py suite=episodic preset=memory/all n_trials=200 seed=1337
python scripts/eval_bench.py suite=episodic preset=memory/hei_nw +ablate=replay.enabled=false
python scripts/eval_bench.py suite=episodic preset=memory/hei_nw eval.post_replay_cycles=1
```

# 10) Reporting (`scripts/report.py`)

- Aggregates all `metrics.json`/`metrics.csv` under `runs/**`.
- Produces a markdown table and simple charts (optional) comparing presets.
- Outputs to `reports/<date>/<suite>/summary.md`.
* **Gate Telemetry table** (per suite): columns `mem, attempts, inserted, aggregated, routed_to_episodic/blocked_new_edges`.
* **ON/OFF delta table** when both present for a date: `duplicate_rateΔ`, `nodesΔ/1k`, `edgesΔ/1k`, and `runtimeΔ/100q` with 95% CIs if multiple seeds.

*(If only one condition present, omit delta gracefully.)*

## 10.1 Example output table (markdown)

```
| Suite     | Preset             | EM   | R@1 | Δ after replay | Runtime/100q |
|-----------|--------------------|------|-----|----------------|--------------|
| Episodic  | baselines/core     | 0.41 | 0.39| –              | 72 s         |
| Episodic  | memory/hei_nw      | 0.58 | 0.57| +0.06          | 81 s         |
```

# 11) Success criteria (v0 targets)

- **HEI‑NW:** +15pp EM over **core** on partial‑cue; +5pp over **RAG**; positive Δ after replay ≥ +3pp.
- **SGC‑RSS:** ≥10pp multi‑hop accuracy over **core**; contradiction rate ≤50% of **core**.
- **SMPD:** ≥90% path success on 5×5; ≥20% steps reduction with macros.
* **Relational:** duplicate rate reduced by ≥30% (ON vs OFF) with multi‑hop accuracy change within ±1pp.
* **Spatial:** edges/1k reduced by ≥25% with success rate within ±1pp and suboptimality unchanged (±0.02 absolute).
* **Overhead:** gating overhead ≤10% runtime/100 queries.

# 12) CI & Codex scope

- CI runs `make lint` and a **dry_run** of the harness on 5 tasks per suite across all presets (ensures plumbing only).
- Codex PRs should include updated fixtures/tests and leave heavy training/eval to local GPU runs.

# 13) Reproducibility

- Fix seeds (Python/NumPy/Torch). Persist generator seeds and config hashes in `meta.json`.
- Avoid network calls; keep all data local.
- Checkpoint adapters and store commit SHAs alongside results.

# 14) Troubleshooting

- **Low episodic gains:** increase k, tune write threshold τ, ensure gating uses model logprobs not only embeddings.
- **High contradiction rate:** raise tuple confidence threshold; prefer schema‑fit fast‑track; add provenance rollbacks.
- **Map explosion:** enable node merge by cosine similarity, set TTL for stale nodes.
- **CI timeouts:** use `n_trials=5` and `dry_run=true`.

# 15) Directory conventions

```
runs/
  2025-08-13/
    hei_nw/episodic/
      metrics.json
      metrics.csv
      meta.json
    sgc_rss/semantic/
    smpd/spatial/
reports/
  2025-08-13/
    episodic/summary.md
    semantic/summary.md
    spatial/summary.md
```

# 15) Harness implementation requirements

- **Real harness:** `scripts/eval_model.py` (new) must:
  - Load base model + active LoRA adapters, wire Episodic/Relational/Spatial adapters into forward.
  - Load suite JSONLs from `data/` according to `configs/eval/*/*.yaml`.
  - Run retrieval/replay when memory presets are selected (`configs/eval/memory/*.yaml`), otherwise skip for baselines.
  - Emit `metrics.json`, `metrics.csv`, and `meta.json` to `runs/<date>/<preset>/<suite>/` (same schema as §7.1).
  - Support pre-replay vs post-replay evaluation (1–3 cycles).

- **CI stub (keep):** `scripts/eval_bench.py` continues to operate for plumbing with `dry_run=true`.

- **Data splits & formatting:** Use the provided JSONL suites in `data/` with deterministic seeds and sizes {50, 200, 1000}. Ensure clear train/val/test separation.

# 16) Command matrix (examples)

**Baselines (plumbing):**

```bash
python scripts/eval_bench.py suite=episodic preset=baselines/core n=5 seed=1337 dry_run=true
```

**Real-model smoke runs (n=50 per suite):**

```bash
python scripts/eval_model.py suite=episodic preset=memory/hei_nw n=50 seed=1337 replay.cycles=1
python scripts/eval_model.py suite=semantic preset=memory/sgc_rss n=50 seed=1337 replay.cycles=1
python scripts/eval_model.py suite=spatial  preset=memory/smpd    n=50 seed=1337 replay.cycles=1
```

**Report aggregation:**

```bash
python scripts/report.py --date YYYYMMDD
```

```bash
# Paired relational ON/OFF
python scripts/eval_model.py suite=semantic preset=memory/sgc_rss n=200 seed=1337 relational.gate.enabled=true
python scripts/eval_model.py suite=semantic preset=memory/sgc_rss n=200 seed=1337 relational.gate.enabled=false

# Paired spatial ON/OFF
python scripts/eval_model.py suite=spatial  preset=memory/smpd    n=200 seed=1337 spatial.gate.enabled=true
python scripts/eval_model.py suite=spatial  preset=memory/smpd    n=200 seed=1337 spatial.gate.enabled=false

# Sensitivity (optional)
python scripts/eval_model.py suite=semantic preset=memory/sgc_rss n=200 seed=1337 relational.gate.threshold=0.5
python scripts/eval_model.py suite=spatial  preset=memory/smpd    n=200 seed=1337 spatial.gate.block_threshold=1.2
```

---
# 17) Memory-trace ablations & metrics

**Ablations:**
- `traces=off` (adapters wired but given empty memory),
- `traces=episodic|semantic|spatial` (alone),
- `traces=all`.

**Metrics additions:**
- Retrieval: hit@K, avg K used, trace token count M.
- Gating: write rate %, avg S, replay contribution % (if used).
- Overhead: +ms/step from retrieval.

**Command examples:**
python scripts/eval_model.py suite=episodic preset=memory/hei_nw n=200 seed=1337 traces=off
python scripts/eval_model.py suite=episodic preset=memory/hei_nw n=200 seed=1337 traces=episodic
