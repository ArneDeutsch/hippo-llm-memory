# 0) Purpose

A concrete, repeatable plan to **validate** HEI‑NW, SGC‑RSS, and SMPD on a
single 12 GB GPU setup. It defines datasets, baselines, run matrix, metrics,
ablations, file formats, and commands so Codex/CI and local runs yield
comparable, auditable results.

Evaluation code, metrics, reporting, and synthetic tasks reside in the
`hippo_eval` package. The former `hippo_mem.*` modules are maintained as shims
that issue `DeprecationWarning`.
# 0.1) Quick smoke test

For a fast end‑to‑end check at `n=50`, `seed=1337`, run:

```bash
bash scripts/smoke_n50.sh
```

This script performs baselines → memory (teach+replay) → report and fails if
`post_*` metrics are missing.

# 0.2) Decision criteria & stop-go gate

A run is **meaningful** only if it satisfies the [Meaningful Run Contract](EVAL_PROTOCOL.md#meaningful-run-contract). At minimum:

- Baseline EM for each suite falls within the expected ranges (see §1.2).
- `metrics.json` for memory runs includes non-NaN `pre_*` fields.
- Persisted stores carry `store_meta.json` with `source != "stub"` and `replay.samples >= 1`.

If any criterion fails, abort the run, fix the configuration, and re-run.

# 1) Hypotheses (what we expect to improve)

- **HEI‑NW (episodic):** one‑shot episodic recall from partial cues with durability after replay; lower interference than long‑context.
- **SGC‑RSS (semantic/relational):** higher multi‑hop factual accuracy and lower contradiction rate; faster stabilization for schema‑fit items.
- **SMPD (spatial/procedural):** higher path success and lower path suboptimality; fewer steps via macro reuse on repeated tasks.
* **Relational gates:** reduce duplicate churn and hub growth **without degrading** multi‑hop accuracy; faster stabilization via reduced noise.
* **Spatial gates:** reduce graph/map growth (nodes/edges per 1k obs) and planning latency with unchanged success/suboptimality.

## 1.1) Metric selection per suite

- **semantic**: `EM(raw)` is the primary metric; `EM(norm)` is reported for
  context.
- **other suites**: `EM(norm)` remains primary with `EM(raw)` optional.

## 1.2) Expected baseline EM ranges

Approximate EM ranges for `n=50`. The default `semantic` and `episodic_cross`
splits saturate and are kept only as smoke tests. Use their `hard` variants for
meaningful evaluation:

| Suite            | Expected EM range |
|------------------|------------------|
| episodic         | 0.20–0.40        |
| semantic (hard)  | 0.55–0.80        |
| spatial          | 0.05–0.20        |

Runs outside these ranges likely indicate misconfiguration or saturated datasets.

The legacy `semantic` and `episodic_cross` splits remain in `data/` for quick
smoke tests but are marked “non-informative for uplift”.

## 1.3) Success bars

- **episodic:** `ΔEM(core→memory) ≥ 0.10` and `EM(memory) ≥ EM(longctx)` with
  `memory_hit_rate ≥ 0.3`.
- **semantic:** EM uplift over `baselines/longctx` on the `semantic(hard)` split.
- **spatial:** `EM ≥ 0.10` or `steps_to_goal` reduced by ≥20%.

# 2) Baselines

Define four presets under `configs/eval/baselines/`:

- `core.yaml` — base model without memory or retrieval.
- `rag.yaml` — nearest‑neighbour text retrieval concatenated to the prompt.
- `longctx.yaml` — same base model with the longest feasible context (no memory modules).
- `span_short.yaml` — chat template ON with `max_new_tokens≤8` for exact‑match decoding.

Memory presets mirror this `span_short` decoding profile when EM/Span metrics are reported.

Baseline runs target small instruction-tuned models:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `microsoft/Phi-3.5-mini-instruct` *(128k context)*
- `meta-llama/Llama-3.2-3B-Instruct` *(128k context)*

Phi‑3.5‑mini and Llama‑3.2‑3B support large contexts for future long‑context evaluations.

**Milestone 8 run matrix**

- **Presets:** `baselines/core`, `baselines/rag`, `baselines/longctx`, `baselines/span_short`.
- **Suites:** `episodic`, `semantic`, `spatial`.
- **Sizes:** `50`, `200`, `1000` items.
- **Seeds:** `1337`, `2025`, `4242`.

Each combo writes `metrics.json`, `metrics.csv`, and `meta.json` under
`runs/<DATE>/<preset>/<suite>/<size>_<seed>/`.

**How to reproduce Milestone 8**

```bash
make datasets DATE=20250822
make eval-baselines DATE=20250822 && python scripts/report.py --date 20250822
```

Memory variants under `configs/eval/memory/`:

- `hei_nw.yaml`, `sgc_rss.yaml`, `smpd.yaml`, and `all.yaml` (combined).

# 3) Suites & generators

Implemented by `scripts/build_datasets.py`. All generators are **deterministic** with `seed`.

Each suite provides a minimal `n=50` split for smoke and cross‑session experiments.

Generators expose `profile` options forwarded via `dataset_profile`. Profiles
tune difficulty, expected baseline EM, and memory uplift (see `EVAL_PROTOCOL.md` §Dataset profiles):

| Suite          | `base` evaluates                      | `hard` adds to probe                       |
| -------------- | ------------------------------------- | ------------------------------------------ |
| episodic       | single‑hop recall                     | extra distractors and swaps                |
| episodic_cross | flush‑delimited recall                | extra distractors, limited entity pool     |
| semantic       | schema‑fit triples                   | near‑miss relations, contradictions       |
| spatial        | simple grids and paths                | dead‑ends and alternative routes          |

Select `dataset_profile=hard` when baselines exceed 0.98 EM. `semantic` and
`episodic_cross` defaults already saturate; their easy splits remain only for
smoke tests.

The **default** profile targets the baseline EM ranges in §1.2 and should show a memory uplift of roughly +0.2 EM.
The **hard** profile drives baseline EM toward zero but still expects ≥ +0.2 uplift when memory is enabled.

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

### 3.1 Episodic variants (memory-dependent)

We add three episodic variants to force memory usage beyond trivial one-shot extraction:
- **`episodic_multi`** — multi-turn episodes with distractors and last-mention-wins corrections.
- **`episodic_cross`** — cross-episode recall after session flush; facts only available via store replay.
- **`episodic_capacity`** — episodes longer than the decoding context budget; retrieval required.
Generators live in `hippo_eval/datasets.py` and are addressable via the CLI
(`scripts/build_datasets.py`).

# 4) Run matrix

Set a single `RUN_ID` and source `scripts/env_prelude.sh` so `DATE=$RUN_ID` and
common paths (`$RUNS`, `$STORES`, `$REPORTS`) are defined. Derive deterministic
session ids such as `hei_$RUN_ID` so replay phases find the correct stores.

For each **suite**:

- Presets: `baselines/{core,rag,longctx,span_short}` and `memory/{hei_nw,sgc_rss,smpd}` (or `all` for combined).
- Episodic variants (`episodic_multi`, `episodic_cross`, `episodic_capacity`) run under `baselines/span_short` and all `memory/*` presets.
- Sizes: `n ∈ {50, 200, 1000}` per suite.
- Seeds: `{1337, 2025, 4242}`.
- Replay: evaluate **pre‑replay** and **post‑replay** (1–3 cycles).

For relational and spatial suites, execute **paired runs** with gates `enabled=true` and `enabled=false` per (size, seed). Report ON→OFF deltas for duplicate rate and map growth alongside primary accuracy metrics.

Gate toggles are surfaced via CLI flags such as
`episodic.gate.enabled=false`, `relational.gate.enabled=false`, and
`spatial.gate.enabled=false`.

# 4.1 Cross-session protocol & persistence

Memory stores implement `save(dir, session_id)` / `load(dir, session_id)` and the harness
exposes overrides `store_dir=…`, `session_id=…`, `persist=true`, and `mode={teach,replay,test}`.
A typical experiment:

1. **Teach** – present facts with gates enabled; metrics are not graded; stores are saved.
2. **Replay** – optional background rehearsal using a scheduler policy.
3. **Test** – new process loads stores and answers queries without re-presenting facts.

Replay policies are configurable via `replay.policy={uniform,priority,spaced}`,
`replay.rate`, `replay.noise_level`, and `replay.max_items`.

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

## 5.4 Answer format & normalization policy
All suites using span extraction MUST follow a short‑answer policy.
**Model instruction:** “Answer with the exact shortest span; no punctuation; no extra words.”
**Metrics:** We report three scores side‑by‑side:
- **EM (raw):** `pred.strip() == gold` (exact string match).
- **EM (normalized):** lower‑case, strip punctuation and articles (`a|an|the`)
  from both sides before comparison. Normalizer defined in
  `hippo_eval/metrics/scoring.py`.
- **Token‑F1:** whitespace token F1.
Diagnostics we also log: `pred_len`, `gold_len`, `overlong` (pred_len > gold_len), and `format_violation` (any terminal punctuation or contains a period).

## 5.5 Gate telemetry & KPIs

* **Raw counters** (by memory): `attempts, inserted, aggregated, routed_to_episodic (relational), blocked_new_edges (spatial)`.
* **Retrieval counters:** `requests`, `hits`, `r@5`, `r@10`.
* **Derived:**
  * **Duplicate rate (relational):** `aggregated / attempts`.
  * **Map growth per 1k obs (spatial):** `nodes_added/1k`, `edges_added/1k`.
  * **Gating overhead:** Δ runtime/100 queries (gates ON vs OFF).
  * **Refusal rate:** fraction of predictions matching refusal regexes.
* **Integrity checks:**
  * gates ON must not reduce **accuracy** by >1pp vs OFF at matched seeds/sizes.
  * `retrieval.requests > 0` and `refusal_rate ≤ 0.5` for memory presets.

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
- Include `metrics["gates"]` and `meta["config"].{relational.gate, spatial.gate}` in outputs. When present, `scripts/report.py` renders §5.5 tables; otherwise it skips.

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
{"git_sha":"...","model":"gpt2","config_hash":"...","ablate":{"episodic.use_gate":false}}
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
python scripts/eval_bench.py suite=episodic preset=baselines/span_short n_trials=200 seed=1337
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

## 9.5 Short-span decoding parity

```bash
python scripts/eval_model.py suite=episodic preset=baselines/span_short n=50 seed=1337 use_chat_template=true max_new_tokens=8
python scripts/eval_model.py suite=episodic preset=memory/hei_nw   n=50 seed=1337 use_chat_template=true max_new_tokens=8
```

# 10) Reporting (`scripts/report.py`)

- Aggregates all `metrics.json`/`metrics.csv` under `runs/**`.
- Produces a markdown table and simple charts (optional) comparing presets.
- Outputs to `reports/<date>/<suite>/summary.md`.
- Templates are loaded from `hippo_eval/reporting/templates`; the root
  `reports/` directory contains generated artifacts only.
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

- Baselines **must not be saturated**: target EM(raw) < 60% on H2-like episodic; we expect **+8–12pp EM(norm)** from memory on `episodic_multi` at n=200.
- **HEI‑NW:** +15pp EM over **core** on partial‑cue; +5pp over **RAG**; positive Δ after replay ≥ +3pp.
- **SGC‑RSS:** ≥10pp multi‑hop accuracy over **core**; contradiction rate ≤50% of **core**.
- **SMPD:** ≥90% path success on 5×5; ≥20% steps reduction with macros.
* **Relational:** duplicate rate reduced by ≥30% (ON vs OFF) with multi‑hop accuracy change within ±1pp.
* **Spatial:** edges/1k reduced by ≥25% with success rate within ±1pp and suboptimality unchanged (±0.02 absolute).
* **Delayed recall:** teach→test cycle with persisted stores yields ≥+20pp EM on `episodic@50` vs `baselines/core`.
* **Overhead:** gating overhead ≤10% runtime/100 queries.

# 12) CI & Codex scope

- CI runs `make lint` and a **dry_run** of the harness on 5 tasks per suite across all presets (ensures plumbing only).
- Preflight: dump resolved config (`config_snapshot.yaml`) at run start.
- Post-run assertions: fail if `retrieval.requests==0`, `refusal_rate>0.5`, or replay cycles < configured.
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
- **Step 9 aborts (`EM uplift < ...`):** ensure a replay run wrote `post_*` metrics,
  lower the threshold via `--min-uplift 0.05`, or collect more seeds and use
  `--uplift-mode ci`.

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
python scripts/eval_bench.py suite=episodic preset=baselines/span_short n=5 seed=1337 dry_run=true
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

## Consolidation Evaluation Protocol (Milestone 9.5)

**Goal:** Prove **systems consolidation**: after replay‑driven LoRA training, the model answers **without memory**.

**Protocol**
1. **Teach** — run with memory ON and writes enabled; save stores to `--store_dir`/`--session_id` (no grading).
2. **Pre** — evaluate the same suite with **memory OFF**; record EM/F1.
3. **Consolidate** — train LoRA via `scripts/replay_consolidate.py` from saved stores (optionally distill from a teacher-with-memory).
4. **Post** — evaluate with **memory OFF** again; report deltas.
5. **Ablations** — run `span_short`, `longctx`, `rag` to isolate gains.

**Metrics & assertions**
- Gate uplift using `test_consolidation.py`'s flags:
  `--uplift-mode [fixed|ci]`, `--min-uplift 0.05`, `--alpha 0.05`.
- Default pass condition: `post.EM - pre.EM ≥ min_uplift` on
  `episodic@50` (seed=1337).
- Memory runs show `retrieval.requests > 0`, `replay.samples > 0` in
  `metrics.json`.
- Refusal‑rate ≤ 0.5 on span suites.
- Sanity suite ≤ 1% degradation post‑LoRA.

**Decoding for span suites**
- Chat template: **ON**.
- System prompt: “Answer with the **exact shortest span** from the prompt. No explanations.”
- `max_new_tokens`: 8–16.

## Migration notes

- Evaluation, metrics, reporting, and synthetic task code now live in
  `hippo_eval/*`.
- Reporting templates moved to `hippo_eval/reporting/templates`, leaving the
  root `reports/` directory for generated outputs only.
- Importing modules through former `hippo_mem.*` paths triggers a
  `DeprecationWarning` but remains temporarily supported.
