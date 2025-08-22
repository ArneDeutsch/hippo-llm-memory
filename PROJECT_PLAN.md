# Milestone 1 – Research consolidation & design blueprint

**Objective**: establish a shared theoretical foundation, map hippocampal mechanisms to LLM gaps, and produce design and evaluation documents.

**Work packages**

1. [x] **Literature synthesis & novelty check**: review research/hippocampal-memorystorage.md, large-language-models.md and synthesise the neuro-LLM mapping in experiment-synthesis.md to ensure a clear understanding of HEI‑NW, SGC‑RSS and SMPD hypotheses. Produce a short summary in research/summary.md and confirm originality in research/validation.md.
2. [x] **Design specification**: define the system architecture, data structures and algorithms for the episodic, relational and spatial modules, drawing from DESIGN.md (LLM core with episodic, relational and spatial adapters linked to FAISS/SQLite stores, knowledge graph and place graph). Update or finalise DESIGN.md as needed.
3. [x] **Evaluation plan**: outline datasets, baselines, run matrix, metrics and ablation toggles in EVAL_PLAN.md following the existing plan. Include success criteria (e.g., EM, F1, multi-hop accuracy, path success) and define baseline presets and memory variants.

- [x] **Gate**: DESIGN.md and EVAL_PLAN.md approved; research/summary.md and validation.md updated; CI passes linting and tests for documentation generation.

# Milestone 2 – Baseline infrastructure & smoke tests

**Objective**: set up repository scaffolding, CI, synthetic dataset generator, evaluation harness and baseline training script to verify the pipeline end-to-end.

**Work packages**

1. [x] **Repository & CI setup**: finalise directory structure, coding standards, Makefile and GitHub Actions; ensure flake8 / black and pytest run on each commit.
2. [x] **Dataset generator**: implement deterministic synthetic generators for episodic, semantic and spatial suites (W4 stories, schema-fit/mismatch facts, grid worlds) as described in the evaluation plan. The existing scripts/build_datasets.py already provides this functionality.
3. [x] **Evaluation harness**: build a lightweight evaluation script ( scripts/eval_bench.py ) capable of reading datasets, instantiating memory modules, computing metrics and writing metrics.json/csv and meta.json. Ensure ablation flags from EVAL_PLAN.md are supported.
4. [x] **Baseline training wrapper**: implement a small LoRA/QLoRA training script ( scripts/train_lora.py ) for dry-run smoke tests; integrate HuggingFace models and LoRA adapters with minimal memory modules.

- [x] **Gate**: repository builds and tests pass; scripts/build_datasets.py generates JSONL files; scripts/eval_bench.py runs in dry-run mode and produces metrics files; CI shows a green status.

# Milestone 3 – Episodic memory (HEI‑NW) prototype

**Objective**: implement the HEI‑NW episodic module with write gating, sparse encoding, modern-Hopfield completion and prioritized replay.

**Work packages**

1. [x] **Write gate**: implement the neuromodulated write gate combining surprise, novelty and reward signals (threshold τ) as described in the design . Provide a simple API to compute scores and decide on writes.
2. [x] **Episodic store**: implement a FAISS + SQLite store with k‑WTA sparse keys, product quantization and Hopfield‑style completion; support write, recall, delete, decay and prune operations.
3. [x] **Replay queue and scheduler**: implement a prioritized replay queue mixing salience, recency and diversity ; provide sampling functions for consolidation.
4. [x] **Episodic adapter**: implement a cross-attention adapter that attends over recalled traces and outputs fused embeddings; ensure compatibility with LoRA/QLoRA.
5. [x] **Tests**: write unit tests to verify one-shot recall, partial‑cue retrieval amid distractors and gating behaviour (similar to tests/test_episodic.py).

- [x] **Gate**: hippo_mem/episodic/ contains gating.py , store.py , replay.py and adapter.py ; tests pass; scripts/eval_bench.py can instantiate and query the episodic module.

# Milestone 4 – Relational semantic memory (SGC‑RSS) prototype

**Objective**: implement a relational store and schema routing for the SGC‑RSS hypothesis.

**Work packages**

1. [x] **Tuple extractor & schema index**: implement a heuristic extractor to identify (head, relation, tail, context) tuples from text and a schema index for routing; ensure extraction precision ≥0.9 as per tests.
2. [x] **Knowledge graph store**: implement a NetworkX + SQLite knowledge graph with node/edge embeddings and radius‑based retrieval. Include message‑passing updates and persistence.
3. [x] **Relational adapter**: implement a dual-path cross‑attention adapter merging subgraph embeddings with episodic traces; gating head merges outputs. Write tests to verify deterministic fusion.
4. [x] **Schema routing tests**: test multi-hop retrieval and schema-fit vs mismatch routing using simple examples (similar to tests/test_relational.py).

- [x] **Gate**: hippo_mem/relational/ modules and tests are complete; tuple extraction achieves ≥90 % precision; multi-hop retrieval returns correct subgraphs; adapter fusion is deterministic.

# Milestone 5 – Spatial and procedural memory (SMPD) prototype

**Objective**: implement the SMPD spatial map, macro library and adapter.

**Work packages**

1. [x] **PlaceGraph & planner**: implement a deterministic graph structure that observes sequences of contexts, builds nodes with optional path integration and supports A/Dijkstra planning.
2. [x] **Macro library**: implement MacroLib to store, update and suggest macros with success‑weighted ranking.
3. [x] **Spatial adapter**: implement a light cross‑attention adapter that interfaces with the PlaceGraph and MacroLib; ensure compatibility with LoRA/QLoRA.
4. [x] **Tests**: verify deterministic graph growth, path planning equivalence between A and Dijkstra, and that macro updates improve suggestion ranking .

- [x] **Gate**: hippo_mem/spatial/ modules exist; tests for deterministic map growth, path planning and macro suggestion pass; scripts/eval_bench.py can instantiate the spatial module.

# Milestone 6 – Consolidation & replay framework

**Objective**: implement offline consolidation inspired by hippocampal replay and add maintenance tasks.

**Work packages**

1. [x] **Priority replay scheduler**: implement a scheduler that samples traces from the episodic store for consolidation based on salience, recency and diversity ; support configurable mixing ratios (e.g., 50% episodic, 30% semantic, 20% fresh tasks).
2. [x] **Consolidation worker**: implement a background process that fetches replay batches, fine‑tunes adapters (keeping the base model frozen), and interleaves hard negatives; integrate this into the training script.
3. [x] **Maintenance jobs**: implement periodic decay and pruning for the episodic store and knowledge graph and path-integration decay for the spatial map (the underlying stores already provide decay/prune methods; they need to be scheduled and logged).
4. [x] **Logging & monitoring**: collect statistics on writes, recalls and hits for each module and record replay cycles and consolidation progress.

- [x] **Gate**: scheduler, consolidation worker and maintenance jobs run during a dry-run training session; logs show replay batches sampled and stores decayed/pruned; integration tests verify no crashes.

# Milestone 7 – Integration with LLM & ablation-ready training

**Objective**: wire memory modules into a small open LLM using LoRA/QLoRA and ensure ablation flags enable controlled experiments.

**Work packages**

1. [x] **Adapter hookup**: modify scripts/train_lora.py to load memory modules (episodic, relational, spatial) based on configuration; ensure cross-attention adapters can be enabled individually or jointly.
2. [x] **Replay scheduling integration**: incorporate the replay scheduler and consolidation worker into the training loop; schedule interleaved batches of episodic, semantic and fresh data; implement dataloaders for synthetic datasets.
3. [x] **Hydra configuration & ablations**: provide YAML configs for base model, memory modules and ablation toggles (e.g., disable Hopfield completion, gating, schema routing, macros) according to EVAL_PLAN.md.
4. [x] **Dry-run training**: run a small fine-tuning session (few steps) to verify that the model parameters update, adapters receive gradients, replay batches are interleaved and ablation flags take effect.

- [x] **Gate**: scripts/train_lora.py executes end-to-end with memory modules and ablations; logs show replay scheduling; CI tests confirm ablation toggles are respected.

# Milestone 7b – End-to-end wiring & dataset integration

**Objective**: attach memory adapters into the Transformer forward path, switch trainer to JSONL suites, verify LoRA attachment, and mix replay.

**Work packages**

1. [x] **Adapter hookup**: insert Episodic/Relational/Spatial adapters after configurable block N; enable/disable via Hydra flags.
   - Evidence: hippo_mem/adapters/patch.py, tests/test_adapter_wiring.py
2. [x] **LoRA attachment checks**: set architecture-specific `target_modules`; log trainable param count; assert >0.
   - Evidence: hippo_mem/adapters/lora.py, tests/test_lora_targets.py
3. [x] **JSONL data loader**: replace IMDB default with loaders for `data/episodic_*`, `data/semantic_*`, `data/spatial_*`; add train/val split selection in config.
   - Evidence: scripts/jsonl_dataset.py, tests/test_data_loader.py
4. [x] **Replay mixing**: integrate `ReplayScheduler` batches by ratio; ensure clean thread lifecycle.
   - Evidence: scripts/replay_dataset.py, tests/test_replay_dataset.py, tests/test_replay_scheduler.py
5. [x] **Tests**: add tests ensuring non-zero trainables and that adapter hooks run at least once per batch.

- [x] **Gate**: PASS (2025-08-20) via `scripts/smoke_7b.sh`.

# Milestone 8a – Runtime retrieval & write-gate plumbing

**Objective:** Feed *real memory traces* from the episodic/relational/spatial stores into adapters during forward; apply write-gating and persist new episodes; standardize adapter I/O.

**Work packages**
1. [x] **TraceSpec:** define a unified schema and shapes for traces passed to adapters (tokens or pooled vectors + optional metadata).
2. [x] **Retrieval hooks:** given batch text, compute query encodings and retrieve top-K: (a) episodic store, (b) semantic KG neighborhoods/paths, (c) spatial local map/plan. Bound K and length.
3. [x] **Projection & packing:** project retrieved features to `d_model`; pack to `memory_tokens` `[B, M, d_model]` (+ optional masks).
4. [x] **Adapter API:** update adapters to accept `memory_tokens` (+ masks) via the patcher; no-op if empty.
5. [x] **Write-gate & store update:** compute gate S; when passing threshold, persist new episode/tuple/place; respect async worker lifecycles.
6. [x] **Config & toggles:** Hydra flags to enable/disable each memory, set K, max tokens, and gating threshold τ.
7. [x] **Telemetry:** log retrieval hit-rates, avg M, % writes, and latency per step.


- [x] **Gate:** write-gate is executed on real batches; acceptance rate & store growth logged; async writer commits.
  - evidence: sample log with `gate_accept_rate>0`
  - evidence: `store_size` increasing
  - evidence: snippet showing adapter block index and retrieval latency

# Milestone 8 – Baseline datasets & evaluation runs (UPDATED)

**Objective.** Build deterministic baseline datasets and execute *baseline* evaluation runs (no learned memory) across all suites to establish reproducible reference metrics. Capture seeds, sizes, configs, environment and git SHA so later memory results are comparable.

**Scope.** Suites = `episodic`, `semantic` (relational), `spatial`. Presets = `baselines/core`, `baselines/rag`, `baselines/longctx`. Sizes = `50`, `200`, `1000` items. Seeds = `1337`, `2025`, `4242`.

## Work packages

1. [ ] **Dataset generation & registry**
   - Use `scripts/build_datasets.py` to generate JSONL datasets for each suite × size × seed.
   - Write files to `data/<suite>/<size>_<seed>.jsonl`.
   - Compute and store SHA256 checksums in `data/<suite>/checksums.json`.
   - Emit a lightweight `data/<suite>/dataset_card.json` with fields: `suite`, `sizes`, `seeds`, `generator_version`, `sha256`, `created_utc`, and the exact CLI used.

2. [ ] **Run‑matrix driver**
   - Add a small driver (CLI or Make target) to iterate over *all* presets × suites × sizes × seeds and invoke `scripts/eval_bench.py` with the correct Hydra overrides.
   - Layout: `runs/YYYYMMDD/<preset>/<suite>/<size>_<seed>/` containing `metrics.json`, `metrics.csv`, and `meta.json` (with `git_sha`, `config_hash`, `ablate`, `seed`, `python`, `platform`).
   - Provide `Makefile` targets:
     - `make eval-baselines DATE=20250822` – full matrix.
     - `make eval-baselines-smoke` – minimal (one suite × one size × one seed) for CI.

3. [ ] **Environment capture & determinism**
   - Extend `scripts/eval_bench.py` to record: Python version, `pip freeze` hash, OS, CPU model, and (if available) CUDA/driver versions into `meta.json`.
   - Assert deterministic generation via checksum validation before each run; fail fast if mismatched.

4. [ ] **Aggregation & reports**
   - Extend `scripts/report.py` to:
     - Aggregate metrics per suite across presets, sizes, and seeds (mean ± std) and write Markdown to `reports/YYYYMMDD/<suite>/summary.md`.
     - Include optional plots when matplotlib is present (saved as PNG).
     - Tolerate missing retrieval/gate fields for baselines while still producing tables.

5. [ ] **CI & smoke wiring**
   - Add `smoke_8.sh` that generates size=50 datasets for all suites, runs the `baselines/core` preset for seed=1337, and runs `scripts/report.py --date YYYYMMDD`.
   - Add/extend `tests/test_report.py` to assert:
       - discovery of latest date dir,
       - presence and schema of aggregated tables,
       - report files exist for each suite.

6. [ ] **Documentation**
   - Update `EVAL_PLAN.md` section “Baselines” to pin the **exact** presets, run‑matrix (suites, sizes, seeds), directory layout, and required artifacts.
   - Note in `DESIGN.md` that memory adapters are not involved in Milestone 8; any retrieval/gating logs MUST be no‑ops for baselines.

## Definition of Done (Gate)

- **Datasets:** For each suite × size × seed, a JSONL exists with a recorded SHA256 in `checksums.json`; `dataset_card.json` present.
- **Runs:** For every preset × suite × size × seed, `metrics.json`, `metrics.csv`, and `meta.json` are present under `runs/YYYYMMDD/...` with non‑empty contents. `meta.json` must include `git_sha` and `config_hash`.
- **Aggregation:** `reports/YYYYMMDD/<suite>/summary.md` exists for all suites and contains per‑preset tables (means across seeds) for each size.
- **Reproducibility:** Re‑running the matrix on the same commit and machine yields identical checksums and *identical metrics* for the mock baselines.
- **CI:** `smoke_8.sh` completes locally in < 3 minutes on CPU and passes the associated tests.

## Notes

- **Compute:** Baselines use `model: mock` and run on CPU; no GPU is required for Milestone 8.
- **Forward‑compatibility:** File layout and metadata mirror what Milestone 9 (memory) will produce so reports can compare across milestones seamlessly.

# Milestone 9 – Memory‑augmented training, evaluation & ablations

**Objective**: train models with each memory module, evaluate them, perform ablations and compare against baselines.

+**Work packages**
0. [ ] **Establish real (non‑oracle) baselines** with `scripts/eval_model.py` (no memory) over the same matrix (sizes 50/200/1000 × seeds 1337/2025/4242).
1. [ ] HEI‑NW evaluation: fine‑tune with episodic memory; log gate accept %, store growth, retrieval hit@k, replay cycles; record compute (tokens, time_ms_per_100, rss_mb).
2. [ ] SGC‑RSS evaluation: train with relational memory; log schema fast‑track rate and contradiction filter stats; record compute columns.
3. [ ] SMPD evaluation: train with spatial memory; log local‑map size and steps‑to‑solve; record compute columns.
4. [ ] Combined model: enable all memories; run at n=200 with 3 seeds; collect trade‑offs.
5. [ ] Ablations: for each memory, toggle its key knobs (gate on/off, completion on/off, fast‑track on/off, macro distillation on/off) at n=200 × 3 seeds.
6. [ ] Reports: create `reports/<date>/index.md` aggregating tables and plots across baselines and memory variants.

**Gate**
- [ ] For each memory variant and baselines (real), runs exist at sizes 50 and 200 with seeds 1337/2025/4242 and include compute & telemetry fields.
- [ ] Improvements over real baselines are demonstrated on n=200 (episodic EM +10 points suggested; others suite‑specific).
- [ ] Ablation effects are clear (directional, non‑noisy).
- [ ] `reports/<date>/index.md` is present with tables/plots and links to per‑suite summaries.

# Milestone 10 – Research paper & public release

**Objective**: synthesise research, implementation and evaluation into a publishable manuscript and release the software and data.

**Work packages**

1. [ ] **Manuscript drafting**: write a paper (in Markdown or LaTeX) describing hippocampal mechanisms, mapping to LLMs, implementation of HEI‑NW, SGC‑RSS and SMPD (referencing algorithms such as k-WTA encoding, modern-Hopfield completion and CA2-style replay), experimental setup (datasets, baselines, evaluation protocol), results (including tables/plots from report.py) and discussion of limitations and future work.
2. [ ] **Internal review & revisions**: circulate the draft within the team for feedback; address comments and finalise.
3. [ ] **Public release**: tag a release in GitHub; provide a clean README.md with installation and reproduction instructions; upload datasets and configuration files; ensure licensing is appropriate.
4. [ ] **Submit preprint**: publish the paper on a preprint server (e.g., arXiv) and share with the community.

- [ ] **Gate**: final manuscript ready for submission; repository tagged and includes release notes; reproduction guide verified; datasets and reports publicly accessible.
