# Executive Summary

The current codebase implements basic versions of the HEI‑NW, SGC‑RSS and SMPD modules but significant gaps remain before Milestones M8–M10 can begin. Core mechanisms—k‑WTA keys, Hopfield completion, schema routing, path planning—exist, yet several integrations (FlashAttention hooks, gradient‑overlap aware replay, logging of maintenance jobs) are partial. Tests cover many behaviors (one‑shot recall, schema thresholding, planner equivalence) but overall coverage is 75% and key modules such as the knowledge graph and retrieval layer fall below targets. Static analysis highlights numerous long lines and high‑complexity functions, notably `ConsolidationWorker.run`. A focused backlog is required to harden the system, expand tests, and refactor risky code before large‑scale data generation and LoRA training.

# Design Intent Recap (Authoritative Requirements)
- **HEI‑NW**: k‑WTA sparse keys, FAISS+PQ store, modern‑Hopfield completion, neuromodulated write gate \(S=α·surprise+β·novelty+γ·reward+δ·pin\), CA2‑like prioritized replay, EpisodicAdapter cross‑attention with MQA/GQA/FlashAttn hooks.
- **SGC‑RSS**: tuple extractor, knowledge‑graph store with embeddings, schema index with fast‑track routing, dual‑path retrieval and gated fusion.
- **SMPD**: PlaceGraph with optional path‑integration, planner (A*/Dijkstra), MacroLib with behavior‑cloned macros, SpatialAdapter tool interface.
- **Shared**: Hydra configs with ablation toggles, consolidation worker mixing 50/30/20 episodic/semantic/fresh batches, nightly decay/pruning jobs, provenance/rollback logging, reproducibility hooks.

# Traceability Matrices
See [TRACEABILITY.md](TRACEABILITY.md) for detailed requirement→code→test mapping.

# Findings by Module
## Episodic (HEI‑NW)
- **k‑WTA & sparse keys** implemented via `sparse_encode` producing index/value pairs【F:hippo_mem/episodic/store.py†L149-L161】.
- **Neuromodulated write gate** computes `α·surprise + β·novelty + γ·reward + δ` and thresholds at `τ`【F:hippo_mem/episodic/gating.py†L52-L106】.
- **FAISS+PQ store** with Hopfield completion provided; maintenance/rollback present but untested【F:hippo_mem/episodic/store.py†L263-L278】.
- **Prioritized replay** mixes salience, recency and diversity, with optional grad‑overlap proxy【F:hippo_mem/episodic/replay.py†L29-L120】.
- **EpisodicAdapter** supports MQA/GQA but lacks FlashAttention integration; tests cover partial‑cue recall and Hopfield completion.
- **Verdict**: ⚠️ Partial—missing FlashAttention hooks, incomplete replay gradient handling, low coverage (70%).

## Relational (SGC‑RSS)
- **Tuple extractor** achieves ≥0.9 precision in tests【F:tests/test_relational.py†L7-L20】.
- **KnowledgeGraph** persists nodes/edges with embeddings and schema fast‑track routing【F:hippo_mem/relational/schema.py†L22-L57】.
- **RelationalAdapter** performs dual‑path gating but retrieval embeddings have limited tests; KG maintenance/rollback untested.
- **Verdict**: ⚠️ Partial—schema routing present but no tests for GNN updates or rollback; coverage 62%.

## Spatial/Procedural (SMPD)
- **PlaceGraph** with optional path integration and deterministic planning via A* or Dijkstra【F:hippo_mem/spatial/map.py†L64-L140】【F:hippo_mem/spatial/map.py†L200-L238】.
- **MacroLib** stores trajectories and updates success statistics【F:hippo_mem/spatial/macros.py†L13-L52】.
- **SpatialAdapter** fuses hidden states with plan embeddings; integration tested【F:tests/test_spatial.py†L49-L67】.
- **Verdict**: ✅ Conforms—core features and tests present, though rollback/decay behaviors lack tests.

## Shared Infrastructure
- **Hydra configs** expose adapter toggles (`episodic.enabled`, `relational`, `spatial.enabled`) and replay mix defaults 0.5/0.3/0.2【F:scripts/train_lora.py†L45-L88】【F:scripts/train_lora.py†L113-L142】.
- **Consolidation worker** interleaves replay batches but raises runtime errors when gradients are missing【3225c1†L1-L15】.
- **Maintenance jobs** via background threads for stores/maps exist but lack logging tests.
- **Provenance/rollback** implemented in stores (`_log_event`, `rollback`) yet not exercised in tests.
- **CI**: single workflow runs lint and tests; linter currently fails due to long lines【3d36b4†L1-L94】.
- **Verdict**: ⚠️ Partial—configs cover core toggles but reproducibility metadata and FlashAttention flags absent.

# Test Adequacy
- Overall coverage 75%; targets unmet for episodic store (70%), knowledge graph (62%), spatial map (69%), and scripts below 85% threshold for core modules【c72013†L19-L45】.
- Unit tests exercise one‑shot recall, partial‑cue retrieval, schema routing, planner equivalence and macro improvement, but lack coverage for maintenance/rollback, replay batch mixing, and retrieval adapters.
- Mutation testing via `mutmut` failed to run due to configuration issues (FileNotFoundError)【1884b1†L1-L24】.
- See [TEST_GAPS.md](TEST_GAPS.md) for detailed specs of missing tests and property‑based suggestions (e.g., k‑WTA invariants, replay similarity constraints, schema fast‑track thresholds, planner optimality).

# Static & Maintainability
- `flake8` reports extensive E501 line-length violations and import-order issues across modules and tests【3d36b4†L1-L94】.
- Bandit flags potential SQL injection in manual query construction and subprocess usage without revision pinning【2baff3†L15-L49】【2baff3†L71-L92】.
- `radon` highlights high cyclomatic complexity in `ConsolidationWorker.run (C=16)` and `EpisodicStore.prune` (B)【8b9bd6†L1-L19】.
- Proposed refactors in [REFACTOR.md](REFACTOR.md) target these hotspots and line-length clean‑ups.

# Runtime & CI Sanity
- Dataset generator produced small synthetic sets for all suites【e42c95†L1-L2】.
- Training dry‑run with all modules enabled triggered background worker but raised a grad‑required error, indicating missing loss wiring【3225c1†L1-L15】.
- Training with all memory modules disabled executed cleanly with maintenance logs only【fd6e55†L1-L8】.
- Evaluation harness produced metrics for `core`, `longctx`, and `rag` presets without errors.

# Backlog
Critical P0/P1 tasks before dataset generation and LoRA training: FlashAttention integration, replay gradient handling, KG and store maintenance tests, linter compliance, and refactor of high‑complexity functions. See [TASKS.yaml](TASKS.yaml) for full actionable backlog.

# Appendices
- Commands executed and tool versions: see `scripts/checks/run_all_checks.sh`.
- Coverage HTML: `coverage/html/index.html`.
- Static analysis artifacts: `static/radon_cc.txt`, `static/radon_mi.txt`.
