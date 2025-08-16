# Executive Summary
Current codebase implements core scaffolding for episodic (HEI‑NW), relational (SGC‑RSS), and spatial (SMPD) memories. Modules compile and unit tests pass, but several design elements are only partially realised. Before entering data generation and LoRA training (Milestones M8–M10), missing schema routing, incomplete Hopfield testing, weak spatial adapter coverage, and absent maintenance jobs must be addressed. Top risks: unimplemented schema fast‑track (blocks SGC‑RSS), lack of ablation enforcement, and security/quality debt in long SQL and try/except blocks. A focused backlog of P0/P1 tasks (see TASKS.yaml) will close functional gaps and raise test coverage.

# Design Intent Recap (Authoritative Requirements)
- **HEI‑NW**: k‑WTA sparse keys, FAISS+PQ store, Hopfield completion, neuromodulated gate, CA2‑like replay, EpisodicAdapter with MQA/GQA and FlashAttention (DESIGN.md §5.2).
- **SGC‑RSS**: tuple extractor, KG with embeddings, schema fast‑track, dual‑path retrieval/fusion (DESIGN.md §5.4).
- **SMPD**: PlaceGraph + path integration, A*/Dijkstra planner, MacroLib, SpatialAdapter (DESIGN.md §5.6).
- **Shared**: Hydra configs with ablations, consolidation worker (50/30/20 mix), maintenance jobs, provenance/rollback, logging (DESIGN.md §6‑11).

# Traceability Matrices
See [TRACEABILITY.md](TRACEABILITY.md) for full mapping. Key verdicts:
- Schema fast‑track routing ❌
- Hopfield completion test coverage ⚠️
- Spatial adapter integration ⚠️

# Findings by Module
## Episodic (HEI‑NW)
- `DGKey` k‑WTA encoding and FAISS store present【F:hippo_mem/episodic/store.py†L28-L80】
- Hopfield completion implemented but untested【F:hippo_mem/episodic/store.py†L240-L255】
- Write gate matches S=α·surprise+β·novelty+γ·reward+δ·pin【F:hippo_mem/episodic/gating.py†L40-L73】
- ReplayQueue mixes score/recency/diversity【F:hippo_mem/episodic/replay.py†L29-L65】
Verdict: core mechanisms implemented; need test for completion and FlashAttention hook.

## Relational (SGC‑RSS)
- Tuple extraction and KG retrieval implemented【F:hippo_mem/relational/tuples.py†L1-L40】【F:hippo_mem/relational/kg.py†L14-L47】
- SchemaIndex exists but unused, blocking fast‑track routing【F:hippo_mem/relational/schema.py†L20-L54】
- Dual‑path adapter fuses KG and episodic features【F:hippo_mem/relational/adapter.py†L8-L36】
Verdict: missing schema routing integration (P0).

## Spatial/Procedural (SMPD)
- PlaceGraph supports path integration and planning via A*/Dijkstra【F:hippo_mem/spatial/map.py†L59-L118】【F:hippo_mem/spatial/map.py†L139-L180】
- MacroLib stores and ranks macros【F:hippo_mem/spatial/macros.py†L14-L53】
- SpatialAdapter defined but lightly exercised.
Verdict: functionality present but adapter coverage weak.

## Shared Infra
- Hydra configs for memory modules present【F:configs/memory/episodic.yaml†L1-L19】
- ReplayScheduler enforces 50/30/20 mix【F:hippo_mem/episodic/replay.py†L171-L194】
- ConsolidationWorker schedules adapter updates【F:hippo_mem/consolidation/worker.py†L1-L64】
- Maintenance jobs and rollback hooks missing; provenance fields exist but no rollback logic.

# Test Adequacy
- Coverage: episodic store 69%, relational KG 81%, spatial adapter 42%, eval script 56%【40d2c9†L21-L42】
- No tests for Hopfield completion, schema routing, ablation toggles (see TEST_GAPS.md).
- Mutation testing failed to run (missing mutmut config).

# Static & Maintainability
- flake8 reports >100 E501 line-length errors【735d85†L1-L99】
- Bandit flags try/except-pass and SQL injection risks in stores【a25898†L5-L70】
- Radon: all functions ≤9 complexity (grade A/B)【540b16†L1-L20】

# Runtime & CI Sanity
- Dry-run training logs episodic/KG/spatial counters【4f4b7c†L1-L8】
- Eval harness runs for core/longctx/rag presets with suite=episodic【a83bb8†L1-L2】【cc0005†L1-L2】【cb0597†L1-L2】
- CI workflow present but no badges; status unknown.

# Backlog
See [TASKS.yaml](TASKS.yaml). Priority P0 tasks before data generation:
- Implement schema fast‑track routing (F-REL-001)
- Schedule decay/pruning & rollback (F-SHARE-001)
- Add schema routing tests (T-REL-002)

# Appendices
- Commands: `pytest`, `pytest --cov`, `python -m flake8`, `python -m bandit -q -r .`, `python -m radon cc -s -a hippo_mem`, `python scripts/train_lora.py dry_run=true max_steps=5`, `python scripts/eval_bench.py suite=episodic preset=core|longctx|rag`
- Environment: Python 3.12, coverage HTML in `coverage/html`, static analysis in `static/`
