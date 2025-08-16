# Executive Summary

Current codebase delivers functional prototypes for HEI-NW, SGC-RSS and SMPD. Core mechanisms (write gate, Hopfield completion, replay scheduler, schema index, PlaceGraph) are present and tested. However, sparse k‑WTA encoding and retrieval utilities are missing or unused. Test coverage for episodic, spatial and training scripts meets targets, but retrieval modules and some scripts fall below thresholds. Extensive style violations and absent mutation configuration raise maintainability risks. Repository is **not yet ready for M8–M10** without addressing k‑WTA integration, retrieval coverage, and schema-routing tests.

Top risks:
1. Missing k‑WTA implementation jeopardises episodic sparsity and design fidelity.
2. Retrieval layer and schema fast-track lack tests, reducing confidence in semantic routing.
3. Lint failures and unused modules may hinder future development.

One-page plan: implement k‑WTA and associated tests (P0), add schema routing and retrieval tests (P1), harden training script and clean lint (P2+). See `TASKS.yaml` for backlog.

# Design Intent Recap (Authoritative Requirements)
- **HEI‑NW**: k‑WTA sparse keys, FAISS+PQ store, modern-Hopfield completion, write gate `S=α·surprise+β·novelty+γ·reward+δ·pin` with τ, CA2-like replay prioritising salience/recency/diversity【F:DESIGN.md†L61-L88】
- **SGC‑RSS**: tuple extractor, KG store with embeddings, schema fast-track routing, dual-path retrieval with gated fusion【F:DESIGN.md†L84-L88】
- **SMPD**: PlaceGraph with optional path integration, A*/Dijkstra planner, MacroLib for replay-to-policy distillation【F:DESIGN.md†L90-L94】
- **Shared**: Hydra configs with ablations, consolidation worker with 50/30/20 mix, nightly decay/pruning, provenance/rollback logging【F:DESIGN.md†L107-L132】

# Traceability Matrices
See [TRACEABILITY.md](TRACEABILITY.md) for full requirement→code→test mapping.

# Findings by Module
## Episodic (HEI‑NW)
- **Write gate** implemented with surprise/novelty/reward/pin and τ threshold【F:hippo_mem/episodic/gating.py†L52-L106】; tests verify pin override【F:tests/test_episodic.py†L67-L85】.
- **Hopfield completion** present via `complete` method【F:hippo_mem/episodic/store.py†L265-L277】.
- **Replay scheduler** mixes salience, recency, diversity and checks gradient overlap【F:hippo_mem/episodic/replay.py†L162-L239】.
- **Gap**: k‑WTA sparse key generation (`DGKey`) not used; store operates on dense vectors.

## Relational (SGC‑RSS)
- Heuristic tuple extractor exists【F:hippo_mem/relational/tuples.py†L1-L40】 and KG with schema fast-track routing【F:hippo_mem/relational/schema.py†L25-L57】.
- Relational adapter fuses KG and episodic features with confidence gating【F:hippo_mem/relational/adapter.py†L6-L30】.
- **Gap**: no test for schema fast-track threshold; schema scoring only checks relation equality.

## Spatial/Procedural (SMPD)
- PlaceGraph supports deterministic observation, optional path integration, A*/Dijkstra planner【F:hippo_mem/spatial/map.py†L125-L205】.
- MacroLib stores trajectories and suggests macros【F:hippo_mem/spatial/macros.py†L13-L52】.
- SpatialAdapter provides cross-attention hook【F:hippo_mem/spatial/adapter.py†L1-L68】.

## Shared Infrastructure
- Consolidation worker schedules 50/30/20 replay and maintenance jobs【F:hippo_mem/consolidation/worker.py†L162-L176】.
- Hydra configs and evaluation harness present (configs/ and scripts/eval_bench.py). Dry-run training and evaluation execute with logging【3f7117†L1-L6】【7e6845†L1-L4】.
- **Gap**: retrieval utilities (`embed.py`, `faiss_index.py`) unused; train script fails when adapters lack LoRA params【d5cbc8†L1-L13】.

# Test Adequacy
- Coverage: overall 85%; episodic, spatial modules ≥90%, relational KG 78%, scripts/train_lora.py 93%, eval_bench.py 74%, retrieval utilities 0%【a7c4d3†L18-L40】.
- Missing behaviors: schema fast-track threshold, k‑WTA, retrieval FAISS operations, train ablation hooks (see `TEST_GAPS.md`).
- Mutation testing: `mutmut` failed due to missing configuration【786fe2†L1-L27】.

# Static & Maintainability
- `flake8` reports widespread E501/E402 violations across modules【c7ab5e†L1-L90】.
- Bandit highlights numerous low-severity `assert_used` issues in tests and a few hardcoded tokens【239e45†L1-L110】.
- Radon shows average complexity A; no hotspots but retrieval modules unused (see `static/radon_cc.txt`).

# Runtime & CI Sanity
- Dry-run training prints maintenance and replay logs【3f7117†L1-L6】 but fails when adapters lack LoRA params in full run【d5cbc8†L1-L13】.
- Evaluation harness runs for core/longctx/rag presets without errors【7e6845†L1-L4】【bee0c1†L1-L4】【03999b†L1-L5】.
- CI workflow executes lint and tests (`.github/workflows/ci.yml`).

# Backlog
Priority tasks before M8–M10:
- Implement k‑WTA and integrate with store (F‑EPI‑001).
- Add schema fast-track and retrieval tests (T‑REL‑001, T‑RET‑002).
- Harden consolidation worker and training script (R‑TRN‑003).
- Address lint violations and remove dead retrieval modules (R‑LNT‑004).
See `TASKS.yaml` for details.

# Appendices
- Commands run: pytest with coverage, flake8, bandit, radon, mutmut attempt, dataset build, train/eval dry runs.
- Environment: Python 3.12, packages per `codex-env/requirements.txt` plus flake8/bandit/radon/mutmut.
- Coverage artifacts: `coverage/html/`, `coverage/coverage.json`.
