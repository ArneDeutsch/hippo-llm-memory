# Run Review — 2025-09-06 16:18 UTC

Run directory: `runs/run20250906`  (SEED=1337, N=50)

## TL;DR
- All three suites show **0.0 EM/F1** at `n=50`, and the LLM outputs are mostly off-format (high `overlong` rates).
- Stores are **present**, but two are clearly broken:
  - **SGC‑RSS** semantic store has only **2 nodes / 1 edges** and **zero embeddings**, i.e., effectively empty.
  - **HEI‑NW** episodic store has **100 traces**, but **51 / 100 keys are zero‑vectors**; retrieval hit‑rate is **0.10** and the model doesn’t use the recalls (`context_match_rate=0`).
  - **SMPD** spatial store counts: **51 nodes / 100 edges**; retrieval queries run (`k=16`) but hits are **0.0** → the map is not helpful/compatible with queries.
- Gating is either **accepting everything** (HEI‑NW) or **skipping everything** (SGC‑RSS), which defeats its diagnostic value.
- The evaluation **measures generation accuracy only**. Without robust context fusion or constrained decoding, the LLM ignores memory and emits defaults (e.g., “Paris”, “New York City”), so we can’t learn much about the memory algorithms yet.

## What the data shows

### Aggregated telemetry

| algo    | suite              |   pre_em |   pre_f1 |   memory_hit_rate |   context_match_rate |   latency_ms_delta |   overlong_rate |   n_rows |   gate.attempts.episodic |   gate.attempts.relational |   gate.attempts.spatial |   gate.accepted.episodic |   gate.accepted.relational |   gate.accepted.spatial |   ret.k.episodic |   ret.k.relational |   ret.k.spatial |   ret.hit_rate.episodic |   ret.hit_rate.relational |   ret.hit_rate.spatial |
|:--------|:-------------------|---------:|---------:|------------------:|---------------------:|-------------------:|----------------:|---------:|-------------------------:|---------------------------:|------------------------:|-------------------------:|---------------------------:|------------------------:|-----------------:|-------------------:|----------------:|------------------------:|--------------------------:|-----------------------:|
| HEI‑NW  | episodic_cross_mem |        0 |        0 |               0.1 |                    0 |           0.490238 |            0.36 |       50 |                       50 |                          0 |                       0 |                       50 |                          0 |                       0 |                1 |                  0 |               0 |                     0.1 |                         0 |                      0 |
| SGC‑RSS | semantic_mem       |        0 |        0 |               0   |                    0 |           0.158288 |            0.98 |       50 |                        0 |                         50 |                       0 |                        0 |                          0 |                       0 |                0 |                  1 |               0 |                     0   |                         0 |                      0 |
| SMPD    | spatial_multi      |        0 |        0 |               0   |                    0 |           0.163013 |            0    |       50 |                        0 |                          0 |                      50 |                        0 |                          0 |                       2 |                0 |                  0 |              16 |                     0   |                         0 |                      0 |

### Store health snapshots

**Episodic (HEI‑NW)**
- Records: **100**, non‑zero element ratio across keys: **0.49**
- Zero‑norm keys: **51**, key ‖min,max‖ = **0.000 .. 1.000**

**Relational (SGC‑RSS)**
- Nodes: **2** (with embeddings: **0**)
- Edges: **1** (with embeddings: **0**)
- ⚠️ All embeddings are **missing** → retrieval ranks degenerate; store size is orders of magnitude below expectation (should be ≥ 2 facts per context).

**Spatial (SMPD)**
- Nodes: **51**, Edges: **100**, Meta records: **1**
- Retrieval: **k=16**, **hit_rate=0.00**, gate accept **2/50**

### Example failure modes (from `audit_sample.jsonl`)
- **Episodic** questions like “Where did Carol go?” expect a short place (e.g., “Library”), but predictions are generic (“Paris”) and **don’t reflect retrieved keys**.
- **Semantic** questions like “In which city did Carol buy the apple?” require chaining *buy* → *store* → *city*. The KG is practically empty, so answers default to “New York City”.
- **Spatial** outputs contain invalid characters (e.g., **C**) and are far longer than the gold paths (format policy not enforced).

## Diagnosis — Implementation vs. Evaluation

### 1) HEI‑NW (episodic)
- **Symptoms:** 10% retrieval hit-rate, but **zero context match** and zero accuracy. Half the stored keys are **all‑zeros**.
- **Likely causes:**
  1. **Key construction bug:** DGKey → dense export may emit zero vectors for half the samples. Check k‑WTA path, `k>0`, and `to_dense` used during persistence.
  2. **Prompt/context packing:** recalls aren’t actually injected or are placed where the model ignores them; audit lacks the retrieved snippets.
  3. **Index training/usage:** FAISS config might be untrained or wrong metric; however hit‑rate>0 suggests indexing minimally works.
- **Impact:** Even when retrieval finds a relevant trace, the LLM ignores it; gating accepts **100%** of attempts → gate is not informative.

### 2) SGC‑RSS (relational)
- **Symptoms:** Store has **≈0 content** and **no embeddings**; gate attempts=50 but **accepted=0**, retrieval runs with `k=1` yet **hit_rate=0**.
- **Likely causes:**
  1. **Tuple extraction thresholds/schemas:** `SchemaIndex` likely has no default schemas matching “bought / is in”, so nothing is promoted to the KG.
  2. **Embeddings never set:** `kg.upsert()` accepts `*_embedding` parameters but does not compute them when absent → JSONL shows `embedding: null` for nodes/edges.
  3. **Gate tuned to always skip:** threshold too high relative to the dumb tuple confidence → everything routed away.
- **Impact:** The semantic memory path is effectively a no‑op; the model falls back to prior/default answers.

### 3) SMPD (spatial)
- **Symptoms:** Graph exists (151 nodes) but **hit_rate=0** and **gate accepts 2/50**; outputs violate `UDLR` policy and are overlong.
- **Likely causes:**
  1. **Query → memory mismatch:** the plan retrieval doesn’t map the test goal to stored transitions (coordinate frames/context keys misaligned).
  2. **Format enforcement missing:** decoder isn’t constrained to `UDLR` and max length; normalization can’t rescue wildly long strings.
  3. **Adapter fusion ineffective:** even on recalls, the LLM doesn’t condition on the retrieved map/plan.

## Are we measuring the right things?
Right now, **generation-only EM/F1** yields zeros across the board, so we learn little about the memory modules. We should add **mechanistic KPIs** and **oracle ceilings**:

- **Mechanistic KPIs** (already partly logged):
  - *Store completeness* (per suite): expected vs actual objects/edges/traces; **fail hard** when below thresholds.
  - *Embedding integrity*: non‑zero norms; coverage %.
  - *Retrieval quality*: hits@k vs oracle keys; latency.
  - *Gate calibration*: acceptance vs attempts; AUC via synthetic labels.
- **Oracle ceilings:** run a non‑generative answerer that reads the retrieved context and computes the gold answer (string match for episodic/semantic; shortest‑path/A* for spatial). This shows whether **the memory contains enough signal** even if the LLM ignores it.
- **Constrained decoding:** enforce the **short‑answer policy** (and `UDLR` for spatial) to isolate *reasoning failures* from *formatting failures*.

## Concrete fixes (ranked, with quick checks)

### P0 — Make stores non‑degenerate and visible in prompts
1. **Relational embeddings ON by default**
   - In `kg.upsert()`, if `head_embedding/tail_embedding/edge_embedding` are `None`, compute with `hippo_mem.retrieval.embed.embed_text(name, dim=16)` and persist.
   - Add a unit test: insert two facts (`bought`, `is in`) and assert node/edge embeddings are non‑empty and `retrieve()` returns a 2‑hop subgraph containing both.
2. **Schema defaults for the synthetic data**
   - Add schemas: `bought`, `is`, `in`, `at`, `located_in` with `threshold≈0.5–0.6` (match `tuples.score_confidence`).
   - Gate sanity test: at `n=50`, require `nodes≥100`, `edges≥100` else **abort run**.
3. **Episodic key export fix**
   - Ensure `EpisodicStore.save()` writes **dense keys** (post `to_dense`) for *all* traces; add assertion: `‖key‖>0` for ≥90% of traces.
   - If k‑WTA may return empty keys (k<=0), clamp config to `k>0` and log violations.
4. **Prompt packing audit**
   - Extend `audit_sample.jsonl` to include the **actual retrieved snippets/tokens** injected into the prompt **and their positions**; fail run if missing.

### P1 — Constrained decoding & answerability checks
5. **Short‑answer enforcer**
   - Post‑process generations with regex to the allowed character set and length; log `normalized_pred` vs raw; drop to empty on violation to make error modes explicit.
6. **Spatial output validator**
   - Enforce `^[UDLR]{1,64}$`; compute **oracle path** via BFS/A* on the constructed grid and compare → get *success rate* even if tokens differ slightly.
7. **Oracle readers**
   - Episodic/Semantic: if the retrieved context contains the gold span, the oracle should achieve EM=1; log this as *upper bound*.

### P2 — Gate calibration & ablations
8. **Gating regression tests**
   - Construct synthetic mini‑batches with known positives/negatives; require acceptance≈50–80% in teach; ensure `use_gate=false` ablation increases writes and `attempts` count.
9. **Retrieval @k sweeps**
   - Sweep `k∈{1,4,8,16}`; require monotonic non‑decreasing hit@k; current **0.0** indicates index or query vector bugs.

## What can we already conclude about usefulness?
- With the present run, **no** algorithm demonstrates utility on end metrics because **stores are empty/low‑quality** and the **model ignores context**.
- After P0 fixes, we expect:
  - **HEI‑NW**: hit@k ≥ 0.5 and visible lift in EM when recalls are injected and short‑answer policy enforced.
  - **SGC‑RSS**: non‑zero embeddings and KG density sufficient to answer 1‑hop/2‑hop questions; oracle EM near 1.0; model EM improving with constrained decoding.
  - **SMPD**: non‑zero hit@k and near‑100% oracle shortest‑path success; model success depends on adapter fusion but should exceed 0 with formatting constraints.

## Minimal additional checks to add (and a few to prune)
**Add:**
- `store.size` lower bounds per suite; **abort** on violation.
- `embedding.nonzero_ratio` ≥ 0.9 (episodic/relational).
- `retrieval.requests>0` and `hit@k>0` for memory presets.
- `audit.has_injected_context==true` on samples.
- Spatial: `valid_action_rate` and `oracle_path_success`.
**Remove/De‑prioritize:**
- Duplicative `pre_em_raw` vs `pre_em_norm` logging at smoke scale; keep one in smoke, both in full runs.
- Per‑row latency when running CPU‑only smoke; keep aggregate only.

---
_Generated from artifacts under `/mnt/data/hippo-llm-memory-main/hippo-llm-memory-main/runs/run20250906`. This report is self‑contained and suitable for code review._