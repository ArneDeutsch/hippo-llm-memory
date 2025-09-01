# Run Audit & Pipeline Review — 20250901_50_1337_2025

_Generated: 2025-09-01 08:16_

## Scope
- Review artifacts under `runs/` and `reports/` for SIZES=50 and SEEDS=(1337, 2025).
- Check what each algorithm actually **memorizes** in `runs/.../stores/` and whether telemetry/metrics credibly evaluate uplift.
- Identify pipeline flaws and propose concrete fixes and validation checks.

---

## TL;DR (Executive Summary)
- **Baseline scoring is suppressed** in the *pre* phase: many `metrics.csv` rows have `NaN` for `em_raw/em_norm/f1`, and `metrics.json` carries zeroed `pre_*` aggregates. This makes baselines appear as **EM=0.0** across suites, masking reality and invalidating uplift claims.
- **Stores reveal stubs/dummy data**, not real traces from tasks:
  - Episodic store contains only a handful of entries with `provenance: "dummy"` and uniform keys.
  - Relational KG has only the toy triple `a rel b`.
  - Spatial store has synthetic `ctx*` nodes produced by the replay stub, unrelated to task semantics.
- **Replay/consolidation didn’t run** (`replay.samples: 0`), so persisted stores do not reflect memory formed from evaluation tasks.
- **Semantic suite looks saturated** for memory variants (EM≈1.0), while baselines read as 0.0 only because scoring is disabled. This produces misleading “SaturationSuspect / GateNoOp” warnings.
- **Spatial suite** shows **type-mismatch** between expected answers and predictions (paths vs indices/coords), so metrics are consistently 0.0 regardless of actual behavior.

**Bottom line:** The current run is **not** a reliable measure of algorithmic uplift. Fix scoring, replay/persistence, and output formats before interpreting EM/F1 or gating telemetry.

---

## What did the algorithms actually memorize? (`runs/.../stores/`)
### Episodic (HEI‑NW)
Store file: `runs/20250901_50_1337_2025/stores/hei_nw/hei_20250901_50_1337_2025/episodic.jsonl`
```json
{"schema": "episodic.v1", "id": 1, "key": [0.3535533845424652, 0.3535533845424652, 0.3535533845424652, 0.3535533845424652, 0.3535533845424652, 0.3535533845424652, 0.3535533845424652, 0.3535533845424652], "value": {"tokens_span": null, "entity_slots": null, "state_sketch": null, "salience_tags": null, "provenance": "dummy"}, "ts": 1756711800.273869, "salience": 1.0}
{"schema": "episodic.v1", "id": 2, "key": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": {"tokens_span": null, "entity_slots": null, "state_sketch": null, "salience_tags": null, "provenance": "the Mall"}, "ts": 1756711821.138685, "salience": 1.0}
{"schema": "episodic.v1", "id": 3, "key": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": {"tokens_span": null, "entity_slots": null, "state_sketch": null, "salience_tags": null, "provenance": "the Mall"}, "ts": 1756711830.9424448, "salience": 1.0}
{"schema": "episodic.v1", "id": 4, "key": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": {"tokens_span": null, "entity_slots": null, "state_sketch": null, "salience_tags": null, "provenance": "the Mall"}, "ts": 1756711830.943678, "salience": 1.0}
{"schema": "episodic.v1", "id": 5, "key": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": {"tokens_span": null, "entity_slots": null, "state_sketch": null, "salience_tags": null, "provenance": "the Mall"}, "ts": 1756711830.9448552, "salience": 1.0}
```
- Only a few entries are present; values show `"provenance": "dummy"` and null sketches/slots.
- This matches the code path where `_init_modules` writes a single dummy vector to exercise plumbing.
- With `replay.samples: 0`, no task-derived traces were consolidated or persisted.

### Relational (SGC‑RSS)
Store file: `runs/20250901_50_1337_2025/stores/sgc_rss/sgc_20250901_50_1337_2025/kg.jsonl`
```json
{"schema": "relational.v1", "type": "node", "name": "a", "embedding": null}
{"schema": "relational.v1", "type": "node", "name": "b", "embedding": null}
{"schema": "relational.v1", "type": "edge", "id": 1, "src": "a", "relation": "rel", "dst": "b", "context": "a rel b", "time": null, "conf": 1.0, "provenance": null, "embedding": null}
```
- The KG contains only a trivial node/edge example (`a →rel→ b`). No task entities/relations were ingested.

### Spatial (SMPD)
Store file: `runs/20250901_50_1337_2025/stores/smpd/smpd_20250901_50_1337_2025/spatial.jsonl` (first lines)
```json
{"schema": "spatial.v1", "type": "meta", "next_id": 52, "last_obs": 51, "step": 202, "position": [0.0, 0.0], "last_coord": null}
{"schema": "spatial.v1", "type": "node", "id": 0, "context": "a", "coord": [37.092, 62.679], "last_seen": 1}
{"schema": "spatial.v1", "type": "node", "id": 1, "context": "b", "coord": [22.982, 27.721], "last_seen": 2}
{"schema": "spatial.v1", "type": "node", "id": 2, "context": "ctx0", "coord": [12.86, 10.364], "last_seen": 153}
{"schema": "spatial.v1", "type": "node", "id": 3, "context": "ctx1", "coord": [29.373, 53.612], "last_seen": 4}
```
- The graph shows synthetic `ctx*` observations, consistent with the replay stub inserting generic contexts, not task-derived spatial knowledge.

> **Inference:** Persisted stores do not reflect meaningful memory formation from the evaluation tasks. Any apparent performance differences are **not** explained by the stored memories.

---

## Are the metrics meaningful?
### 1) Baseline metrics are missing per‑item scores
Baseline (semantic/core, seed=1337) per‑item preview:

| answer   | pred   |   em_raw |   em_norm |   f1 |
|:---------|:-------|---------:|----------:|-----:|
| Berlin   | Berlin |      nan |       nan |  nan |
| London   | London |      nan |       nan |  nan |
| Berlin   | Berlin |      nan |       nan |  nan |

- Note the `NaN` values for `em_raw`, `em_norm`, `f1` despite `pred == answer` (e.g. `Berlin`).
- In `metrics.json` the *pre* aggregates are **zeroed** instead of computed:
```json
{
  "semantic": {
    "pre_em": 0.0,
    "pre_em_raw": 0.0,
    "pre_em_norm": 0.0,
    "pre_f1": 0.0,
    "pre_refusal_rate": 0.0
  },
  "compute": {
    "input_tokens": 3150,
    "generated_tokens": 114,
    "total_tokens": 3264,
    "time_ms_per_100": 128.13225257353483,
    "rss_mb": 2068.890625,
    "latency_ms_mean": 83.60711666005955
  }
}

```
- Reporting then averages these zeros, making all baselines look like EM=0.0.

### 2) Memory variants *do* compute per‑item scores (episodic example)
| answer   | pred               |   em_raw |   em_norm |   f1 |
|:---------|:-------------------|---------:|----------:|-----:|
| Carol    | Carol.             |        0 |         1 |  0   |
| helped   | helped at the Mall |        0 |         0 |  0.4 |
| Carol    | Alice              |        0 |         0 |  0   |

But `metrics.json` shows:
```json
{
  "samples": 0
}

```
- With `replay.samples: 0`, no consolidation occurred; the uplift is therefore not attributable to stored traces.

### 3) Spatial outputs are mis‑scored due to format/type mismatch
Spatial (memory/smpd) per‑item preview:

| answer   | pred                                                                         |   em_raw |   em_norm |   f1 |
|:---------|:-----------------------------------------------------------------------------|---------:|----------:|-----:|
| 2        | 7                                                                            |        0 |         0 |    0 |
| UUUU     | (4, 4) -> (3, 4) -> (2, 4) -> (1, 4) -> (1, 3) -> (1, 2) -> (1, 1) -> (1, 0) |        0 |         0 |    0 |
| [3, 5]   | (3, 4)                                                                       |        0 |         0 |    0 |

- The dataset expects short answers like indices/coords or move codes (e.g. `2`, `[3,5]`, `UUUU`), while predictions are full path strings.
- EM/F1 will be 0.0 in this configuration even if the model’s *behavior* is sensible. This invalidates spatial conclusions.

---

## Quick roll‑up (mean over seeds)
| preset               | suite_task        |   em_mean |   f1_mean |
|:---------------------|:------------------|----------:|----------:|
| baselines/core       | episodic          |     0     |     0     |
| baselines/longctx    | episodic          |     0     |     0     |
| baselines/rag        | episodic          |     0     |     0     |
| baselines/span_short | episodic          |     0     |     0     |
| memory/hei_nw        | episodic          |     0.344 |     0.229 |
| baselines/core       | episodic_capacity |     0     |     0     |
| baselines/longctx    | episodic_capacity |     0     |     0     |
| baselines/rag        | episodic_capacity |     0     |     0     |
| baselines/span_short | episodic_capacity |     0     |     0     |
| memory/hei_nw        | episodic_capacity |     0.92  |     0.19  |
| baselines/core       | episodic_cross    |     0     |     0     |
| baselines/longctx    | episodic_cross    |     0     |     0     |
| baselines/rag        | episodic_cross    |     0     |     0     |
| baselines/span_short | episodic_cross    |     0     |     0     |
| memory/hei_nw        | episodic_cross    |     1     |     0.29  |
| baselines/core       | episodic_multi    |     0     |     0     |
| baselines/longctx    | episodic_multi    |     0     |     0     |
| baselines/rag        | episodic_multi    |     0     |     0     |
| baselines/span_short | episodic_multi    |     0     |     0     |
| memory/hei_nw        | episodic_multi    |     0.57  |     0.56  |
| baselines/core       | semantic          |     0     |     0     |
| baselines/longctx    | semantic          |     0     |     0     |
| baselines/rag        | semantic          |     0     |     0     |
| baselines/span_short | semantic          |     0     |     0     |
| memory/sgc_rss       | semantic          |     1     |     0.917 |
| baselines/core       | spatial           |     0     |     0     |
| baselines/longctx    | spatial           |     0     |     0     |
| baselines/rag        | spatial           |     0     |     0     |
| baselines/span_short | spatial           |     0     |     0     |
| memory/smpd          | spatial           |     0.016 |     0     |

> **Caution:** EM≈1.0 for `memory/sgc_rss` on `semantic` is consistent with an easy/saturated bench *and* the baseline scoring bug. Treat all apparent uplifts with skepticism until fixes (below) are applied.

---

## Recommendations & concrete fixes (prioritized)
1. **Always compute pre‑phase per‑item metrics.**
   - In the harness, `compute_metrics=False` is passed for the pre‑replay sweep; set it to `True` so `metrics.csv` and `metrics.json` carry real `pre_*` values.
   - Ensure `reports/` rolls up *post* vs *pre* correctly; don’t coerce missing `pre_*` to 0.0.
2. **Only persist stores after meaningful replay.**
   - Persist stores **only if** `replay.samples > 0`, or annotate/store versions to distinguish stub vs real contents.
   - Remove dummy writes from `_init_modules` for production runs, or tag them so reporters filter them out.
3. **Fix spatial output/metric alignment.**
   - Decide the canonical spatial target (`final coord`, `steps`, or `policy`) and make the generator, prompt, and scorer agree.
   - Provide task‑type specific formatting functions; add unit tests validating EM/F1 on canned examples.
4. **Strengthen semantic bench to avoid ceiling effects.**
   - Use the `dataset_profile` hook to run a `hard` profile (or increase entity overlap/decoys) so baselines land at a reasonable pre‑EM (e.g. 0.3–0.7) and memory has room to improve.
5. **Gating telemetry must reflect real activity.**
   - Current `GateNoOp` and empty/`nan` counters suggest gates aren’t invoked. Emit counters on both accept and block paths, and assert invariants (e.g., attempts > 0 when memory is enabled).
   - Add a per‑suite check: if memory is on but retrieval tokens == 0 across tasks, fail the run as misconfigured.
6. **Retrieval telemetry sanity checks.**
   - Validate `hits <= total_k`, `hit_rate_at_k ≈ hits/total_k`, and that `requests ≈ #tasks` when retrieval is enabled.
   - Include `k` and `bsz` in `metrics.json` so downstream plots can compute expected totals.
7. **Report hygiene & guardrails.**
   - In `reports/index.md`, avoid truncating numeric cells (currently rendered like `0.000...00`).
   - If `pre_*` metrics are missing, annotate as _missing_ instead of zero and add a ⚠️ row‑level warning.
8. **Unit & smoke tests.**
   - Add tests that run a tiny suite with deterministic predictions and assert that `pre` and `post` aggregates match per‑item CSVs.
   - Add tests that stores grow by the expected amount after replay and that persisted JSONLs contain non‑stub entries.

---

## Decision: are we ready to interpret uplift?
**No.** Given missing pre‑metrics, absent replay, dummy stores, and spatial mis‑scoring, the pipeline does not yet support a credible evaluation of HEI‑NW, SGC‑RSS, or SMPD.

### What to change before re‑running `EVAL_PROTOCOL.md`
- Enable pre‑phase scoring and verify baselines produce sensible EM/F1 on all suites.
- Run at least one replay cycle and persist stores; inspect that stores contain **non‑stub** traces/entities consistent with tasks.
- Fix spatial output/metric consistency; re‑generate the suite if needed.
- Re‑run reports and confirm that `GateNoOp` disappears where expected and telemetry shows non‑zero activity.

---

## Appendix: file paths referenced
- Summaries: `/mnt/data/hippo-llm-memory/hippo-llm-memory-main/runs/20250901_50_1337_2025/summaries/*.csv|json`
- Stores:
  - Episodic: `/mnt/data/hippo-llm-memory/hippo-llm-memory-main/runs/20250901_50_1337_2025/stores/hei_nw/hei_20250901_50_1337_2025/episodic.jsonl`
  - Relational: `/mnt/data/hippo-llm-memory/hippo-llm-memory-main/runs/20250901_50_1337_2025/stores/sgc_rss/sgc_20250901_50_1337_2025/kg.jsonl`
  - Spatial: `/mnt/data/hippo-llm-memory/hippo-llm-memory-main/runs/20250901_50_1337_2025/stores/smpd/smpd_20250901_50_1337_2025/spatial.jsonl`
- Reports roll‑up: `/mnt/data/hippo-llm-memory/hippo-llm-memory-main/reports/20250901_50_1337_2025/index.md` and per‑suite `summary.md`
