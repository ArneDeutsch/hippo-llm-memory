# Run Review — 20250901_50_1337_2025

_Generated: 2025-09-01 14:39 UTC_

## Scope
- Inspect artifacts under `runs/20250901_50_1337_2025/` and `reports/20250901_50_1337_2025/` for SIZES=50, SEEDS=(1337, 2025).
- Verify what each algorithm actually **memorizes** (`stores/`), whether telemetry is meaningful, and whether the evaluation design can reveal uplift.

## Quick Integrity Check (EVAL_PLAN §0.2)

> ❌ **This run does *not* meet “meaningful run” criteria.**

- Baseline *pre* metrics are missing (EM/F1 are null in baselines).
- Store meta indicates STUB sources for: SGC_RSS, SMPD.
- Gating shows 0 attempts across all presets.

## What the stores contain

**Episodic store (HEI‑NW)**  
File: `runs/20250901_50_1337_2025/stores/hei_nw/hei_20250901_50_1337_2025/episodic.jsonl`  
• Items: **53**  
• Unique keys: **4** (variance=0.008722)  
• Provenance counts: {'the Mall': 14, 'the Cafe': 15, 'the Library': 13, 'the Park': 11}  
• Populated fields: {'tokens_span': 0, 'entity_slots': 0, 'state_sketch': 0, 'salience_tags': 0}

**Relational store (SGC‑RSS)**  
File: `runs/20250901_50_1337_2025/stores/sgc_rss/sgc_20250901_50_1337_2025/kg.jsonl`  
• Items: **0**

**Spatial store (SMPD)**  
File: `runs/20250901_50_1337_2025/stores/smpd/smpd_20250901_50_1337_2025/spatial.jsonl`  
• Items: **0**


**Reading:** HEI‑NW store looks synthetic: only four unique keys across 53 items; all content fields (`tokens_span`, `entity_slots`, `state_sketch`, `salience_tags`) are empty. SGC‑RSS and SMPD stores are empty (stubs).

## Metrics — memory presets (pre‑phase)

| preset   | task              |   em_mean |   em_raw_mean |   f1_mean |   overlong_sum |   format_violations_sum |   gate_attempts_total | store_sources   |   store_sizes |
|:---------|:------------------|----------:|--------------:|----------:|---------------:|------------------------:|----------------------:|:----------------|--------------:|
| hei_nw   | episodic          |      0.37 |          0.04 |  0.193335 |             53 |                      84 |                     0 | replay          |            53 |
| hei_nw   | episodic_capacity |      0.94 |          0    |  0.230476 |              6 |                      68 |                     0 | replay          |            53 |
| hei_nw   | episodic_cross    |      1    |          0    |  0.256667 |              0 |                      61 |                     0 | replay          |            53 |
| hei_nw   | episodic_multi    |      0.57 |          0.57 |  0.57     |             40 |                      40 |                     0 | replay          |            53 |
| sgc_rss  | semantic          |      0.98 |          0.95 |  0.95     |              0 |                       4 |                     0 | stub            |             0 |
| smpd     | spatial           |      0    |          0    |  0        |              6 |                       0 |                     0 | stub            |             0 |

> ⚠️ **Scoring anomaly:** `episodic_cross` shows `pre_em_raw = 0.0` while `pre_em_norm = 1.0` for both seeds. Normalization appears to mask zero raw accuracy. Treat the normalized EM with caution and fix the scorer.

## Parameter sweeps (pre‑phase)

| name            | suite    |   em |       f1 | config_hint                  |
|:----------------|:---------|-----:|---------:|:-----------------------------|
| hei_nw_tau_0.3  | episodic | 0.36 | 0.236939 | episodic.gate.enabled=True   |
| hei_nw_tau_0.5  | episodic | 0.36 | 0.214278 | episodic.gate.enabled=True   |
| hei_nw_tau_0.7  | episodic | 0.32 | 0.206247 | episodic.gate.enabled=True   |
| sgc_rss_thr_0.4 | semantic | 1    | 0.94     | relational.gate.enabled=True |
| sgc_rss_thr_0.6 | semantic | 1    | 0.92     | relational.gate.enabled=True |
| sgc_rss_thr_0.8 | semantic | 1    | 0.9      | relational.gate.enabled=True |
| smpd_thr_0.5    | spatial  | 0    | 0        | spatial.gate.enabled=True    |
| smpd_thr_1.0    | spatial  | 0    | 0        | spatial.gate.enabled=True    |
| smpd_thr_2.0    | spatial  | 0    | 0        | spatial.gate.enabled=True    |

## Findings

- **Stores are not real memory artifacts.** SGC‑RSS and SMPD stores are empty stubs; HEI‑NW episodic items have constant‑like keys and no content.
- **Gating never triggers** (0 attempts across all presets), so no writes occur during evaluation.
- **Replay is effectively off** (HEI‑NW shows `source: replay` with 53 items, but SGC‑RSS/SMPD are `source: stub` and `replay_samples=0`).
- **Semantic suite is saturated** (EM≈1.0 with an empty KG), implying tasks are solvable without relational memory. We are not measuring memory benefit.
- **Spatial suite mismatched with scorer.** EM and success are 0.0 with large suboptimality ratios; audit samples show answer formats not aligned with the metric and no memory usage.
- **Baseline *pre* metrics are missing.** Reports display `–` or `None`, so uplift vs. baselines is not computable.
- **Metric normalization can be misleading.** For `episodic_cross`, raw EM=0 but normalized EM=1.0 — the report page emphasizes the normalized value.

## Recommendations

- **Fail fast checks in the harness**: abort a run when (a) any `store_meta.source == 'stub'`, (b) `gate.attempts == 0` for the target memory, or (c) baseline `pre_*` metrics are null.
- **Compute real baseline `pre_*` metrics** and render them in `index.md` with CIs; ensure `metrics.csv` rows include `em_raw`, `em_norm`, `f1`.
- **Fix EM normalization** so that EM/EM_raw/EM_norm are consistent and clearly defined; do not report `EM` if `EM_raw==0`.
- **Make semantic tasks depend on the KG**: during *teach*, extract tuples and write them; during *test*, mask the prompt so the answer is only recoverable from KG retrieval; add an ablation that empties the KG to confirm a drop.
- **Make spatial tasks consume the map/macros**: write observed transitions into `PlaceGraph`, plan with A*/shortest path hints, and grade on path feasibility and near‑optimality; add unit tests on toy grids.
- **Verify episodic content**: persist token spans and entity slots; keys should be diverse (not 4 unique vectors); add tests for kNN recall on partial cues.
- **Tighten telemetry**: log `retrieval.requests/k/hits_at_k` > 0 for any suite using memory; otherwise flag as `NoRetrieval` in the report.
- **Parameter sweeps should move the needle**: current sweeps show SGC‑RSS EM=1.0 for all thresholds and SMPD EM=0.0 for all — use checks to mark such sweeps as non‑informative.
- **Document dataset profiles** (`default` vs `hard`) and choose profiles that avoid baseline saturation for semantic, while keeping episodic capacity/cross truly hard.

## Minimal Go/No‑Go


**No‑Go** until the following are true:
1) Baseline *pre* metrics are present and non‑NaN for all suites.  
2) Store meta shows **non‑stub** sources with **replay_samples ≥ 1** for each memory.  
3) Gating attempts > 0 and at least some writes occur in *teach*.  
4) For semantic + spatial suites, ablating the store causes a measurable drop (ΔEM ≥ 0.2).  
5) `episodic_cross` raw and normalized EM align (no 0/1 contradiction).
