# Overall Summary

[![GatingActive-red](https://img.shields.io/badge/GatingActive-red)](runs/20250902/gates) [![BaselinesOK-brightgreen](https://img.shields.io/badge/BaselinesOK-brightgreen)](runs/20250902/baselines) [![NonStubStores-brightgreen](https://img.shields.io/badge/NonStubStores-brightgreen)](runs/20250902/stores) [![RetrievalActive-brightgreen](https://img.shields.io/badge/RetrievalActive-brightgreen)](runs/20250902/retrieval)

> ✅ Preflight passed

> single-seed run: CI bands unavailable

| Suite | Preset | EM (raw) | <span title="Normalized exact match (lowercase, no punctuation or articles)">EM (norm)</span> | EM | f1 | overlong | format_violation | gate_attempts | generated_tokens | input_tokens | latency_ms_mean | refusal_rate | retrieval_episodic_requests | retrieval_relational_requests | retrieval_spatial_requests | rss_mb | store_size | time_ms_per_100 | total_tokens | ⚠️ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| episodic | baselines/core | 0.040 | 0.040 | 0.040 | 0.181 | 0.520 | 0.840 | 0.000 | 294.000 | 2769.000 | 163.372 | 0.000 | 0.000 | 0.000 | 0.000 | 1692.621 | 0.000 | 266.732 | 3063.000 |  |

![Overall EM](assets/overall_em.png)

## Gate Telemetry
| status | mem | attempts | accepted | blocked | skipped | duplicate_rate | nodes_per_1k | edges_per_1k |
|---|---|---|---|---|---|---|---|---|
| on | episodic | 0 | 0 | 0 | 0 | nan | nan | nan |
| on | relational | 0 | 0 | 0 | 0 | nan | nan | nan |
| on | spatial | 0 | 0 | 0 | 0 | nan | nan | nan |

## Per-suite summaries
- [episodic](episodic/summary.md)