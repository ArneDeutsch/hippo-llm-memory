# episodic_cross Summary

> ⚠️ saturated: pre_em_norm ≥ 0.98

## Uplift
| Preset | Size | EM (raw) | EM (norm) | EM | f1 | overlong | format_violation | generated_tokens | input_tokens | latency_ms_mean | refusal_rate | rss_mb | store_size | time_ms_per_100 | total_tokens | ⚠️ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baselines/core | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 167.500 ± 2.940 | 2058.500 ± 4.900 | 114.318 ± 8.229 | 0.000 ± 0.000 | 2384.904 ± 1125.741 | 0.000 ± 0.000 | 256.850 ± 17.584 | 2226.000 ± 7.840 |  |
| baselines/longctx | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 169.000 ± 5.880 | 2208.500 ± 4.900 | 112.195 ± 3.122 | 0.000 ± 0.000 | 2963.740 ± 0.433 | 0.000 ± 0.000 | 236.017 ± 5.491 | 2377.500 ± 10.780 |  |
| baselines/rag | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 169.000 ± 3.920 | 2058.500 ± 4.900 | 112.544 ± 2.574 | 0.000 ± 0.000 | 2963.217 ± 0.042 | 0.000 ± 0.000 | 252.705 ± 5.889 | 2227.500 ± 0.980 |  |
| baselines/span_short | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 168.000 ± 5.880 | 2058.500 ± 4.900 | 111.632 ± 4.190 | 0.000 ± 0.000 | 2962.176 ± 1.501 | 0.000 ± 0.000 | 250.775 ± 9.294 | 2226.500 ± 0.980 |  |
| memory/hei_nw | 50 | 0.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.290 ± 0.033 | 0.000 ± 0.000 | 0.560 ± 0.039 | 161.000 ± 1.960 | 2058.500 ± 4.900 | 112.487 ± 1.898 | 0.000 ± 0.000 | 1707.293 ± 14.149 | 5.000 ± 0.000 | 253.477 ± 3.489 | 2219.500 ± 6.860 | ⚠️ SaturationSuspect, GateNoOp |

### Warnings
- memory/hei_nw 50: SaturationSuspect, GateNoOp

## Retrieval Telemetry
> Hits reflect actual recalled traces; cue-only fallbacks are excluded from telemetry.
| mem | requests | hits | hit_rate_at_k | tokens_returned | avg_latency_ms |
|---|---|---|---|---|---|
| episodic | 10 | 10 | 0.200 | 10 | 0.088 |
| relational | 0 | 0 | 0.000 | 0 | 0.000 |
| spatial | 0 | 0 | 0.000 | 0 | 0.000 |

## Gate Telemetry
### Gate ON
| mem | attempts | inserted | aggregated | routed_to_episodic | blocked_new_edges |
|---|---|---|---|---|---|
| episodic | 0 | 0 | 0 | 0 | 0 |
| relational | 0 | 0 | 0 | 0 | 0 |
| spatial | 0 | 0 | 0 | 0 | 0 |
