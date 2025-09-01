# episodic_cross Summary

> `seeds:1337,2025` `sizes:50` `profile:hard` `replay.samples:0` `store:replay`

> ⚠️ saturated: pre_em_norm ≥ 0.98

## Uplift
| Preset | Size | EM (raw) | EM (norm) | EM | f1 | overlong | format_violation | generated_tokens | input_tokens | latency_ms_mean | refusal_rate | rss_mb | store_size | time_ms_per_100 | total_tokens | ⚠️ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baselines/core | 50 | – | – | – | – | – | – | 170.000 ± 0.000 | 2058.500 ± 4.900 | 104.031 ± 7.792 | – | 2355.301 ± 1152.074 | 0.000 ± 0.000 | 233.465 ± 16.972 | 2228.500 ± 4.900 |  |
| baselines/longctx | 50 | – | – | – | – | – | – | 169.000 ± 3.920 | 2208.500 ± 4.900 | 100.368 ± 2.197 | – | 2948.244 ± 1.336 | 0.000 ± 0.000 | 211.135 ± 3.836 | 2377.500 ± 8.820 |  |
| baselines/rag | 50 | – | – | – | – | – | – | 170.500 ± 0.980 | 2058.500 ± 4.900 | 101.258 ± 0.240 | – | 2946.859 ± 0.015 | 0.000 ± 0.000 | 227.201 ± 0.139 | 2229.000 ± 3.920 |  |
| baselines/span_short | 50 | – | – | – | – | – | – | 166.000 ± 3.920 | 2058.500 ± 4.900 | 100.874 ± 0.558 | – | 2946.352 ± 0.153 | 0.000 ± 0.000 | 226.800 ± 1.156 | 2224.500 ± 0.980 |  |
| memory/hei_nw | 50 | 0.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.257 ± 0.059 | 0.000 ± 0.000 | 0.610 ± 0.098 | 164.000 ± 1.960 | 2058.500 ± 4.900 | 114.537 ± 0.800 | 0.000 ± 0.000 | 1756.787 ± 124.204 | 53.000 ± 0.000 | 257.748 ± 2.593 | 2222.500 ± 6.860 | ⚠️ SaturationSuspect, GateNoOp |

### Warnings
- memory/hei_nw 50: SaturationSuspect, GateNoOp

## Retrieval Telemetry
> Hits reflect actual recalled traces; cue-only fallbacks are excluded from telemetry.
| mem | k | batch_size | requests | hits_at_k | hit_rate_at_k | tokens_returned | avg_latency_ms |
|---|---|---|---|---|---|---|---|
| episodic | 0 | 0 | 10 | 10 | 0.200 | 10 | 0.091 |
| relational | 0 | 0 | 0 | 0 | 0.000 | 0 | 0.000 |
| spatial | 0 | 0 | 0 | 0 | 0.000 | 0 | 0.000 |

## Gate Telemetry
### Gate ON
| mem | attempts | inserted | aggregated | routed_to_episodic | blocked_new_edges |
|---|---|---|---|---|---|
| episodic | 0 | 0 | 0 | 0 | 0 |
| relational | 0 | 0 | 0 | 0 | 0 |
| spatial | 0 | 0 | 0 | 0 | 0 |
