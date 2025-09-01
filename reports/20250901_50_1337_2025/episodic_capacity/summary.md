# episodic_capacity Summary

> `seeds:1337,2025` `sizes:50` `profile:hard` `replay.samples:0` `store:replay`

## Uplift
| Preset | Size | EM (raw) | EM (norm) | EM | f1 | overlong | format_violation | generated_tokens | input_tokens | latency_ms_mean | refusal_rate | rss_mb | store_size | time_ms_per_100 | total_tokens | ⚠️ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baselines/core | 50 | – | – | – | – | – | – | 189.000 ± 3.920 | 27950.000 ± 0.000 | 317.923 ± 3.495 | – | 2945.443 ± 1.451 | 0.000 ± 0.000 | 56.497 ± 0.613 | 28139.000 ± 3.920 |  |
| baselines/longctx | 50 | – | – | – | – | – | – | 186.000 ± 5.880 | 28100.000 ± 0.000 | 321.542 ± 0.885 | – | 2949.076 ± 0.057 | 0.000 ± 0.000 | 56.843 ± 0.145 | 28286.000 ± 5.880 |  |
| baselines/rag | 50 | – | – | – | – | – | – | 177.000 ± 11.760 | 27950.000 ± 0.000 | 312.347 ± 7.457 | – | 2946.906 ± 0.230 | 0.000 ± 0.000 | 55.530 ± 1.302 | 28127.000 ± 11.760 |  |
| baselines/span_short | 50 | – | – | – | – | – | – | 175.000 ± 11.760 | 27950.000 ± 0.000 | 310.936 ± 6.410 | – | 2946.537 ± 0.195 | 0.000 ± 0.000 | 55.283 ± 1.117 | 28125.000 ± 11.760 |  |
| memory/hei_nw | 50 | 0.000 ± 0.000 | 0.940 ± 0.000 | 0.940 ± 0.000 | 0.230 ± 0.026 | 0.060 ± 0.000 | 0.680 ± 0.039 | 179.500 ± 2.940 | 27950.000 ± 0.000 | 339.362 ± 1.407 | 0.000 ± 0.000 | 1768.486 ± 150.732 | 53.000 ± 0.000 | 60.327 ± 0.244 | 28129.500 ± 2.940 | ⚠️ GateNoOp |

### Warnings
- memory/hei_nw 50: GateNoOp

## Retrieval Telemetry
> Hits reflect actual recalled traces; cue-only fallbacks are excluded from telemetry.
| mem | k | batch_size | requests | hits_at_k | hit_rate_at_k | tokens_returned | avg_latency_ms |
|---|---|---|---|---|---|---|---|
| episodic | 0 | 0 | 10 | 10 | 0.200 | 10 | 0.090 |
| relational | 0 | 0 | 0 | 0 | 0.000 | 0 | 0.000 |
| spatial | 0 | 0 | 0 | 0 | 0.000 | 0 | 0.000 |

## Gate Telemetry
### Gate ON
| mem | attempts | inserted | aggregated | routed_to_episodic | blocked_new_edges |
|---|---|---|---|---|---|
| episodic | 0 | 0 | 0 | 0 | 0 |
| relational | 0 | 0 | 0 | 0 | 0 |
| spatial | 0 | 0 | 0 | 0 | 0 |
