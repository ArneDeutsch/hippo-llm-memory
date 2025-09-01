# episodic Summary

> `seeds:1337,2025` `sizes:50` `profile:default` `replay.samples:0` `store:replay`

## Uplift
| Preset | Size | EM (raw) | EM (norm) | EM | f1 | overlong | format_violation | generated_tokens | input_tokens | latency_ms_mean | refusal_rate | rss_mb | store_size | time_ms_per_100 | total_tokens | ⚠️ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baselines/core | 50 | – | – | – | – | – | – | 317.000 ± 3.920 | 2771.000 ± 3.920 | 187.969 ± 34.663 | – | 2342.295 ± 1079.221 | 0.000 ± 0.000 | 304.404 ± 56.135 | 3088.000 ± 0.000 |  |
| baselines/longctx | 50 | – | – | – | – | – | – | 313.000 ± 39.200 | 2921.000 ± 3.920 | 174.359 ± 25.302 | – | 2707.760 ± 0.149 | 0.000 ± 0.000 | 269.498 ± 35.528 | 3234.000 ± 43.120 |  |
| baselines/rag | 50 | – | – | – | – | – | – | 304.000 ± 1.960 | 2771.000 ± 3.920 | 165.735 ± 1.381 | – | 2703.191 ± 0.130 | 0.000 ± 0.000 | 269.533 ± 1.730 | 3075.000 ± 5.880 |  |
| baselines/span_short | 50 | – | – | – | – | – | – | 312.000 ± 27.440 | 2771.000 ± 3.920 | 169.668 ± 10.750 | – | 2698.346 ± 0.203 | 0.000 ± 0.000 | 275.183 ± 15.334 | 3083.000 ± 23.520 |  |
| memory/hei_nw | 50 | 0.040 ± 0.000 | 0.370 ± 0.059 | 0.370 ± 0.059 | 0.193 ± 0.030 | 0.530 ± 0.020 | 0.840 ± 0.078 | 296.500 ± 24.500 | 2771.000 ± 3.920 | 182.299 ± 11.711 | 0.000 ± 0.000 | 1723.447 ± 61.981 | 53.000 ± 0.000 | 297.159 ± 16.332 | 3067.500 ± 28.420 | ⚠️ GateNoOp |

### Warnings
- memory/hei_nw 50: GateNoOp

## Retrieval Telemetry
> Hits reflect actual recalled traces; cue-only fallbacks are excluded from telemetry.
| mem | k | batch_size | requests | hits_at_k | hit_rate_at_k | tokens_returned | avg_latency_ms |
|---|---|---|---|---|---|---|---|
| episodic | 0 | 0 | 10 | 10 | 0.200 | 10 | 0.094 |
| relational | 0 | 0 | 0 | 0 | 0.000 | 0 | 0.000 |
| spatial | 0 | 0 | 0 | 0 | 0.000 | 0 | 0.000 |

## Gate Telemetry
### Gate ON
| mem | attempts | inserted | aggregated | routed_to_episodic | blocked_new_edges |
|---|---|---|---|---|---|
| episodic | 0 | 0 | 0 | 0 | 0 |
| relational | 0 | 0 | 0 | 0 | 0 |
| spatial | 0 | 0 | 0 | 0 | 0 |
