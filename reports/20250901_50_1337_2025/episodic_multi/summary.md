# episodic_multi Summary

> `seeds:1337,2025` `sizes:50` `profile:default` `replay.samples:0` `store:replay`

## Uplift
| Preset | Size | EM (raw) | EM (norm) | EM | f1 | overlong | format_violation | generated_tokens | input_tokens | latency_ms_mean | refusal_rate | rss_mb | store_size | time_ms_per_100 | total_tokens | ⚠️ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baselines/core | 50 | – | – | – | – | – | – | 164.000 ± 31.360 | 6100.000 ± 0.000 | 115.996 ± 14.421 | – | 2697.971 ± 0.057 | 0.000 ± 0.000 | 92.598 ± 11.046 | 6264.000 ± 31.360 |  |
| baselines/longctx | 50 | – | – | – | – | – | – | 122.500 ± 14.700 | 6250.000 ± 0.000 | 98.623 ± 4.328 | – | 2708.258 ± 0.191 | 0.000 ± 0.000 | 77.404 ± 3.214 | 6372.500 ± 14.700 |  |
| baselines/rag | 50 | – | – | – | – | – | – | 163.500 ± 22.540 | 6100.000 ± 0.000 | 118.044 ± 9.444 | – | 2707.387 ± 1.164 | 0.000 ± 0.000 | 94.249 ± 7.199 | 6263.500 ± 22.540 |  |
| baselines/span_short | 50 | – | – | – | – | – | – | 171.000 ± 25.480 | 6100.000 ± 0.000 | 121.004 ± 9.868 | – | 2699.914 ± 2.618 | 0.000 ± 0.000 | 96.496 ± 7.474 | 6271.000 ± 25.480 |  |
| memory/hei_nw | 50 | 0.570 ± 0.098 | 0.570 ± 0.098 | 0.570 ± 0.098 | 0.570 ± 0.098 | 0.400 ± 0.078 | 0.400 ± 0.078 | 160.000 ± 11.760 | 6100.000 ± 0.000 | 131.306 ± 5.168 | 0.000 ± 0.000 | 1745.766 ± 102.532 | 53.000 ± 0.000 | 104.900 ± 3.931 | 6260.000 ± 11.760 | ⚠️ GateNoOp |

### Warnings
- memory/hei_nw 50: GateNoOp

## Retrieval Telemetry
> Hits reflect actual recalled traces; cue-only fallbacks are excluded from telemetry.
| mem | k | batch_size | requests | hits_at_k | hit_rate_at_k | tokens_returned | avg_latency_ms |
|---|---|---|---|---|---|---|---|
| episodic | 0 | 0 | 10 | 10 | 0.200 | 10 | 0.088 |
| relational | 0 | 0 | 0 | 0 | 0.000 | 0 | 0.000 |
| spatial | 0 | 0 | 0 | 0 | 0.000 | 0 | 0.000 |

## Gate Telemetry
### Gate ON
| mem | attempts | inserted | aggregated | routed_to_episodic | blocked_new_edges |
|---|---|---|---|---|---|
| episodic | 0 | 0 | 0 | 0 | 0 |
| relational | 0 | 0 | 0 | 0 | 0 |
| spatial | 0 | 0 | 0 | 0 | 0 |
