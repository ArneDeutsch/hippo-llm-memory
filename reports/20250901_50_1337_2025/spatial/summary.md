# spatial Summary

> `seeds:1337,2025` `sizes:50` `profile:default` `replay.samples:0` `store:stub`

## Uplift
| Preset | Size | EM (raw) | EM (norm) | EM | f1 | overlong | format_violation | generated_tokens | input_tokens | latency_ms_mean | refusal_rate | rss_mb | steps_to_goal | store_size | suboptimality_ratio | success_rate | time_ms_per_100 | total_tokens | ⚠️ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baselines/core | 50 | – | – | – | – | – | – | 814.500 ± 159.740 | 5850.000 ± 1764.000 | 431.917 ± 54.943 | – | 2697.281 ± 0.138 | – | 0.000 ± 0.000 | – | – | 331.652 ± 121.055 | 6664.500 ± 1604.260 |  |
| baselines/longctx | 50 | – | – | – | – | – | – | 902.500 ± 408.660 | 6000.000 ± 1764.000 | 478.267 ± 175.100 | – | 2708.082 ± 0.130 | – | 0.000 ± 0.000 | – | – | 356.529 ± 196.845 | 6902.500 ± 1355.340 |  |
| baselines/rag | 50 | – | – | – | – | – | – | 884.000 ± 254.800 | 5850.000 ± 1764.000 | 469.479 ± 87.727 | – | 2706.873 ± 0.019 | – | 0.000 ± 0.000 | – | – | 357.081 ± 145.164 | 6734.000 ± 1509.200 |  |
| baselines/span_short | 50 | – | – | – | – | – | – | 968.500 ± 342.020 | 5850.000 ± 1764.000 | 505.136 ± 141.540 | – | 2698.436 ± 0.264 | – | 0.000 ± 0.000 | – | – | 380.379 ± 183.119 | 6818.500 ± 1421.980 |  |
| memory/smpd | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.060 ± 0.118 | 0.000 ± 0.000 | 972.500 ± 420.420 | 5850.000 ± 1764.000 | 543.792 ± 199.529 | 0.000 ± 0.000 | 1773.717 ± 2.852 | 19.860 ± 11.329 | 0.000 ± 0.000 | 6.110 ± 2.221 | 0.000 ± 0.000 | 410.189 ± 227.010 | 6822.500 ± 1343.580 | ⚠️ GateNoOp |

### Warnings
- memory/smpd 50: GateNoOp

## Retrieval Telemetry
> Hits reflect actual recalled traces; cue-only fallbacks are excluded from telemetry.
| mem | k | batch_size | requests | hits_at_k | hit_rate_at_k | tokens_returned | avg_latency_ms |
|---|---|---|---|---|---|---|---|
| episodic | 0 | 0 | 0 | 0 | 0.000 | 0 | 0.000 |
| relational | 0 | 0 | 0 | 0 | 0.000 | 0 | 0.000 |
| spatial | 3 | 0 | 10 | 0 | 0.000 | 160 | 0.022 |

## Gate Telemetry
### Gate ON
| mem | attempts | inserted | aggregated | routed_to_episodic | blocked_new_edges |
|---|---|---|---|---|---|
| episodic | 0 | 0 | 0 | 0 | 0 |
| relational | 0 | 0 | 0 | 0 | 0 |
| spatial | 0 | 0 | 0 | 0 | 0 |
