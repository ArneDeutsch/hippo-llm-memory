# spatial Summary

| Preset | Size | EM (raw) | EM (norm) | EM | f1 | overlong | format_violation | generated_tokens | input_tokens | latency_ms_mean | refusal_rate | rss_mb | steps_to_goal | suboptimality_ratio | success_rate | time_ms_per_100 | total_tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baselines/core | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | – | – | – | – | – |
| baselines/longctx | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | – | – | – | – | – |
| baselines/rag | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | – | – | – | – | – |
| baselines/span_short | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | – | – | – | – | – |
| memory/smpd | 50 | 0.000 ± 0.000 | 0.040 ± 0.000 | 0.040 ± 0.000 | 0.000 ± 0.000 | 0.340 ± 0.000 | 0.020 ± 0.000 | 1018.000 ± 0.000 | 3484.000 ± 0.000 | 549.275 ± 0.000 | 0.000 ± 0.000 | 1782.590 ± 0.000 | 4.320 ± 0.000 | 1.266 ± 0.000 | 0.040 ± 0.000 | 610.070 ± 0.000 | 4502.000 ± 0.000 |

## Retrieval Telemetry
| mem | requests | hit_rate_at_k | avg_latency_ms | tokens_returned |
|---|---|---|---|---|
| episodic | 50 | 1.000 | 0.447 | 50 |
| relational | 50 | 0.000 | 0.109 | 50 |
| spatial | 50 | 0.000 | 0.062 | 800 |

## Gate Telemetry
### Gate ON
| mem | attempts | inserted | aggregated | routed_to_episodic | blocked_new_edges |
|---|---|---|---|---|---|
| relational | 0 | 0 | 0 | 0 | 0 |
| spatial | 0 | 0 | 0 | 0 | 0 |
