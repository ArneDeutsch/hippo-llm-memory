# episodic_capacity Summary

| Preset | Size | EM (raw) | EM (norm) | EM | f1 | overlong | format_violation | generated_tokens | input_tokens | latency_ms_mean | refusal_rate | rss_mb | time_ms_per_100 | total_tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baselines/core | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | – | – |
| baselines/longctx | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | – | – |
| baselines/rag | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | – | – |
| baselines/span_short | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | – | – |
| memory/hei_nw | 50 | 0.000 ± 0.000 | 0.940 ± 0.000 | 0.940 ± 0.000 | 0.164 ± 0.000 | 0.060 ± 0.000 | 0.780 ± 0.000 | 192.000 ± 0.000 | 15150.000 ± 0.000 | 214.800 ± 0.000 | 0.000 ± 0.000 | 1803.586 ± 0.000 | 70.012 ± 0.000 | 15342.000 ± 0.000 |

## Retrieval Telemetry
| mem | requests | hit_rate_at_k | avg_latency_ms | tokens_returned |
|---|---|---|---|---|
| episodic | 50 | 1.000 | 0.464 | 50 |
| relational | 50 | 0.000 | 0.107 | 50 |
| spatial | 50 | 0.000 | 0.063 | 800 |

## Gate Telemetry
### Gate ON
| mem | attempts | inserted | aggregated | routed_to_episodic | blocked_new_edges |
|---|---|---|---|---|---|
| relational | 0 | 0 | 0 | 0 | 0 |
| spatial | 0 | 0 | 0 | 0 | 0 |
