# episodic_cross Summary

| Preset | Size | EM (raw) | EM (norm) | EM | f1 | overlong | format_violation | generated_tokens | input_tokens | latency_ms_mean | refusal_rate | rss_mb | time_ms_per_100 | total_tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baselines/core | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | – | – |
| baselines/longctx | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | – | – |
| baselines/rag | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | – | – |
| baselines/span_short | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | – | – |
| memory/hei_nw | 50 | 0.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.200 ± 0.000 | 0.000 ± 0.000 | 0.700 ± 0.000 | 177.000 ± 0.000 | 2050.000 ± 0.000 | 115.122 ± 0.000 | 0.000 ± 0.000 | 1708.293 ± 0.000 | 258.536 ± 0.000 | 2227.000 ± 0.000 |

## Retrieval Telemetry
| mem | requests | hit_rate_at_k | avg_latency_ms | tokens_returned |
|---|---|---|---|---|
| episodic | 50 | 1.000 | 0.445 | 50 |
| relational | 50 | 0.000 | 0.108 | 50 |
| spatial | 50 | 0.000 | 0.063 | 800 |

## Gate Telemetry
### Gate ON
| mem | attempts | inserted | aggregated | routed_to_episodic | blocked_new_edges |
|---|---|---|---|---|---|
| relational | 0 | 0 | 0 | 0 | 0 |
| spatial | 0 | 0 | 0 | 0 | 0 |
