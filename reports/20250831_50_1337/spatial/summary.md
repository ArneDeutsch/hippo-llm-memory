# spatial Summary

## Uplift
> single-seed run: CI bands unavailable

| Preset | Size | EM (raw) | EM (norm) | EM | f1 | overlong | format_violation | generated_tokens | input_tokens | latency_ms_mean | refusal_rate | rss_mb | steps_to_goal | store_size | suboptimality_ratio | success_rate | time_ms_per_100 | total_tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baselines/core | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | – | 5.000 ± 0.000 | – | – | – | – |
| baselines/longctx | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | – | 5.000 ± 0.000 | – | – | – | – |
| baselines/rag | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | – | 5.000 ± 0.000 | – | – | – | – |
| baselines/span_short | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | – | 5.000 ± 0.000 | – | – | – | – |
| memory/smpd | 50 | 0.000 ± 0.000 | 0.020 ± 0.000 | 0.020 ± 0.000 | 0.000 ± 0.000 | 0.340 ± 0.000 | 0.000 ± 0.000 | 1017.000 ± 0.000 | 3484.000 ± 0.000 | 545.677 ± 0.000 | 0.000 ± 0.000 | 1782.500 ± 0.000 | 4.280 ± 0.000 | 155.000 ± 0.000 | 1.267 ± 0.000 | 0.020 ± 0.000 | 606.207 ± 0.000 | 4501.000 ± 0.000 |

## Retrieval Telemetry
> Hits reflect actual recalled traces; cue-only fallbacks are excluded from telemetry.
| mem | requests | hits | hit_rate_at_k | tokens_returned | avg_latency_ms |
|---|---|---|---|---|---|
| episodic | 50 | 50 | 1.000 | 50 | 0.520 |
| relational | 50 | 0 | 0.000 | 50 | 0.113 |
| spatial | 50 | 0 | 0.000 | 800 | 0.066 |

## Gate Telemetry
### Gate ON
| mem | attempts | inserted | aggregated | routed_to_episodic | blocked_new_edges |
|---|---|---|---|---|---|
| episodic | 0 | 0 | 0 | 0 | 0 |
| relational | 0 | 0 | 0 | 0 | 0 |
| spatial | 0 | 0 | 0 | 0 | 0 |
