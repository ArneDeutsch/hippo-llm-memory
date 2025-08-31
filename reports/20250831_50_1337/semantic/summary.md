# semantic Summary

> ⚠️ saturated: pre_em_norm ≥ 0.98

## Uplift
> single-seed run: CI bands unavailable

| Preset | Size | EM (raw) | EM (norm) | EM | f1 | overlong | format_violation | generated_tokens | input_tokens | latency_ms_mean | refusal_rate | rss_mb | store_size | time_ms_per_100 | total_tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| memory/sgc_rss | 50 | 0.660 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.660 ± 0.000 | 0.000 ± 0.000 | 0.340 ± 0.000 | 128.000 ± 0.000 | 2450.000 ± 0.000 | 100.624 ± 0.000 | 0.000 ± 0.000 | 1782.191 ± 0.000 | 155.000 ± 0.000 | 195.226 ± 0.000 | 2578.000 ± 0.000 |
| ablate/sgc_rss_no_gate | 50 | 0.520 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.520 ± 0.000 | 0.000 ± 0.000 | 0.480 ± 0.000 | 135.000 ± 0.000 | 2450.000 ± 0.000 | 96.171 ± 0.000 | 0.000 ± 0.000 | 1800.656 ± 0.000 | 155.000 ± 0.000 | 186.074 ± 0.000 | 2585.000 ± 0.000 |
| ablate/longctx_no_retrieval | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | 5.000 ± 0.000 | – | – |
| baselines/rag | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | 5.000 ± 0.000 | – | – |
| baselines/core | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | 5.000 ± 0.000 | – | – |
| baselines/span_short | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | 5.000 ± 0.000 | – | – |
| baselines/longctx | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | 5.000 ± 0.000 | – | – |

## Retrieval Telemetry
> Hits reflect actual recalled traces; cue-only fallbacks are excluded from telemetry.
| mem | requests | hits | hit_rate_at_k | tokens_returned | avg_latency_ms |
|---|---|---|---|---|---|
| episodic | 50 | 50 | 1.000 | 50 | 0.485 |
| relational | 50 | 0 | 0.000 | 50 | 0.109 |
| spatial | 50 | 0 | 0.000 | 800 | 0.063 |

## Gate Telemetry
### Gate ON
| mem | attempts | inserted | aggregated | routed_to_episodic | blocked_new_edges |
|---|---|---|---|---|---|
| episodic | 0 | 0 | 0 | 0 | 0 |
| relational | 0 | 0 | 0 | 0 | 0 |
| spatial | 0 | 0 | 0 | 0 | 0 |
