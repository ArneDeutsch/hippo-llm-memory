# episodic Summary

## Uplift
> single-seed run: CI bands unavailable

| Preset | Size | EM (raw) | EM (norm) | EM | f1 | overlong | format_violation | generated_tokens | input_tokens | latency_ms_mean | refusal_rate | rss_mb | store_size | time_ms_per_100 | total_tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baselines/core | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | 5.000 ± 0.000 | – | – |
| baselines/longctx | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | 5.000 ± 0.000 | – | – |
| baselines/rag | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | 5.000 ± 0.000 | – | – |
| baselines/span_short | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | 0.000 ± 0.000 | – | 5.000 ± 0.000 | – | – |
| memory/hei_nw | 50 | 0.120 ± 0.000 | 0.620 ± 0.000 | 0.620 ± 0.000 | 0.332 ± 0.000 | 0.380 ± 0.000 | 0.620 ± 0.000 | 223.000 ± 0.000 | 1970.000 ± 0.000 | 137.411 ± 0.000 | 0.000 ± 0.000 | 1715.285 ± 0.000 | 155.000 ± 0.000 | 313.361 ± 0.000 | 2193.000 ± 0.000 |

## Retrieval Telemetry
> Hits reflect actual recalled traces; cue-only fallbacks are excluded from telemetry.
| mem | requests | hits | hit_rate_at_k | tokens_returned | avg_latency_ms |
|---|---|---|---|---|---|
| episodic | 50 | 50 | 1.000 | 50 | 0.590 |
| relational | 50 | 0 | 0.000 | 50 | 0.114 |
| spatial | 50 | 0 | 0.000 | 800 | 0.080 |

## Gate Telemetry
### Gate ON
| mem | attempts | inserted | aggregated | routed_to_episodic | blocked_new_edges |
|---|---|---|---|---|---|
| episodic | 0 | 0 | 0 | 0 | 0 |
| relational | 0 | 0 | 0 | 0 | 0 |
| spatial | 0 | 0 | 0 | 0 | 0 |
