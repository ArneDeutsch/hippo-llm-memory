# episodic Summary

> `seeds:1337` `sizes:50` `profile:default` `replay.samples:0` `store:replay`

## Uplift
> single-seed run: CI bands unavailable

| Preset | Size | EM (raw) | <span title="Normalized exact match (lowercase, no punctuation or articles)">EM (norm)</span> | EM | f1 | overlong | format_violation | generated_tokens | input_tokens | latency_ms_mean | refusal_rate | rss_mb | store_size | time_ms_per_100 | total_tokens | ⚠️ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baselines/core | 50 | 0.040 ± 0.000 | 0.040 ± 0.000 | 0.040 ± 0.000 | 0.181 ± 0.000 | 0.520 ± 0.000 | 0.840 ± 0.000 | 294.000 ± 0.000 | 2769.000 ± 0.000 | 163.372 ± 0.000 | 0.000 ± 0.000 | 1692.621 ± 0.000 | 0.000 ± 0.000 | 266.732 ± 0.000 | 3063.000 ± 0.000 |  |

## Retrieval Telemetry

> Hits reflect actual recalled traces; cue-only fallbacks are excluded from telemetry.

| mem | k | batch_size | requests | hits_at_k | hit_rate_at_k | tokens_returned | avg_latency_ms |
|---|---|---|---|---|---|---|---|
| episodic | 0 | 0 | 0 | 0 | 0.000 | 0 | 0.000 |
| relational | 0 | 0 | 0 | 0 | 0.000 | 0 | 0.000 |
| spatial | 0 | 0 | 0 | 0 | 0.000 | 0 | 0.000 |

## Gate Telemetry
### Gate ON
| mem | attempts | accepted | blocked | skipped | inserted | aggregated | routed_to_episodic | blocked_new_edges |
|---|---|---|---|---|---|---|---|---|
| episodic | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| relational | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| spatial | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
