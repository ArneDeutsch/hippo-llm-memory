# episodic_cross Summary

> `seeds:1337` `sizes:50` `profile:hard` `replay.samples:0` `store:replay`

> ⚠️ non-informative for uplift: pre_em_norm ≥ 0.98

## Uplift
> single-seed run: CI bands unavailable

| Preset | Size | EM (raw) | <span title="Normalized exact match (lowercase, no punctuation or articles)">EM (norm)</span> | EM | f1 | overlong | format_violation | generated_tokens | input_tokens | latency_ms_delta | latency_ms_mean | memory_hit_rate | refusal_rate | rss_mb | store_size | time_ms_per_100 | total_tokens | ΔEM vs longctx | uplift_vs_longctx_em_norm | ΔF1 vs longctx | ⚠️ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baselines/core | 50 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 132.000 ± 0.000 | 3093.000 ± 0.000 | 0.000 ± 0.000 | 106.928 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1839.926 ± 0.000 | 0.000 ± 0.000 | 165.831 ± 0.000 | 3225.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |  |
| baselines/longctx | 50 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 132.000 ± 0.000 | 3243.000 ± 0.000 | 0.000 ± 0.000 | 99.833 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 2996.488 ± 0.000 | 0.000 ± 0.000 | 147.950 ± 0.000 | 3375.000 ± 0.000 | – | – | – |  |
| baselines/rag | 50 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 132.000 ± 0.000 | 3093.000 ± 0.000 | 0.000 ± 0.000 | 99.938 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 2996.160 ± 0.000 | 0.000 ± 0.000 | 154.992 ± 0.000 | 3225.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |  |
| baselines/span_short | 50 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 132.000 ± 0.000 | 3093.000 ± 0.000 | 0.000 ± 0.000 | 99.376 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 2995.664 ± 0.000 | 0.000 ± 0.000 | 154.124 ± 0.000 | 3225.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |  |
| memory/hei_nw | 50 | 0.980 ± 0.000 | 0.980 ± 0.000 | 0.980 ± 0.000 | 0.993 ± 0.000 | 0.020 ± 0.000 | 0.000 ± 0.000 | 133.000 ± 0.000 | 3093.000 ± 0.000 | 0.382 ± 0.000 | 97.949 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 2973.047 ± 0.000 | 103.000 ± 0.000 | 151.860 ± 0.000 | 3226.000 ± 0.000 | -0.020 ± 0.000 | -0.020 ± 0.000 | -0.007 ± 0.000 |  |

## Retrieval Telemetry

> Hits reflect actual recalled traces; cue-only fallbacks are excluded from telemetry.

| mem | k | batch_size | requests | hits_at_k | hit_rate_at_k | tokens_returned | avg_latency_ms |
|---|---|---|---|---|---|---|---|
| episodic | 0 | 0 | 10 | 10 | 0.200 | 10 | 0.076 |
| relational | 0 | 0 | 0 | 0 | 0.000 | 0 | 0.000 |
| spatial | 0 | 0 | 0 | 0 | 0.000 | 0 | 0.000 |

## Gate Telemetry
### Gate ON
| mem | attempts | accepted | skipped |
|---|---|---|---|
| episodic | 10 | 10 | 0 |
| relational | 0 | 0 | 0 |
| spatial | 0 | 0 | 0 |
