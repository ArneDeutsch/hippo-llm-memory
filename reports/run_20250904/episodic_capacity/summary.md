# episodic_capacity Summary

> `seeds:1337` `sizes:50` `profile:hard` `replay.samples:0` `store:replay`

## Uplift
> single-seed run: CI bands unavailable

| Preset | Size | EM (raw) | <span title="Normalized exact match (lowercase, no punctuation or articles)">EM (norm)</span> | EM | f1 | overlong | format_violation | generated_tokens | input_tokens | latency_ms_delta | latency_ms_mean | memory_hit_rate | refusal_rate | rss_mb | store_size | time_ms_per_100 | total_tokens | ΔEM vs longctx | uplift_vs_longctx_em_norm | ΔF1 vs longctx | ⚠️ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baselines/core | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.273 ± 0.000 | 0.000 ± 0.000 | 0.420 ± 0.000 | 169.000 ± 0.000 | 38366.000 ± 0.000 | 0.000 ± 0.000 | 453.886 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 2995.949 ± 0.000 | 0.000 ± 0.000 | 58.897 ± 0.000 | 38535.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.027 ± 0.000 |  |
| baselines/longctx | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.300 ± 0.000 | 0.000 ± 0.000 | 0.300 ± 0.000 | 163.000 ± 0.000 | 38516.000 ± 0.000 | 0.000 ± 0.000 | 456.193 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 2996.332 ± 0.000 | 0.000 ± 0.000 | 58.976 ± 0.000 | 38679.000 ± 0.000 | – | – | – |  |
| baselines/rag | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.310 ± 0.000 | 0.000 ± 0.000 | 0.380 ± 0.000 | 168.000 ± 0.000 | 38366.000 ± 0.000 | 0.000 ± 0.000 | 452.795 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 2996.273 ± 0.000 | 0.000 ± 0.000 | 58.757 ± 0.000 | 38534.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.010 ± 0.000 |  |
| baselines/span_short | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.307 ± 0.000 | 0.000 ± 0.000 | 0.400 ± 0.000 | 168.000 ± 0.000 | 38366.000 ± 0.000 | 0.000 ± 0.000 | 453.601 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 2995.891 ± 0.000 | 0.000 ± 0.000 | 58.862 ± 0.000 | 38534.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.007 ± 0.000 |  |
| memory/hei_nw | 50 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.300 ± 0.000 | 0.000 ± 0.000 | 0.440 ± 0.000 | 170.000 ± 0.000 | 38366.000 ± 0.000 | 0.371 ± 0.000 | 427.261 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1738.691 ± 0.000 | 103.000 ± 0.000 | 55.441 ± 0.000 | 38536.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |  |

## Retrieval Telemetry

> Hits reflect actual recalled traces; cue-only fallbacks are excluded from telemetry.

| mem | k | batch_size | requests | hits_at_k | hit_rate_at_k | tokens_returned | avg_latency_ms |
|---|---|---|---|---|---|---|---|
| episodic | 0 | 0 | 10 | 10 | 0.200 | 10 | 0.074 |
| relational | 0 | 0 | 0 | 0 | 0.000 | 0 | 0.000 |
| spatial | 0 | 0 | 0 | 0 | 0.000 | 0 | 0.000 |

## Gate Telemetry
### Gate ON
| mem | attempts | accepted | skipped |
|---|---|---|---|
| episodic | 10 | 10 | 0 |
| relational | 0 | 0 | 0 |
| spatial | 0 | 0 | 0 |
