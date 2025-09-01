# semantic Summary

> `seeds:1337,2025` `sizes:50` `profile:default` `replay.samples:0` `store:stub`

> ⚠️ saturated: pre_em_norm ≥ 0.98

## Uplift
| Preset | Size | EM (raw) | EM (norm) | EM | f1 | overlong | format_violation | generated_tokens | input_tokens | latency_ms_mean | refusal_rate | rss_mb | store_size | time_ms_per_100 | total_tokens | ⚠️ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| memory/sgc_rss | 50 | 0.950 ± 0.059 | 0.980 ± 0.039 | 0.980 ± 0.039 | 0.950 ± 0.059 | 0.000 ± 0.000 | 0.040 ± 0.039 | 111.000 ± 0.000 | 3150.000 ± 0.000 | 85.030 ± 1.197 | 0.000 ± 0.000 | 1758.363 ± 78.101 | 0.000 ± 0.000 | 130.424 ± 1.837 | 3261.000 ± 0.000 | ⚠️ SaturationSuspect, GateNoOp |
| ablate/sgc_rss_no_gate | 50 | 0.910 ± 0.059 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.910 ± 0.059 | 0.000 ± 0.000 | 0.090 ± 0.059 | 114.500 ± 2.940 | 3150.000 ± 0.000 | 86.406 ± 2.256 | 0.000 ± 0.000 | 1744.021 ± 78.917 | 0.000 ± 0.000 | 132.399 ± 3.330 | 3264.500 ± 2.940 | ⚠️ SaturationSuspect |
| ablate/longctx_no_retrieval | 50 | – | – | – | – | – | – | 110.000 ± 0.000 | 3300.000 ± 0.000 | 96.889 ± 0.921 | – | 1764.451 ± 104.780 | 0.000 ± 0.000 | 142.117 ± 1.345 | 3410.000 ± 0.000 |  |
| baselines/rag | 50 | – | – | – | – | – | – | 113.000 ± 1.960 | 3150.000 ± 0.000 | 74.785 ± 1.817 | – | 2706.982 ± 0.088 | 0.000 ± 0.000 | 114.642 ± 2.717 | 3263.000 ± 1.960 |  |
| baselines/core | 50 | – | – | – | – | – | – | 112.500 ± 2.940 | 3150.000 ± 0.000 | 74.223 ± 0.673 | – | 2712.408 ± 29.863 | 0.000 ± 0.000 | 113.796 ± 0.929 | 3262.500 ± 2.940 |  |
| baselines/span_short | 50 | – | – | – | – | – | – | 114.000 ± 5.880 | 3150.000 ± 0.000 | 76.004 ± 1.774 | – | 2698.307 ± 0.302 | 0.000 ± 0.000 | 116.474 ± 2.505 | 3264.000 ± 5.880 |  |
| baselines/longctx | 50 | – | – | – | – | – | – | 111.000 ± 1.960 | 3300.000 ± 0.000 | 91.066 ± 5.193 | – | 2735.037 ± 53.000 | 0.000 ± 0.000 | 133.540 ± 7.697 | 3411.000 ± 1.960 |  |

### Warnings
- memory/sgc_rss 50: SaturationSuspect, GateNoOp
- ablate/sgc_rss_no_gate 50: SaturationSuspect

## Retrieval Telemetry
> Hits reflect actual recalled traces; cue-only fallbacks are excluded from telemetry.
| mem | k | batch_size | requests | hits_at_k | hit_rate_at_k | tokens_returned | avg_latency_ms |
|---|---|---|---|---|---|---|---|
| episodic | 0 | 0 | 0 | 0 | 0.000 | 0 | 0.000 |
| relational | 0 | 0 | 14 | 0 | 0.000 | 14 | 0.045 |
| spatial | 0 | 0 | 0 | 0 | 0.000 | 0 | 0.000 |

## Gate Telemetry
### Gate ON
| mem | attempts | inserted | aggregated | routed_to_episodic | blocked_new_edges |
|---|---|---|---|---|---|
| episodic | 0 | 0 | 0 | 0 | 0 |
| relational | 0 | 0 | 0 | 0 | 0 |
| spatial | 0 | 0 | 0 | 0 | 0 |
