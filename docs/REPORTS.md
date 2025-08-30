# Reports guide

`python scripts/report.py --date $RUN_ID --plots` aggregates metrics from
`runs/$RUN_ID` into `reports/$RUN_ID` and renders tables and plots.

## Key tables

- **Uplift:** shows `pre`, `post`, and `Δ` metrics per preset. When multiple
  seeds are present, a 95% CI column appears.
- **Retrieval:** reflects *actual* recalls only (cue injections do not count as
  hits).
- **Gate ON vs OFF:** compares store size, accepts, and accuracy deltas when
  gates are toggled via `*.gate.enabled`.
- Suites with `pre_em(norm) ≥ 0.98` are flagged as **saturated**.

## Expected layout

```
reports/$RUN_ID/
  episodic/summary.md
  semantic/summary.md
  spatial/summary.md
  index.md
```

## Troubleshooting

If uplift columns are empty, ensure a replay run wrote `post_*` metrics
and that `test_consolidation.py` passed its uplift gate.
