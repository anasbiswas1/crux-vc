# Core data products

| Product | Unit | Purpose |
|---|---|---|
| `funding_events.parquet` | company/date/round type | Deterministically deduplicated funding events |
| `investment_edges.parquet` | company/investor/date/type | Valid unique investor edges |
| `cohort_labels.parquet` | company | Landmark, time block, outcomes, event timing |
| `features_strict.parquet` | company | Event-timed predictors only |
| `features_extended.parquet` | company | Strict plus snapshot-static metadata with caveat |
| `split_ids.parquet` | company | Frozen development/calibration/test assignment |
| `oof_predictions.parquet` | company/model/outcome/fold | Blocked development predictions |
| `calibrated_predictions.parquet` | company/model/outcome/block | Frozen probability outputs |
| `attributions_long.parquet` | company/model/outcome/refit/group | Signed grouped log-odds attributions |

Every produced table has a schema sidecar and is referenced from a completion manifest.
