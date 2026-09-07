# Notebook workflow

Run the notebooks in numerical order. Each notebook embeds a machine-readable contract copied from `config/notebook_contracts.yaml`; downstream stages refuse missing completion manifests or incompatible protocol state.

| Stage | Notebook | Permitted data blocks | Principal outputs |
|---:|---|---|---|
| 00 | `00_environment_and_manifest.ipynb` — environment and manifest | `source_metadata` | `protocol/source_manifest.json`, `protocol/seed_registry.csv`, `protocol/runtime_manifest.json` + 1 more |
| 01 | `01_source_schema_license_audit.ipynb` — source schema license audit | `source_data` | `results/audits/01_source_schema_audit.json`, `results/audits/01_date_quality_audit.csv`, `results/audits/01_missingness.csv` + 1 more |
| 02 | `02_event_deduplication_and_cutoff.ipynb` — event deduplication and cutoff | `source_data` | `data/interim/companies.parquet`, `data/interim/funding_events.parquet`, `data/interim/investment_edges.parquet` + 2 more |
| 03 | `03_landmark_cohort_and_competing_events.ipynb` — landmark cohort and competing events | `all_landmark_dates_for_label_construction` | `data/processed/cohort_labels.parquet`, `results/audits/03_cohort_flow.csv`, `results/audits/03_label_overlap.csv` + 2 more |
| 04 | `04_feature_time_ledger.ipynb` — feature time ledger | `all_landmark_dates_for_feature_construction` | `data/processed/features_strict.parquet`, `data/processed/features_extended.parquet`, `protocol/feature_time_ledger.csv` + 1 more |
| 05 | `05_split_freeze_and_structural_gates.ipynb` — split freeze and structural gates | `all_labels_for_structural_count_audit` | `protocol/split_ids.parquet`, `protocol/development_folds.parquet`, `results/audits/05_block_outcome_counts.csv` + 4 more |
| 06 | `06_blocked_nested_model_selection.ipynb` — blocked nested model selection | `development` | `results/predictions/development_oof_predictions.parquet`, `results/models/development_model_registry.parquet`, `results/models/near_optimal_registry.parquet` + 2 more |
| 07 | `07_probability_calibration.ipynb` — probability calibration | `development`, `probability_calibration` | `results/models/calibration_registry.parquet`, `results/predictions/probability_calibration_2008.parquet`, `results/models/calibrated/*.joblib` |
| 08 | `08_near_optimal_model_registry.ipynb` — near optimal model registry | `development`, `probability_calibration` | `results/models/matched_reference_configs.parquet`, `protocol/final_task_registry.parquet`, `protocol/final_task_chunk_plan.csv` + 1 more |
| 09 | `09_development_explanation_pilot.ipynb` — development explanation pilot | `development`, `probability_calibration` | `results/attributions/development_pilot_attributions.parquet`, `results/attributions/development_pilot_conditional_imputation.parquet`, `results/audits/09_approximation_repeat_diagnostics.csv` + 8 more |
| 10 | `10_design_power_and_protocol_lock.ipynb` — design power and protocol lock | `development`, `probability_calibration` | `results/audits/10_rq1_power_operating_characteristics.csv`, `results/selective/frozen_candidate_policies.parquet`, `protocol/frozen_explanation_losses.json` + 2 more |
| 11 | `11_final_predictions_and_attributions.ipynb` — final predictions and attributions | `development`, `probability_calibration`, `final_test` | `results/stage11_chunks/*.parquet`, `results/predictions/final_predictions.parquet`, `results/attributions/final_attributions_long.parquet` |
| 12 | `12_rq1_outcome_multiverse.ipynb` — rq1 outcome multiverse | `final_test_derived_outputs` | `results/inference/rq1_specification_effects.csv`, `results/inference/rq1_cross_spec_distances.parquet`, `results/inference/rq1_within_refit_distances.parquet` + 1 more |
| 13 | `13_rq2_model_family_decomposition.ipynb` — rq2 model family decomposition | `final_test_derived_outputs` | `results/inference/rq2_model_pair_distances.parquet`, `results/inference/rq2_family_effects.csv`, `results/inference/rq2_bootstrap_draws.parquet` + 2 more |
| 14 | `14_rq3_faithfulness_and_sanity.ipynb` — rq3 faithfulness and sanity | `development_controls`, `final_test_derived_outputs` | `results/controls/rq3_conditional_deletion_curves.parquet`, `results/controls/rq3_faithfulness_case_summary.parquet`, `results/controls/rq3_randomization_controls.csv` + 4 more |
| 15 | `15_decision_consequence_audit.ipynb` — decision consequence audit | `final_test_derived_outputs` | `results/decision/topk_portfolio_overlap.csv`, `results/decision/screening_performance.csv`, `results/decision/expected_utility_grid.csv` + 2 more |
| 16 | `16_conformal_and_selective_baselines.ipynb` — conformal and selective baselines | `risk_calibration`, `final_test_derived_outputs` | `results/selective/risk_calibration_gate_scores_2009.parquet`, `results/selective/test_gate_scores_common_population.parquet`, `results/selective/test_losses_common_population.parquet` + 4 more |
| 17 | `17_explanation_selective_risk_policy.ipynb` — explanation selective risk policy | `risk_calibration`, `final_test_derived_outputs`, `iid_simulation` | `results/selective/risk_calibration_explanation_losses_2009.parquet`, `results/attributions/esrc_risk_calibration_attributions_2009.parquet`, `results/controls/esrc_risk_calibration_faithfulness_2009.parquet` + 5 more |
| 18 | `18_temporal_and_feature_robustness.ipynb` — temporal and feature robustness | `development`, `probability_calibration`, `final_test_derived_outputs` | `results/inference/strict_extended_feature_robustness.csv`, `results/predictions/strict_extended_test_predictions.parquet`, `results/inference/event_count_matched_scarcity_envelope.csv` + 2 more |
| 19 | `19_phbench_external_stress_optional.ipynb` — phbench external stress optional | `optional_external_data` | `results/audits/19_phbench_status.json`, `results/inference/phbench_clean_to_leaky_curve.csv`, `results/audits/19_kaggle_snapshot_status.json` + 1 more |
| 20 | `20_inference_figures_and_release.ipynb` — inference figures and release | `derived_outputs_only` | `figures/*.png`, `results/release/tables/*`, `results/release/release_table_inventory.csv` + 2 more |

## Hard execution boundaries

- Stages `00`–`05` reconstruct and freeze data, labels, features, splits, backgrounds, audit cases, and the Phase-0 lock.
- Stages `06`–`10` use only development/2008 data for model selection, calibration, explanation diagnostics, power analysis, and the final design lock.
- Stage `11` is the only raw final-test execution stage; it verifies both locks, logs access, and supports immutable task chunks.
- Stages `12`–`18` consume frozen outputs and the independently held 2009 risk-calibration block according to their contracts.
- Stage `19` is optional and fails gracefully when external data are unavailable.
- Stage `20` packages only derived, identifier-scanned release artifacts.

## Stage 11 chunking

`CRUX_N_CHUNKS` fixes the task partition, `CRUX_CHUNK_INDEX` selects one chunk, and `CRUX_MERGE_ONLY=1` performs the canonical merge only after every expected chunk manifest exists.

## Stage 17 interpretation

Notebook `17` builds the operational 2009 losses `L_S`, `L_C`, `L_F`, and `L_Y` using the development-frozen scale, matched deployment models, a finite frozen refit panel, fixed background/seed/orderings, and fixed conditional-perturbation draws. Missing or failed explanations receive loss 1. An empirical certified row is not released as a theorem-backed guarantee unless `CRUX_ESRC_THEOREM_CHECKED=1` is explicitly set after independent mathematical review; the 2010 chronological track never receives a distribution-free guarantee.

## Completion manifests

Every stage writes `results/manifests/<stage>_<name>.json`, including input/output hashes, configuration fingerprint, runtime versions, permitted blocks, and contract notes. Existing manifests cannot be silently overwritten with different output hashes.
