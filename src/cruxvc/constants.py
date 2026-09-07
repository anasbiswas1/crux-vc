"""Scientific constants shared across notebooks."""
from __future__ import annotations

from typing import Final

ADMINISTRATIVE_CUTOFF: Final = "2013-10-01"
LANDMARK_START: Final = "2005-01-01"
LANDMARK_END: Final = "2010-10-01"

CONFIRMATORY_OUTCOMES: Final = ("F18", "F36", "B+36")
EXPLORATORY_OUTCOMES: Final = ("C+36",)
SECONDARY_OUTCOMES: Final = ("A36", "Broad36")
ALL_OUTCOMES: Final = CONFIRMATORY_OUTCOMES + EXPLORATORY_OUTCOMES + SECONDARY_OUTCOMES

LANDMARK_TYPES: Final = frozenset({"angel", "series-a"})
QUALIFYING_FINANCING_TYPES: Final = frozenset(
    {
        "angel",
        "series-a",
        "series-b",
        "series-c-plus",
        "venture",
        "other",
        "private-equity",
        "crowdfunding",
    }
)
B_PLUS_TYPES: Final = frozenset({"series-b", "series-c-plus"})
C_PLUS_TYPES: Final = frozenset({"series-c-plus"})

PROHIBITED_FEATURES: Final = frozenset(
    {
        "funding_total_usd",
        "funding_rounds",
        "last_funding_at",
        "last_milestone_at",
        "status",
        "snapshot_status",
        "acquisition_price",
        "future_round_count_36",
    }
)

STAGE_DEPENDENCIES: Final = {
    "00": [],
    "01": ["00"],
    "02": ["01"],
    "03": ["02"],
    "04": ["03"],
    "05": ["04"],
    "06": ["05"],
    "07": ["06"],
    "08": ["07"],
    "09": ["08"],
    "10": ["09"],
    "11": ["10"],
    "12": ["11"],
    "13": ["11"],
    "14": ["11"],
    "15": ["11"],
    "16": ["11"],
    "17": ["16"],
    "18": ["11"],
    "19": ["10"],
    "20": ["12", "13", "14", "15", "16", "18"],
}

STAGE_PERMITTED_BLOCKS: Final = {
    "00": ("source_metadata",),
    "01": ("source_data",),
    "02": ("source_data",),
    "03": ("all_landmark_dates_for_label_construction",),
    "04": ("all_landmark_dates_for_feature_construction",),
    "05": ("all_labels_for_structural_count_audit",),
    "06": ("development",),
    "07": ("development", "probability_calibration"),
    "08": ("development", "probability_calibration"),
    "09": ("development", "probability_calibration"),
    "10": ("development", "probability_calibration"),
    "11": ("development", "probability_calibration", "final_test"),
    "12": ("final_test_derived_outputs",),
    "13": ("final_test_derived_outputs",),
    "14": ("development_controls", "final_test_derived_outputs"),
    "15": ("final_test_derived_outputs",),
    "16": ("risk_calibration", "final_test_derived_outputs"),
    "17": ("risk_calibration", "final_test_derived_outputs", "iid_simulation"),
    "18": ("development", "probability_calibration", "final_test_derived_outputs"),
    "19": ("optional_external_data",),
    "20": ("derived_outputs_only",),
}

STAGE_NAMES: Final = {
    "00": "environment_and_manifest",
    "01": "source_schema_license_audit",
    "02": "event_deduplication_and_cutoff",
    "03": "landmark_cohort_and_competing_events",
    "04": "feature_time_ledger",
    "05": "split_freeze_and_structural_gates",
    "06": "blocked_nested_model_selection",
    "07": "probability_calibration",
    "08": "near_optimal_model_registry",
    "09": "development_explanation_pilot",
    "10": "design_power_and_protocol_lock",
    "11": "final_predictions_and_attributions",
    "12": "rq1_outcome_multiverse",
    "13": "rq2_model_family_decomposition",
    "14": "rq3_faithfulness_and_sanity",
    "15": "decision_consequence_audit",
    "16": "conformal_and_selective_baselines",
    "17": "explanation_selective_risk_policy",
    "18": "temporal_and_feature_robustness",
    "19": "phbench_external_stress_optional",
    "20": "inference_figures_and_release",
}
