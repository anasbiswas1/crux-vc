from __future__ import annotations

import numpy as np
import pandas as pd

from cruxvc.inference import attribution_vectors, rq1_delta_spec, three_way_interpretation
from cruxvc.metrics import partial_area_risk_coverage, rank_biased_overlap, sqrt_jsd_base2


def test_explanation_distances_and_risk_coverage():
    assert np.isclose(sqrt_jsd_base2([1, 0], [1, 0]), 0.0)
    assert np.isclose(sqrt_jsd_base2([1, 0], [0, 1]), 1.0)
    assert np.isclose(sqrt_jsd_base2([0, 0], [0, 0]), 0.0)
    assert np.isclose(sqrt_jsd_base2([0, 0], [1, 0]), 1.0)
    assert np.isclose(rank_biased_overlap(["a", "b"], ["a", "b"]), 1.0)
    assert np.isclose(partial_area_risk_coverage([0.5, 0.6, 0.7, 0.8], [0.4, 0.3, 0.2, 0.1]), 0.25)


def test_rq1_subtracts_within_label_refit_noise():
    rows = []
    vectors = {
        ("A", "r1"): [1.0, 0.0],
        ("A", "r2"): [1.0, 0.0],
        ("B", "r1"): [0.0, 1.0],
        ("B", "r2"): [0.0, 1.0],
    }
    for case_id in [1, 2]:
        for (outcome, refit), vector in vectors.items():
            for group, value in zip(["g1", "g2"], vector):
                rows.append(
                    {
                        "case_id": case_id,
                        "outcome": outcome,
                        "family": "f",
                        "config_id": "cfg",
                        "model_id": f"{outcome}-{refit}",
                        "refit_id": refit,
                        "background_id": "bg",
                        "approximation_seed": 1,
                        "explanation_method": "grouped_permutation_shap",
                        "feature_group": group,
                        "phi_log_odds": value,
                    }
                )
    wide = attribution_vectors(pd.DataFrame(rows))
    summary, cross, within = rq1_delta_spec(wide, "A", "B")
    assert len(cross) == 4
    assert len(within) == 4
    assert np.isclose(summary["D_ab"], 1.0)
    assert np.isclose(summary["D_W"], 0.0)
    assert np.isclose(summary["delta_spec"], 1.0)
    assert np.isclose(summary["NACF"], 1.0)
    assert three_way_interpretation(0.2, (0.12, 0.3), meaningful_effect=0.1, equivalence_half_width=0.1) == "material_sensitivity"


def test_partial_aurc_refuses_unsupported_extrapolation():
    value = partial_area_risk_coverage([0.55, 0.65, 0.75], [0.4, 0.3, 0.2], lower=0.5, upper=0.8)
    assert np.isnan(value)


def test_rq1_honours_frozen_audit_design_weights():
    rows = []
    for case_id, weight, vectors in [
        (1, 9.0, {"A": [1.0, 0.0], "B": [0.0, 1.0]}),
        (2, 1.0, {"A": [1.0, 0.0], "B": [1.0, 0.0]}),
    ]:
        for outcome, vector in vectors.items():
            for refit in ["r1", "r2"]:
                for group, value in zip(["g1", "g2"], vector):
                    rows.append(
                        {
                            "case_id": case_id,
                            "outcome": outcome,
                            "family": "f",
                            "config_id": "cfg",
                            "model_id": f"{outcome}-{refit}",
                            "refit_id": refit,
                            "background_id": "bg",
                            "approximation_seed": 1,
                            "explanation_method": "grouped_permutation_shap",
                            "audit_stratum": "s1" if case_id == 1 else "s2",
                            "design_weight": weight,
                            "audit_inclusion_probability": 1 / weight,
                            "feature_group": group,
                            "phi_log_odds": value,
                        }
                    )
    wide = attribution_vectors(pd.DataFrame(rows))
    summary, _, _ = rq1_delta_spec(wide, "A", "B")
    assert np.isclose(summary["D_ab"], 0.9)
    assert np.isclose(summary["delta_spec"], 0.9)
