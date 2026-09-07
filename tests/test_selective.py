from __future__ import annotations

import numpy as np
import pandas as pd

from cruxvc.selective import (
    calibrate_finite_policy_family,
    finite_sample_quantile,
    split_conformal_binary_sets,
)


def test_split_conformal_set_construction():
    p_cal = np.array([0.05, 0.10, 0.20, 0.80, 0.90, 0.95])
    y_cal = np.array([0, 0, 0, 1, 1, 1])
    output = split_conformal_binary_sets(p_cal, y_cal, [0.05, 0.5, 0.95], alpha=0.2)
    assert finite_sample_quantile([0.1, 0.2, 0.3], 0.1) == 0.3
    assert output.loc[0, "include_0"]
    assert output.loc[2, "include_1"]
    assert set(output["set_size"]).issubset({0, 1, 2})


def test_frozen_policy_family_can_certify_only_good_policy():
    n = 400
    score = np.linspace(0, 1, n)
    calibration = pd.DataFrame(
        {
            "score": score,
            "L_S": np.full(n, 0.02),
            "L_C": np.full(n, 0.03),
            "L_F": np.full(n, 0.05),
            "L_Y": np.full(n, 0.01),
        }
    )
    candidates = pd.DataFrame(
        [
            {"candidate_id": "good", "score_name": "score", "threshold": 0.90, "lower_is_better": True},
            {"candidate_id": "too_small", "score_name": "score", "threshold": 0.10, "lower_is_better": True},
        ]
    )
    results, selected = calibrate_finite_policy_family(
        calibration,
        candidates,
        score_columns=["score"],
        loss_budgets={"L_S": 0.2, "L_C": 0.25, "L_F": 0.5, "L_Y": 0.2},
        q_min=0.5,
        familywise_delta=0.05,
        minimum_accepted=100,
    )
    good = results.set_index("candidate_id").loc["good"]
    small = results.set_index("candidate_id").loc["too_small"]
    assert bool(good["certified"])
    assert not bool(small["certified"])
    assert selected is not None and selected["candidate_id"] == "good"


def test_cross_fitted_conformal_outputs_are_deterministic_and_aligned():
    from cruxvc.selective import (
        cross_fitted_mondrian_binary_sets,
        cross_fitted_split_conformal_binary_sets,
    )

    rng = np.random.default_rng(41)
    y = np.tile([0, 1], 100)
    p = np.clip(0.15 + 0.70 * y + rng.normal(0, 0.08, len(y)), 0.01, 0.99)
    first = cross_fitted_split_conformal_binary_sets(p, y, folds=5, seed=7)
    second = cross_fitted_split_conformal_binary_sets(p, y, folds=5, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == len(y)
    assert first["crossfit_fold"].nunique() == 5

    mondrian = cross_fitted_mondrian_binary_sets(
        p,
        y,
        folds=5,
        seed=7,
        minimum_class_count=20,
    )
    assert len(mondrian) == len(y)
    assert set(mondrian["set_size"]).issubset({0, 1, 2})
