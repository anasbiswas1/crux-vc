from __future__ import annotations

import numpy as np
import pandas as pd

from cruxvc.calibration import CalibratedModel, PlattCalibrator
from cruxvc.models import fit_model, predict_positive, stable_config_id


def test_logistic_model_and_calibrated_wrapper_are_deterministic():
    rng = np.random.default_rng(7)
    n = 200
    frame = pd.DataFrame(
        {
            "numeric": rng.normal(size=n),
            "category": np.where(np.arange(n) % 2, "a", "b"),
        }
    )
    y = (frame["numeric"] + (frame["category"] == "a").astype(float) > 0.2).astype(int)
    params = {"C": 1.0, "l1_ratio": 0.5}
    model_a = fit_model(frame, y, ["numeric", "category"], "logistic_elastic_net", params, seed=11)
    model_b = fit_model(frame, y, ["numeric", "category"], "logistic_elastic_net", params, seed=11)
    p_a = predict_positive(model_a, frame, ["numeric", "category"])
    p_b = predict_positive(model_b, frame, ["numeric", "category"])
    assert np.allclose(p_a, p_b)
    assert stable_config_id("logistic_elastic_net", params) == stable_config_id("logistic_elastic_net", params)

    calibrator = PlattCalibrator().fit(p_a, y)
    wrapped = CalibratedModel(model_a, calibrator, ("numeric", "category"))
    calibrated = wrapped.predict_proba(frame)
    assert calibrated.shape == (n, 2)
    assert np.allclose(calibrated.sum(axis=1), 1.0)
