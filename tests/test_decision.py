from __future__ import annotations

import numpy as np
import pandas as pd

from cruxvc.decision import subgroup_selection_composition


def test_collapsed_small_cell_selection_rate_is_recomputed_from_counts():
    predictions = pd.DataFrame(
        {
            "case_id": range(6),
            "outcome": "F36",
            "model_id": "m",
            "probability": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        }
    )
    metadata = pd.DataFrame(
        {
            "case_id": range(6),
            "sector": ["a", "a", "b", "b", "b", "c"],
        }
    )
    result = subgroup_selection_composition(
        predictions,
        metadata,
        subgroup_columns=["sector"],
        budgets=[0.5],
        minimum_cell=4,
    )
    collapsed = result[result["level"].eq("__SMALL_CELL_COLLAPSED__")].iloc[0]
    assert collapsed["population_n"] == 6
    assert collapsed["selected_n"] == 3
    assert np.isclose(collapsed["selection_rate"], 0.5)
