from __future__ import annotations

import numpy as np
import pandas as pd

from cruxvc.esrc import case_construct_fragility_loss, case_reference_panel_loss


def _rows(case_id: int, outcome: str, panel_id: str, vector: tuple[float, float]):
    return [
        {
            "case_id": case_id,
            "outcome": outcome,
            "panel_id": panel_id,
            "feature_group": group,
            "phi_log_odds": value,
        }
        for group, value in zip(["g1", "g2"], vector)
    ]


def test_operational_instability_and_construct_losses_use_fixed_vectors():
    deployed = pd.DataFrame(
        _rows(1, "F36", "deployment", (1.0, 0.0))
        + _rows(2, "F36", "deployment", (0.5, 0.5))
    )
    panel = pd.DataFrame(
        _rows(1, "F36", "r1", (1.0, 0.0))
        + _rows(1, "F36", "r2", (0.0, 1.0))
        + _rows(2, "F36", "r1", (0.5, 0.5))
        + _rows(2, "F36", "r2", (0.5, 0.5))
    )
    instability = case_reference_panel_loss(deployed, panel).set_index("case_id")
    assert np.isclose(instability.loc[1, "L_S"], 0.5)
    assert np.isclose(instability.loc[2, "L_S"], 0.0)
    assert instability["reference_panel_available"].eq(2).all()

    constructs = pd.DataFrame(
        _rows(1, "F18", "deployment", (1.0, 0.0))
        + _rows(1, "F36", "deployment", (1.0, 0.0))
        + _rows(1, "B+36", "deployment", (0.0, 1.0))
        + _rows(2, "F18", "deployment", (0.5, 0.5))
        + _rows(2, "F36", "deployment", (0.5, 0.5))
        + _rows(2, "B+36", "deployment", (0.5, 0.5))
    )
    fragility = case_construct_fragility_loss(constructs).set_index("case_id")
    assert np.isclose(fragility.loc[1, "L_C"], 0.5)
    assert np.isclose(fragility.loc[2, "L_C"], 0.0)
    assert fragility["construct_contrasts_available"].eq(2).all()


def test_missing_reference_member_receives_worst_distance():
    deployed = pd.DataFrame(_rows(1, "F36", "deployment", (1.0, 0.0)) + _rows(2, "F36", "deployment", (1.0, 0.0)))
    panel = pd.DataFrame(_rows(1, "F36", "r1", (1.0, 0.0)) + _rows(2, "F36", "r1", (1.0, 0.0)) + _rows(1, "F36", "r2", (1.0, 0.0)))
    losses = case_reference_panel_loss(deployed, panel).set_index("case_id")
    assert np.isclose(losses.loc[2, "L_S"], 0.5)
    assert losses.loc[2, "reference_panel_available"] == 1


def test_entirely_failed_reference_model_is_counted_as_worst_loss():
    deployed = pd.DataFrame(_rows(1, "F36", "deployment", (1.0, 0.0)))
    panel = pd.DataFrame(_rows(1, "F36", "r1", (1.0, 0.0)))
    losses = case_reference_panel_loss(
        deployed,
        panel,
        expected_panel_ids=["r1", "r2"],
    ).set_index("case_id")
    assert np.isclose(losses.loc[1, "L_S"], 0.5)
    assert losses.loc[1, "reference_panel_expected"] == 2
    assert losses.loc[1, "reference_panel_available"] == 1
