from __future__ import annotations

import pandas as pd
import pytest

from cruxvc.cohort import build_landmark_cohort
from cruxvc.events import build_event_tables
from cruxvc.features import build_features, feature_time_ledger, validate_feature_matrix
from test_events_and_cohort import source_frames


def test_feature_builder_respects_landmark_time_and_registry():
    events = build_event_tables(*source_frames())
    cohort = build_landmark_cohort(events.funding_events, events.acquisitions).cohort
    products = build_features(cohort, events.companies, events.funding_events, events.investment_edges)

    assert len(products.strict) == len(cohort)
    assert len(products.extended) == len(cohort)
    assert not products.strict["case_id"].duplicated().any()
    assert "future_round_count_36" not in products.strict.columns
    assert "funding_total_usd" not in products.extended.columns

    c1_id = cohort.set_index("company_permalink").loc["/c1", "case_id"]
    c1 = products.strict.set_index("case_id").loc[c1_id]
    # /i1 has exactly one prior event, and the landmark event itself is excluded.
    assert c1["investor_prior_event_count_mean"] == 1.0
    assert c1["landmark_investor_count"] == 1

    c3_id = cohort.set_index("company_permalink").loc["/c3", "case_id"]
    assert products.strict.set_index("case_id").loc[c3_id, "landmark_investor_count"] == 1


def test_prohibited_feature_fails_fast():
    ledger = feature_time_ledger()
    frame = pd.DataFrame(
        {
            "case_id": [1],
            "company_permalink": ["/x"],
            "t0": [pd.Timestamp("2005-01-01")],
            "future_round_count_36": [1],
        }
    )
    with pytest.raises(RuntimeError, match="Prohibited features"):
        validate_feature_matrix(frame, ledger, feature_set="strict")
