from __future__ import annotations

import pandas as pd

from cruxvc.cohort import build_landmark_cohort
from cruxvc.events import build_event_tables, normalize_round_type


def source_frames():
    companies = pd.DataFrame(
        {
            "permalink": ["/c1", "/c2", "/c3", "/c4", "/c5", "/c6", "/c8", "/prior"],
            "founded_at": ["2003-01-01", "2004-01-01", None, "2005-01-01", None, None, "2007-01-01", "2000-01-01"],
            "category_code": ["software", "web", "mobile", "web", "software", "other", "web", "finance"],
            "country_code": ["USA"] * 8,
            "region": ["CA", "NY", "MA", "TX", "CA", "WA", "CA", "NY"],
        }
    )
    rounds = pd.DataFrame(
        [
            # Prior event used only for historical investor/market features.
            ["/prior", "2004-01-01", "seed", 50.0],
            # c1: exact duplicate plus a conflicting amount for the same event.
            ["/c1", "2005-01-01", "angel", 100.0],
            ["/c1", "2005-01-01", "angel", 100.0],
            ["/c1", "2005-01-01", "angel", 150.0],
            ["/c1", "2006-07-01", "series b", 500.0],  # exactly +18 months
            # c2: same event family, exact +18 calendar months.
            ["/c2", "2006-01-31", "series a", 200.0],
            ["/c2", "2007-07-31", "venture", 600.0],
            # c3: exact +36 months at the administrative cutoff.
            ["/c3", "2010-10-01", "angel", 300.0],
            ["/c3", "2013-10-01", "series c", 900.0],
            # c4 is removed because acquisition precedes t0.
            ["/c4", "2007-01-01", "angel", 100.0],
            # c5 is removed because two distinct types occur at the earliest date.
            ["/c5", "2007-01-01", "angel", 100.0],
            ["/c5", "2007-01-01", "series a", 100.0],
            # c6 is removed because seed is not an eligible landmark.
            ["/c6", "2007-01-01", "seed", 100.0],
            # c8 has a same-day acquisition, which is not a post-landmark A36 event.
            ["/c8", "2008-01-01", "angel", 100.0],
            # Beyond the cutoff must be removed.
            ["/c8", "2014-01-01", "series b", 1000.0],
        ],
        columns=["company_permalink", "funded_at", "funding_round_type", "raised_amount_usd"],
    )
    investments = pd.DataFrame(
        [
            ["/prior", "/i1", "2004-01-01", "seed"],
            ["/c1", "/i1", "2005-01-01", "angel"],
            ["/c1", "/i1", "2005-01-01", "angel"],  # exact duplicate
            ["/c2", "/i1", "2006-01-31", "series a"],
            ["/c2", "/i2", "2006-01-31", "series a"],
            ["/c3", "/i3", "2010-10-01", "angel"],
            ["/c8", "/i4", "2008-01-01", "angel"],
            [None, "/bad", "2008-01-01", "angel"],
        ],
        columns=["company_permalink", "investor_permalink", "funded_at", "funding_round_type"],
    )
    acquisitions = pd.DataFrame(
        [
            ["/c1", "/buyer1", "2008-01-01"],  # exactly +36 months
            ["/c4", "/buyer2", "2006-12-31"],  # before t0
            ["/c8", "/buyer3", "2008-01-01"],  # same day, not subsequent
        ],
        columns=["company_permalink", "acquirer_permalink", "acquired_at"],
    )
    return companies, rounds, investments, acquisitions


def test_round_normalization():
    assert normalize_round_type("Series C") == "series-c-plus"
    assert normalize_round_type("post_ipo_equity") == "post-ipo"
    assert normalize_round_type("SERIES-A") == "series-a"


def test_deterministic_event_collapse_and_horizon_labels():
    result = build_event_tables(*source_frames(), administrative_cutoff="2013-10-01")
    c1_landmark = result.funding_events.loc[
        result.funding_events["company_permalink"].eq("/c1")
        & result.funding_events["funded_at"].eq(pd.Timestamp("2005-01-01"))
        & result.funding_events["round_type"].eq("angel")
    ].iloc[0]
    assert c1_landmark["source_row_count"] == 2  # exact duplicate removed first
    assert c1_landmark["amount_max_usd"] == 150.0
    assert c1_landmark["amount_sum_usd"] == 250.0
    assert bool(c1_landmark["amount_conflict"])
    assert result.audit["exact_round_duplicates"] == 1
    assert result.audit["exact_investment_duplicates"] == 1
    assert result.audit["malformed_investment_rows_missing_company"] == 1
    assert result.funding_events["funded_at"].max() == pd.Timestamp("2013-10-01")

    built = build_landmark_cohort(
        result.funding_events,
        result.acquisitions,
        landmark_start="2005-01-01",
        landmark_end="2010-10-01",
        administrative_cutoff="2013-10-01",
    )
    cohort = built.cohort.set_index("company_permalink")
    assert set(cohort.index) == {"/c1", "/c2", "/c3", "/c8"}

    assert cohort.loc["/c1", ["F18", "F36", "B+36", "C+36", "A36"]].tolist() == [1, 1, 1, 0, 1]
    assert cohort.loc["/c2", ["F18", "F36", "B+36", "C+36", "A36"]].tolist() == [1, 1, 0, 0, 0]
    assert cohort.loc["/c3", ["F18", "F36", "B+36", "C+36", "A36"]].tolist() == [0, 1, 1, 1, 0]
    assert cohort.loc["/c8", ["F18", "F36", "B+36", "C+36", "A36"]].tolist() == [0, 0, 0, 0, 0]
    assert built.audit["nested_violation_Bplus_not_F36"] == 0
    assert built.audit["nested_violation_Cplus_not_Bplus"] == 0
    assert cohort.loc["/c3", "F36_event_at"] == pd.Timestamp("2013-10-01")
