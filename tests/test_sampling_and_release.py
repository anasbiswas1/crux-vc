from __future__ import annotations

import pandas as pd

from cruxvc.explanations import proportional_stratified_sample
from cruxvc.validation import release_scan


def test_proportional_stratified_sample_returns_exact_requested_size():
    frame = pd.DataFrame(
        {
            "case_id": range(50),
            "landmark_round_type": ["angel"] * 30 + ["series-a"] * 20,
            "joint": ["a"] * 15 + ["b"] * 15 + ["a"] * 10 + ["b"] * 10,
        }
    )
    sample = proportional_stratified_sample(
        frame,
        n=23,
        strata=["landmark_round_type", "joint"],
        seed=9,
    )
    assert len(sample) == 23
    assert sample["case_id"].is_unique
    assert sample.groupby(["landmark_round_type", "joint"]).size().gt(0).all()


def test_release_scan_ignores_method_text_but_flags_identifier_columns(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "docs" / "method.md").write_text("company_permalink is discussed here", encoding="utf-8")
    pd.DataFrame({"anonymous_case": ["x"], "score": [0.2]}).to_csv(
        tmp_path / "results" / "safe.csv", index=False
    )
    assert release_scan(tmp_path) == []
    pd.DataFrame({"company_permalink": ["/company/example"]}).to_csv(
        tmp_path / "results" / "unsafe.csv", index=False
    )
    problems = release_scan(tmp_path)
    assert any("direct identifier columns" in problem for problem in problems)
