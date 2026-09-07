"""Development-only parameter/label randomization and known-signal recovery helpers."""
from __future__ import annotations

import copy
from typing import Any, Sequence

import numpy as np


def randomize_fitted_model_parameters(model: Any, seed: int = 260826) -> Any:
    """Return a deep copy with prediction parameters randomized when supported.

    This is a diagnostic, not a deployable model. Unsupported estimators raise rather
    than silently pretending that a different control was performed.
    """
    randomized = copy.deepcopy(model)
    rng = np.random.default_rng(seed)
    estimator = randomized.named_steps.get("model") if hasattr(randomized, "named_steps") else randomized

    if hasattr(estimator, "coef_"):
        estimator.coef_ = rng.normal(0, np.std(estimator.coef_) or 1, size=estimator.coef_.shape)
        if hasattr(estimator, "intercept_"):
            estimator.intercept_ = rng.normal(0, 1, size=estimator.intercept_.shape)
        return randomized

    if hasattr(estimator, "estimators_") and estimator.__class__.__name__.lower().startswith("randomforest"):
        for tree in estimator.estimators_:
            values = tree.tree_.value
            replacement = rng.gamma(shape=1.0, scale=1.0, size=values.shape)
            tree.tree_.value[:] = replacement
        return randomized

    if hasattr(estimator, "term_scores_"):
        estimator.term_scores_ = [rng.permutation(np.asarray(scores).ravel()).reshape(np.asarray(scores).shape) for scores in estimator.term_scores_]
        return randomized

    raise NotImplementedError(
        f"Parameter-randomization control is not implemented for {type(estimator).__name__}. "
        "Use a supported confirmatory family rather than substituting a different control."
    )


def known_signal_recovery(
    attributions,
    truth,
    *,
    top_k: int = 5,
):
    import pandas as pd

    attr = attributions.copy()
    keys = [c for c in ["case_id", "outcome", "model_id", "refit_id", "background_id", "approximation_seed"] if c in attr.columns]
    attr["rank"] = attr.groupby(keys)["absolute_phi"].rank(method="first", ascending=False)
    inclusion = attr[attr["rank"].le(top_k)].groupby(["outcome", "feature_group"]).size().rename("top_k_count")
    totals = attr.groupby("outcome")["case_id"].nunique().rename("case_count")
    summary = inclusion.reset_index().merge(totals, on="outcome", how="left")
    summary["top_k_inclusion"] = summary["top_k_count"] / summary["case_count"]
    truth_frame = truth[truth["outcome"].ne("shared")].copy()
    shared = truth[truth["role"].eq("shared")]["feature_group"].unique().tolist()
    expanded = []
    for outcome in truth_frame["outcome"].unique():
        for group in shared:
            expanded.append({"outcome": outcome, "feature_group": group, "role": "shared"})
    truth_expanded = pd.concat([truth_frame[["outcome", "feature_group", "role"]], pd.DataFrame(expanded)], ignore_index=True)
    merged = truth_expanded.merge(summary, on=["outcome", "feature_group"], how="left").fillna({"top_k_inclusion": 0.0})
    return merged
