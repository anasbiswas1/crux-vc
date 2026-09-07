# Methods implementation notes

This code distinguishes planned methods from empirical results. Expected row/event/cohort counts are assertions only after the source hashes match. Until the notebooks regenerate them, they remain plan-derived expectations.

The primary explanation is a model-agnostic grouped permutation Shapley approximation over frozen semantic coalitions and a common development background. The conditional variant is explicitly labelled a Gower-kNN conditional-imputation sensitivity, not exact conditional SHAP or proof of on-manifold sampling.

The selective-risk extension implements a finite, frozen candidate-family workflow and conservative bounded-mean tests. It remains an application of established risk-control machinery unless separately supported by a non-inherited theorem or algorithm and independent mathematical checking.

The deployment-time explanation-loss score is trained on development-only operational targets rather than on raw approximation variance alone. Notebook 09 constructs development `L_S`, `L_C`, and `L_F`, freezes the positive faithfulness scale, evaluates the combined `L_S`/`L_F` target by internal cross-fitting, and then fits the frozen score used in 2009/2010.

Notebook 17 reconstructs the finite 2009 operational panel from the design-lock specification. The same-outcome instability loss is conditional on the frozen refit panel; missing panel members contribute distance one. Construct fragility is the average of the two confirmatory matched outcome contrasts. Faithfulness uses frozen conditional-imputation deletion draws and the development-frozen scale. These are operational losses, not certificates of scientific truth.

The exhaustive 300-case explanation multiverse retains the frozen landmark/outcome stratum, inverse inclusion probability, and design weight. RQ1 and RQ2 point estimates are inverse-probability weighted, and their bootstrap procedures resample cases inside those frozen strata before resampling computational explanation replicates.

Conditional-deletion AOPC is evaluated on the probability assigned to the original predicted class. This keeps the operational faithfulness quantity bounded in `[0, 1]`, while grouped attributions and crossed decomposition remain on calibrated log-odds as prespecified.

The uncertainty benchmark separates same-family bootstrap epistemic variance from cross-family hard-prediction disagreement. Partial AURC is reported only when realized test coverage spans the complete preregistered `[0.50, 0.80]` interval; the implementation does not extrapolate missing curve support.
