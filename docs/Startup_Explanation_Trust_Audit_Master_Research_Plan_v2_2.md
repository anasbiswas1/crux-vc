# When “Startup Success” Changes Meaning

## Master Research Plan v2.2: A Construct–Rashomon Audit and Selective-Explanation Venture-Screening Framework

**Document status:** Redesigned, preliminary-literature-verified, data-feasibility-audited master plan; executable core after Phase-0 regeneration and protocol lock  
**Evidence-search cut-off:** 25 August 2026  
**Correction date:** 26 August 2026  
**Primary intended venue:** Decision Support Systems  
**Secondary venue:** Expert Systems with Applications  
**Methodological stretch venues:** Machine Learning or Data Mining and Knowledge Discovery only if the explanation-risk-control extension yields a general method, theorem, and multi-domain validation  
**Recommended short name:** CRUX-VC — Construct–Rashomon Uncertainty Examination for Venture Capital

---

## 1. Executive decision

The project should proceed, but not under the original novelty or conformal-certification claims.

The literature already establishes that:

1. startup “success” has inconsistent operational definitions;
2. multiple startup outcomes have been predicted and explained in the same pipeline;
3. equally predictive models can provide incompatible feature attributions;
4. faithfulness metrics can disagree;
5. selective abstention and epistemic uncertainty can identify less stable explanations; and
6. conformal prediction and XAI have already been combined in several ways.

The original claims of being the first study to vary a startup-success definition or the first to connect conformal/selective regions with explanation reliability must be removed. The broader claim about outcome-definition effects on explanations remains **open**: the preliminary scan did not find a prior matched design, but the previously cited Frontiers paper does not support the contrary conclusion and no cross-domain “first” claim may be made until the archived search is complete.

The stronger paper is a controlled construct-validity study:

> Quantify how explanatory claims and venture-screening decisions change across defensible, horizon-aligned startup-outcome specifications, and separate that variation from model-family, resampling, hyperparameter, and explainer uncertainty.

The high-confidence contribution is a **startup outcome-specification multiverse plus crossed explanation-uncertainty decomposition and decision-consequence audit**. The optional extension instantiates existing selective-risk-control tools for prespecified explanation losses. It is an applied extension unless this project proves a genuinely new theorem or algorithm. Ordinary conformal singleton gating remains only a baseline.

### 1.1 Revised quality assessment

| Design | Novelty | Scientific credibility | Execution feasibility | Likely ceiling |
|---|---:|---:|---:|---|
| Original plan | 4/10 | 4/10 | 3/10 | Weak applied paper |
| Revised core: label multiverse + variance decomposition | 8/10 | 9/10 | 8.5/10 | Strong DSS/ESWA paper |
| Core + decision-consequence analysis | 8.5/10 | 9/10 | 8/10 | Strong Q1 decision-support paper |
| Core + valid LTT/SCRC explanation-risk application | 8.5–9/10 | Conditional on exact losses and proof conditions | 6.5/10 | Strong DSS/ESWA paper |
| Core + genuinely new selective-risk theorem/algorithm | 9–9.5/10 | Conditional on proof | 5.5/10 | Method journal only with multi-domain validation |

No plan can honestly be called perfect before its data, power, and positive-control gates pass. This plan is designed so that failure at any gate produces a scientifically defensible fallback rather than an invalid claim.

---

## 2. Scientific reframing

### 2.1 The construct problem

“Startup success” is not directly observed. Funding progression, Series B/C attainment, acquisition, IPO, survival, employment, valuation, patents, and revenue are different observable events, representing different theoretical understandings of progress or investor return.

[Jacobs and Wallach’s measurement framework](https://dl.acm.org/doi/10.1145/3442188.3445901) distinguishes an unobservable construct from the measurements used to operationalize it. The startup-prediction literature often collapses that distinction by naming one convenient database event “success.” The 2025 [SAISE systematic-review preprint](https://arxiv.org/abs/2508.05491), covering 57 empirical studies, reports inconsistent success definitions as a foundational weakness of the field. This preprint motivates the archived search but does not by itself establish the field-wide gap.

Consequently, an attribution difference between acquisition and follow-on funding is not automatically an XAI failure. It may reflect a real difference between two estimands. The correct question is whether a published general claim such as “these are the drivers of startup success” remains defensible across clearly specified outcome constructs.

### 2.2 Multiverse logic

Multiverse and specification-curve methods evaluate how conclusions change across reasonable analytical specifications rather than reporting one convenient analysis. Relevant methodological precedents include [Steegen et al.](https://journals.sagepub.com/doi/abs/10.1177/1745691616658637) and [Simonsohn, Simmons, and Nelson](https://www.nature.com/articles/s41562-020-0912-z).

This project adapts that logic to predictive explanations while controlling the elements that usually change unnoticed:

- one decision landmark;
- one eligible risk set;
- one information horizon;
- one frozen feature matrix;
- one temporal split;
- one attribution target and background distribution;
- matched model configurations and random seeds; and
- the same local audit cases.

### 2.3 Rashomon logic

The Rashomon effect means that several models can perform similarly while relying on different features. [Model Class Reliance](https://jmlr.org/papers/v20/18-760.html), [Variable Importance Clouds](https://arxiv.org/abs/1901.03209), and [consensus feature attributions](https://jmlr.org/papers/v24/23-0149.html) already formalize this issue. The proposed study does not claim to discover model multiplicity. It uses that literature to estimate whether outcome operationalization is a larger or smaller source of explanation variation than model multiplicity.

### 2.4 Decision-support logic

The study is not a causal analysis of what makes startups succeed. SHAP values, permutation importance, and the proposed robustness metrics describe fitted predictive systems. They do not identify interventions.

The operational decision is:

> Given a fixed analyst-review budget at an early financing landmark, which startups are ranked for further due diligence, which evidence patterns are stable enough to report, and which cases should be escalated for review?

That decision framing determines the evaluation metrics: precision and lift at fixed review budgets, portfolio overlap, rank reversals, subgroup composition, analyst workload, calibration, and selective risk.

---

## 3. Literature-review protocol and evidence status

### 3.1 Sources actually verified for this revision

The present review verified primary or official records available through:

- journal publisher pages and DOI records;
- JMLR and PMLR;
- NeurIPS proceedings and OpenReview;
- ACM and IEEE records;
- arXiv primary records and full-text pages;
- official GitHub repositories and dataset documentation; and
- backward and forward citation tracing from the closest papers.

The evidence-search cut-off is 25 August 2026. Peer-reviewed articles and conference papers are distinguished from preprints; publisher-record-only entries are not used as decisive novelty evidence until their full text is archived and checked. During the 26 August correction audit, DOI 10.3389/fdgth.2026.1851084 was confirmed to be an unrelated postoperative-analgesia XAI study, not an outcome-definition experiment; it has therefore been removed from the novelty evidence.

Before manuscript submission, the research team must reproduce and archive the search in Scopus, Web of Science, IEEE Xplore, ACM Digital Library, ScienceDirect, SpringerLink, PMLR/OpenReview, and arXiv. Search exports, deduplication rules, inclusion decisions, and a dated screening log must be released as supplementary material. Until that database search is complete, manuscript wording must use “no study found in our documented search,” never “nobody,” “never,” or “first in any domain.”

### 3.2 Search concepts

Use Boolean combinations of:

1. startup OR venture OR entrepreneurial finance;
2. success prediction OR funding OR next round OR Series B OR Series C OR acquisition OR IPO OR exit OR survival;
3. SHAP OR feature attribution OR explainable AI OR counterfactual OR variable importance;
4. label definition OR outcome operationalization OR measurement OR construct validity OR multiverse OR specification curve;
5. explanation stability OR faithfulness OR robustness OR Rashomon OR model multiplicity;
6. uncertainty OR calibration OR conformal OR selective prediction OR abstention OR reject option; and
7. risk control OR learn-then-test OR selective risk.

### 3.3 Inclusion rules

Include studies that satisfy at least one of the following:

- predict a startup, venture, financing, survival, acquisition, IPO, patent, valuation, or exit outcome;
- compare startup outcome definitions or stages;
- assess explanation disagreement, robustness, faithfulness, or uncertainty;
- connect explanations with abstention, calibration, conformal methods, or risk control; or
- address construct validity, target-variable construction, multiverse analysis, or proxy-label risks.

Exclude opinion pieces without methods, unverifiable vendor claims, student projects without sufficient methodological detail, duplicate preprint/journal versions when a peer-reviewed record exists, and studies whose “startup” outcome is unrelated to firm progression or investor screening.

---

## 4. Verified literature map

### 4.1 Startup prediction, outcome construction, and explainability

| Study | Data and task | What is already done | Remaining gap relevant here |
|---|---|---|---|
| [Arroyo et al., 2019](https://doi.org/10.1109/ACCESS.2019.2938659) | More than 120,000 Crunchbase firms; predicts closure, funding, acquisition, or IPO over a three-year simulation window | Multiple startup events, time-aware decision support, feature importance | No controlled construct multiverse or attribution-uncertainty decomposition |
| [Ross, Das, Sciro, and Raza, 2021 — CapitalVX](https://www.sciencedirect.com/science/article/pii/S2405918821000040) | Crunchbase; exit/failure/private and follow-on-funding tasks | Separates exit from follow-on funding | No common-protocol explanation-stability or faithfulness audit |
| [Żbikowski and Antosiuk, 2021](https://www.sciencedirect.com/science/article/pii/S0306457321000595) | Large Crunchbase success prediction with bias/leakage controls | Recognizes target-definition and look-ahead problems | Does not vary the outcome and quantify explanatory consequences |
| [Corea et al., 2021](https://www.sciencedirect.com/science/article/pii/S2666827021000311) | Data-driven early-stage investment framework | Questions whether attracting investment itself constitutes success | No label multiverse or explanation reliability |
| [Kim, Kim, and Geum, 2023](https://doi.org/10.1016/j.techfore.2023.122614) | 218,207 Crunchbase companies; multiple ML models and feature importance | Large-scale interpretable startup prediction | One selected construct and no explanation robustness |
| [Gavrilenko et al., 2023](https://arxiv.org/abs/2312.06236) | Crunchbase, Google, and Twitter; funding within a fixed horizon | Rich digital features and horizon-specific financing prediction | One financing endpoint |
| [Razaghzadeh Bidgoli et al., 2024](https://link.springer.com/article/10.1186/s13731-024-00436-x) | 400 startups; IPO, next-round, acquisition categories; RF/GB/MLP/LR/SVM; SHAP and permutation importance | Explicitly recognizes different definitions and uses several outcome categories with XAI | Does not hold cohort, horizon, feature information, model/explainer, and cases fixed or quantify disagreement against a retraining-noise baseline |
| [Maarouf, Feuerriegel, and Pröllochs, 2025](https://www.sciencedirect.com/science/article/pii/S0377221724007136) | 20,172 Crunchbase profiles; fused structured/text model | Modern predictive and decision-support benchmark | One composite success target; no explanation audit |
| [SAISE systematic-review preprint, 2025](https://arxiv.org/abs/2508.05491) | Preprint review of 57 AI startup-evaluation studies | Reports fragmented success definitions, validation weaknesses, and immature XAI practice | Motivates, but cannot replace, the independent archived gap search |
| [Help Me Screen, 2025](https://doi.org/10.1145/3763001) | Long-horizon dynamic VC network and five-year success task | Modern temporal/network screening design | No outcome-specification or explanation-reliability audit |
| [Mashhadi et al., revised 2026](https://arxiv.org/abs/2510.09465) | ArXiv preprint; Crunchbase–USPTO firm-quarter panel; funding at 12 months, patent growth at 24 months, exit at 36 months; temporal splits, calibration, SHAP/importance | Closest domain threat: several outcomes and explanations in one leakage-aware pipeline | Outcomes use different horizons and are interpreted separately; no matched label effect or crossed explanation decomposition |
| [PHBench, 2026](https://arxiv.org/abs/2605.02974) | 67,292 Product Hunt posts; 528 Series-A events within 18 months | Reproducible rare-event benchmark with temporal decay analysis | One endpoint; documented timing and follower leakage; cannot replicate the label multiverse |
| [Meng, Liu, and Yuan, 2026](https://doi.org/10.1016/j.aej.2026.03.030) | *Uncertainty-aware multi-task learning for startup success and valuation prediction*, Alexandria Engineering Journal 141:570–582 | Closest prior art combining startup prediction, valuation, and uncertainty | No fixed-feature construct-variation attribution design or crossed explanation decomposition |
| [Pan, Pan, and Sun, 2026](https://www.nature.com/articles/s41598-026-44162-8) | Multi-layer startup decision support using knowledge graphs and federated learning; large claimed source data but intensive evaluation on only 20 startups | Recent domain-specific decision support and interpretation | Internal scale mismatch warrants caution; no controlled outcome-specification audit |
| [Khamphukun and Narkbunnum, 2026](https://www.mdpi.com/2078-2489/17/7/702) | 66,368-firm and 923-firm Crunchbase-derived datasets; three success constructs, three model families, five-fold stratified CV | Definitively invalidates the claim that startup papers never vary success constructs | No common-landmark event-history multiverse, crossed attribution decomposition, or decision-consequence audit |
| [Maarouf, Bakiaj, and Feuerriegel preprint, 2026](https://arxiv.org/abs/2601.16568v2) | kNN in-context learning for startup-success prediction using Crunchbase | Recent LLM-based startup prediction | Detailed sample/cutoff/outcome claims remain excluded until individually verified; no construct-sensitivity audit |
| [Bae et al., DIALECTIC, EACL Industry Track 2026](https://arxiv.org/abs/2603.12274) | Multi-agent LLM startup evaluation backtested across five VC funds | Occupies interpretable startup decision support | No verified basis here for a 259-case claim; no attribution multiverse or formal explanation-risk control |

### 4.2 Rashomon sets, attribution disagreement, and faithfulness

| Study | Established result | Consequence for this plan |
|---|---|---|
| [Fisher, Rudin, and Dominici, 2019](https://jmlr.org/papers/v20/18-760.html) | Variable reliance should be studied across all well-performing models | Single-model importance is an inadequate baseline |
| [Dong and Rudin, 2019](https://arxiv.org/abs/1901.03209) | Variable-importance clouds reveal importance variation among approximately equally accurate models | RQ2 cannot claim generic novelty |
| [Laberge et al., 2023](https://jmlr.org/papers/v24/23-0149.html) | Consensus partial orders extract attribution relations stable across a Rashomon set | The proposed robust-driver map must extend, not duplicate, consensus attribution |
| [Rashomon Importance Distribution](https://arxiv.org/abs/2309.13775) | Importance can be estimated across good models and data perturbations with uncertainty | Data-resampling instability must be included |
| [Thackshanaramana B, Hypothesis Class Determines Explanation preprint, 2026](https://arxiv.org/abs/2603.15821) | Prediction-equivalent model classes can disagree substantially on attributions; introduces the “Explanation Lottery” | The new estimand must compare construct effects with class effects |
| [CASHomon Sets preprint, 2026](https://arxiv.org/abs/2603.15321) | Cross-algorithm near-optimal model sets expose feature-importance variability | A simple hand-picked model grid is not itself innovative |
| [Barr et al., 2023](https://arxiv.org/abs/2311.07763) | Tabular faithfulness metrics can disagree | RQ3 is diagnostic validation, not a new discovery |
| [Adebayo et al., 2018](https://papers.nips.cc/paper/8160-sanity-checks-for-saliency-maps) | Model and data randomization are essential explanation sanity checks | These become mandatory go/no-go controls |
| [Faithfulness Under the Distribution, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/409fcc9d24b549969b8b9be68b56a7be-Paper-Conference.pdf) | On image benchmarks, insertion/deletion scores can be dominated by out-of-distribution perturbation artifacts | Motivates, but does not directly validate, a tabular distribution-aware conditional-perturbation sensitivity |
| [Normalized AOPC, ACL 2025](https://arxiv.org/abs/2408.08137) | Raw AOPC can mislead cross-model faithfulness comparisons | Use normalized and raw paired conditional-perturbation summaries with frozen scaling |
| [Probabilistic Stability Guarantees, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/file/67eee231405df68062cf9256a054118d-Paper-Conference.pdf) | Model-agnostic probabilistic stability certification for feature attributions | Broad attribution-stability-certification novelty is unavailable |

### 4.3 Selective explanations, conformal XAI, and risk control

| Study | Established result | Consequence for this plan |
|---|---|---|
| [Selective Ensembles, ICLR 2022](https://openreview.net/forum?id=HfUyCRBeQc) | Abstention can improve prediction and attribution consistency | “First selective region for stable explanations” is false |
| [Selective Explanations, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/647af5f6b2538524f6c047c1d9170fd9-Abstract-Conference.html) | A system can selectively allocate more accurate explanation computation | Selective explanation quality is an established topic |
| [Learning to Reject Low-Quality Explanations, 2025](https://arxiv.org/abs/2507.12900) | A learned rejector can abstain based on explanation quality, including user feedback | Generic explanation rejection is not novel |
| [Robust Explanations Through Uncertainty Decomposition preprint, 2025](https://arxiv.org/abs/2507.12913) | Epistemic uncertainty can act as a rejection signal for unreliable explanations | Any new gate must benchmark epistemic uncertainty |
| [Uncertainty Gating for Cost-Aware XAI preprint, 2026](https://arxiv.org/abs/2603.29915) | Epistemic uncertainty stratifies explanation stability and faithfulness | RQ4 cannot attribute improvement uniquely to conformal prediction |
| [But Are You Sure?, UAI 2023](https://proceedings.mlr.press/v206/marx23a.html) | Frequentist and Bayesian uncertainty sets for explanations of near-optimal or optimal models | Generic explanation-uncertainty sets are established; this paper is not a conformal method |
| [Estimating Quality of Approximated Shapley Values Using Conformal Prediction, PMLR 2024](https://proceedings.mlr.press/v230/alkhatib24a.html) | Conformal prediction can assess the quality of approximate Shapley values | Explanation uncertainty already has conformal precedents |
| [Calibrated Explanations, PMLR 2024](https://proceedings.mlr.press/v230/lofstrom24a.html) | Calibrated factual/counterfactual explanations quantify uncertainty | Calibration plus XAI is established |
| [Explaining conformal classification sets, PMLR 2025](https://proceedings.mlr.press/v266/johansson25a.html) | SHAP can explain conformal p-values and set membership | Ordinary conformal-XAI integration is not a contribution |
| [Counterfactual Explanations for Conformal Prediction Sets, PMLR 2025](https://proceedings.mlr.press/v266/maalej25a.html) | Counterfactual explanations for set-valued conformal predictions | Not a startup construct multiverse or attribution-variance audit |
| [Confidence on the Focal, JRSSB 2025](https://doi.org/10.1093/jrsssb/qkaf016) | Marginal conformal coverage need not hold after focal-unit selection | Singleton acceptance is not a conditional correctness certificate |
| [Conformal Risk Control, ICLR 2024](https://openreview.net/forum?id=33XGfHLtZg) | Bounded monotone risks can be controlled under exchangeability | A valid explanation-risk target is possible, but aggregate rather than pointwise |
| [Learn then Test](https://arxiv.org/abs/2110.01052) | Multiple-testing calibration can select configurations satisfying finite-sample risk requirements | Recommended foundation for the applied policy extension |
| [Selective Conformal Risk Control preprint, revised 2026](https://arxiv.org/abs/2512.12844) | Conditional risk can be controlled over selected samples under its stated transductive or PAC conditions | The applied policy must implement a valid theorem exactly; a new-method claim needs a non-inherited result |
| [Signed Evidence Flow preprint, revised July 2026](https://arxiv.org/abs/2606.21875) | Measures signed support, opposition, conflict, and perturbation stability; uses a held-out directional diagnostic before review triage | Relevant evidence-structure/reliability prior art, but no conformal screen is attributed without body-level confirmation |

---

## 5. Novelty verdict

### 5.1 Claims eliminated

The manuscript must not claim:

- that no startup study compares different success definitions;
- that model-class attribution disagreement is new;
- that faithfulness-metric disagreement is new;
- that selective or uncertainty-gated explanation reliability is new;
- that this is the first conformal-XAI integration;
- that singleton conformal sets certify explanation reliability;
- that a chronological test automatically inherits exchangeable conformal guarantees; or
- that positive and null results are equally publishable without power and equivalence criteria.

### 5.2 Open cross-domain claim

The preliminary missed-paper scan found no study that changes an outcome definition while holding the cases, information, model/explainer protocol, and attribution target fixed and then decomposes the resulting explanation change. However, the previously cited Frontiers paper is unrelated and cannot be used to establish prior art. Therefore:

- the question remains open until the archived multidisciplinary search is complete;
- the manuscript must not currently claim “first in any domain”; and
- failure to find such a study after the full search may support carefully bounded wording such as “no prior matched outcome-definition attribution audit was identified in our preregistered search.”

### 5.3 Primary novelty currently survives the scan

> No prior startup-screening study was identified in the preliminary documented search that jointly varies horizon-aligned outcome specifications, holds the landmark, eligible risk set, feature-information set, evaluation cases, attribution target, and model configurations fixed, decomposes attribution uncertainty across construct/model/data/explainer factors, and evaluates downstream decision consequences. This claim will be retained only if the preregistered database and citation-chain search confirms it.

The closest prior art is the [SAISE review preprint](https://arxiv.org/abs/2508.05491), which documents construct fragmentation, and [Meng, Liu, and Yuan](https://doi.org/10.1016/j.aej.2026.03.030), which combines uncertainty-aware multi-task startup prediction and valuation but does not run the matched construct-variation attribution design.

### 5.4 Secondary novelty currently survives the scan

> No startup-domain analysis was found in the preliminary documented search that benchmarks conformal-set size, calibrated confidence, epistemic uncertainty, density, and selective-ensemble disagreement as empirical stratifiers of explanation unreliability under one frozen evaluation protocol. This provisional claim also depends on the archived search.

The word “empirical” is mandatory. SAISE and Meng et al. remain required prior art even though neither performs this benchmark.

### 5.5 Conditional risk-control positioning

Without new theory, use:

> We instantiate and evaluate finite-family selective-risk-control frameworks for construct-aware explanation losses in startup screening.

This is an applied contribution, not a new conformal-XAI method. A general-method claim requires a result not inherited from LTT/CRC/SCRC—for example, a proved procedure for adaptive accepted-set ratio risks, exact selection-conditional explanation-loss control, or non-exchangeable explanation-risk control. Any guarantee is aggregate, assumption-dependent, and never an individual certificate that an explanation is scientifically true.

---

## 6. Proposed contributions

### C1. Horizon-aligned startup outcome-specification multiverse

Construct several prespecified outcomes on the same companies and frozen features, separating:

- **severity/threshold:** any follow-on financing versus Series B-or-later;
- **horizon:** the same follow-on event within 18 versus 36 months; and
- **event family:** financing progression versus recorded acquisition.

The first two axes are confirmatory. Cross-family acquisition comparisons are secondary because acquisition is a different business event, not a substitute label.

### C2. Noise-adjusted construct fragility

Introduce a matched, symmetric explanation-distance estimand that subtracts the average within-label refitting instability from cross-specification disagreement. This reduces—without claiming to eliminate—the risk that scarce events, random seeds, or model-fitting noise are misreported as a construct effect; an event-count-matched sensitivity separately estimates the scarcity envelope.

### C3. Crossed explanation-uncertainty decomposition

Estimate the contributions of:

- outcome specification;
- model family;
- hyperparameter/configuration;
- bootstrap sample and seed;
- explainer background and approximation replicate;
- startup case; and
- interactions.

Outcome and model family are fixed design factors. Seed, resample, configuration, and explainer replicate are random sources. Report the decomposition descriptively, not causally.

### C4. Construct-robust driver map

Replace a single universal ranking with four evidence states:

1. construct-robust;
2. outcome-family-specific;
3. model-contingent; and
4. unresolved/non-faithful.

The map combines attribution inclusion probability, rank stability, direction of prespecified accumulated-local-effect contrasts, faithfulness controls, and simultaneous confidence intervals. It extends Rashomon consensus analysis to an outcome-specification multiverse.

### C5. Explanation-adjusted decision consequences

Quantify whether alternative outcome definitions or explanation policies change:

- top-\(K\) screening lists;
- rank reversals and threshold crossings;
- sector and geography concentration;
- prediction calibration;
- expected utility across plausible false-positive/false-negative cost ratios; and
- analyst-review workload.

### C6. Matched-coverage uncertainty-gating benchmark

Compare ordinary conformal sets with confidence, margin, entropy, epistemic variance, data-density/OOD scores, and selective-ensemble disagreement in both a deployable 2009-threshold evaluation and a retrospective identical-batch-coverage ranking benchmark. The conformal method must earn its place empirically.

### C7. Conditional extension: LTT/SCRC-calibrated explanation-risk policy

Instantiate a finite-family risk-control procedure against fixed, bounded explanation losses rather than label-set size. Call it **Explanation Selective Risk Control (ESRC)** only as a project label, not as an assertion of algorithmic novelty. Retain it only if:

- the loss passes semi-synthetic and randomization validation;
- the candidate gate is deployment-time computable;
- the risk procedure’s assumptions and theorem are verified;
- calibration samples meet event and acceptance-size requirements; and
- generic tabular benchmarks demonstrate generality.

---

## 7. Research questions and estimands

### RQ1. Outcome-specification sensitivity

Among startups with the same landmark, follow-up eligibility, features, split, matched model configuration, explainer, and audit cases, how much do local and global attributions change across:

1. F18 versus F36, isolating horizon sensitivity for the same event type; and
2. F36 versus B+36, isolating financing-milestone severity at the same horizon?

For grouped normalized absolute attribution vectors \(\phi_a(x)\) and \(\phi_b(x)\), let \(d\) be the square root of Jensen–Shannon divergence using base-2 logarithms, so \(d\in[0,1]\). Match configuration, seed/refit identifier, case, background, and explainer replicate across outcomes. Define:

\[
D_{ab}=\mathbb{E}\left[d\{\phi_{a,m,r}(X),\phi_{b,m,r}(X)\}\right],
\]

\[
D_W=\frac{1}{2}\left[
\mathbb{E}d\{\phi_{a,m,r_1}(X),\phi_{a,m,r_2}(X)\}
+
\mathbb{E}d\{\phi_{b,m,r_1}(X),\phi_{b,m,r_2}(X)\}
\right],
\]

and make the signed, symmetric excess distance the primary inferential estimand:

\[
\Delta_{\text{spec}}=D_{ab}-D_W.
\]

Define the descriptive normalized noise-adjusted construct fragility only when the explanation pipeline is sufficiently stable:

\[
\operatorname{NACF}=\frac{\Delta_{\text{spec}}}{1-D_W},
\qquad D_W\le 1-\varepsilon.
\]

If \(D_W>1-\varepsilon\), NACF is undefined and the explanation pipeline fails the stability gate. A nonnegative \(\max(0,\operatorname{NACF})\) may be plotted descriptively, but all intervals and tests use signed \(\Delta_{\text{spec}}\). Weight equally first over eligible configurations/refit pairs and then over startups, so labels with more successful fits do not dominate. Sensitivity is supported if the lower confidence bound exceeds the prespecified smallest meaningful difference. Practical equivalence is supported only if the full confidence interval lies inside the equivalence region. Otherwise the result is inconclusive.

Because prevalence is part of an operational outcome specification but can also increase fitting instability, add a development-only event-count-matched sensitivity: downsample the more prevalent positive class to the rarer outcome’s event count while keeping negatives, configurations, and seeds matched. Report this as a scarcity envelope, not as the primary estimand.

### RQ2. Model multiplicity relative to specification sensitivity

Within each outcome, how much larger is inter-family attribution disagreement than intra-family disagreement among validation-defined near-optimal models, especially on prediction-equivalent cases?

\[
\Delta_{\text{family}}
=
\mathbb{E}_{\text{equal family pairs}}(D_{\text{inter-family}})
-
\mathbb{E}_{\text{equal families}}(D_{\text{intra-family}}).
\]

Weight family pairs equally, then models within each family pair, then startups, or use an equivalent preregistered balanced sampler. Prediction-equivalent local comparisons require the same predicted class and calibrated probability difference no greater than 0.05. Add a common-case and matched-margin sensitivity because filtering can otherwise change the case mix. Compare \(\Delta_{\text{family}}\) with \(\Delta_{\text{spec}}\) only on the same attribution scale, feature set, case population, and weighting scheme. The scientific question is their relative magnitude, not whether model multiplicity exists.

### RQ3. Faithfulness and diagnostic validity

Do the explanations:

- outperform random and bottom-ranked feature groups under distribution-aware conditional perturbations;
- respond appropriately to model-parameter and label randomization;
- recover known signals in a semi-synthetic benchmark; and
- yield consistent or explicitly discordant conclusions across prespecified faithfulness metrics?

Substantive driver interpretation is prohibited unless the explanation pipeline passes the positive and negative controls.

### RQ4. Decision consequences

How do outcome specifications and explanation uncertainty affect top-\(K\) screening overlap, rank reversals, analyst workload, calibration, and sector/geographic composition?

This RQ connects abstract explanation disagreement to an actual decision-support consequence.

### RQ5. Empirical uncertainty gating

At matched acceptance coverage, which signal best identifies cases with lower prediction error and lower explanation loss:

- conformal set size;
- calibrated maximum probability;
- margin or entropy;
- ensemble epistemic variance;
- data density/OOD score; or
- selective-ensemble disagreement?

Primary gate metric for a fresh draw from the evaluation population:

\[
R_E(g,c)=\mathbb{E}\{L_E(X)\mid g_c(X)=1\},
\]

where the gate threshold’s calibration source is stated explicitly and \(\Pr\{g_c(X)=1\}>0\). Use normalized partial area under the risk–coverage curve over \([0.50,0.80]\) as the confirmatory summary; lower is better. Gate comparisons are paired on the same test startups.

### RQ6. Conditional selective-risk application or method extension

Can an independent risk-calibration sample select the highest-coverage policy from a finite, pre-frozen candidate family while satisfying:

\[
R_S(\theta)\le \rho_S,\qquad
R_C(\theta)\le \rho_C,\qquad
R_F(\theta)\le \rho_F,\qquad
R_Y(\theta)\le \rho_Y,\qquad
q(\theta)\ge q_{\min},
\]

where \(R_S\) is accepted-set same-outcome instability, \(R_C\) is accepted-set construct fragility, \(R_F\) is accepted-set operational faithfulness-proxy loss, \(R_Y\) is accepted-set prediction error, and \(q\) is acceptance coverage?

The proof, if valid, establishes the guarantee under its stated IID conditions; the IID simulation/benchmark track evaluates empirical behavior. The chronological 2010 track evaluates transport only. Applying existing LTT/SCRC machinery to these losses is an application unless a new theoretical or algorithmic result is proved.

---

## 8. Public-data feasibility and frozen primary design

### 8.1 Primary source

Use the [Datahoarder October 2013 Crunchbase mirror](https://github.com/datahoarder/crunchbase-october-2013). The repository exists and documents four CSVs with roughly 18,000 companies and more than 52,000 investment events. The previously recorded commit `9303ebdc38fd6dbaa597e12a81edeec0246a61e2` has not been independently confirmed; Notebook 00 must resolve and record the actual checked-out commit before execution.

Do not use the nominal [mtwilliams repository](https://github.com/mtwilliams/crunchbase-in-2013) as the primary data source. Its snapshot directory contains no generated CSVs, and its README states that legacy Crunchbase API credentials are required.

Direct audit of the usable mirror found:

- 17,727 companies;
- 31,679 funding-round rows;
- 31,616 distinct round rows after removing 63 exact duplicates;
- 52,870 raw investment rows, containing 152 duplicate company/investor/date/type keys and one malformed row with no company;
- 52,717 valid unique investment edges after key deduplication and removal of the malformed row;
- approximately 4,649 distinct nonempty acquisition records;
- final event month October 2013; and
- no dated IPO event table.

The 344 snapshot records with status equal to IPO cannot be converted into IPO-within-horizon labels. IPO is therefore excluded from the public-data confirmatory design.

### 8.2 Source hashes

Freeze the following SHA-256 values:

| File | SHA-256 |
|---|---|
| companies | 83217e8165db15124aec2799f5f28bda910ab09cc1ed9553b066eb97f60f195d |
| rounds | 1b27a2b561678fe6257556d729819edc98a9581b18d6d8fad82413658e055212 |
| investments | 8f13f28d35915633c0a814dd22d49540ee115717d10d387ebc3a49183a35b65f |
| acquisitions | c3fb008fa75c48cafea7bdbd4eca050c11a55383d558998d349433f201b9f779 |

### 8.3 Landmark cohort

The primary unit is one company.

\[
t_0=\text{exact funded-at date of the first unambiguous recorded angel or Series-A financing event}.
\]

Build the event table deterministically before selecting the cohort:

1. remove exact duplicate rows;
2. collapse rows sharing company, funded-at date, and round type to one event;
3. use the maximum reported amount as primary, the sum as a sensitivity, and flag conflicts;
4. exclude a company if its earliest funded-at date contains multiple distinct round types;
5. join investors only after round aggregation, using company/date/type because the investment table has no round ID; and
6. forbid row-order-dependent operations, including `GroupBy.first()`, for landmark features.

Eligibility:

1. the unique earliest-date round type is angel or Series A;
2. \(t_0\) lies from 1 January 2005 through 1 October 2010 inclusive;
3. 36 complete calendar months are observable before the 1 October 2013 administrative cutoff;
4. the company is not recorded as acquired before \(t_0\);
5. the deterministic event-building rules above are satisfied; and
6. a later event on the same date is not treated as subsequent, while horizon endpoints use exact dates plus calendar-month offsets.

The deterministic rules are expected to yield 4,234 companies, based on the provisional audit. This is not yet an executable reproduction result. Notebook 02–03 must regenerate and assert the cohort flow and all counts from raw files before any value is called verified or frozen. Angel and Series-A landmarks remain distinct strata and must be adjusted for and analysed separately in sensitivity analyses.

### 8.4 Outcome-specification grid

| Code | Definition | Axis and inferential role |
|---|---|---|
| F18 | Any distinct subsequent qualifying pre-IPO financing within 18 calendar months | Confirmatory horizon contrast |
| F36 | Any distinct subsequent qualifying pre-IPO financing within 36 calendar months | Confirmatory reference outcome |
| B+36 | Recorded Series B or Series C+ within 36 months | Confirmatory severity contrast |
| C+36 | Recorded Series C+ within 36 months | Exploratory; event counts fail the preferred gates |
| A36 | Recorded acquisition within 36 months | Secondary, different event family |
| Broad36 | F36 or A36 | Sensitivity only |

Acquisition is a recorded milestone, not automatically a successful or valuable exit. Price is observed for too few records to define a reliable valuable-acquisition endpoint.

For F18/F36, qualifying primary event types are angel, Series A/B/C+, venture, other, private equity, and crowdfunding; explicitly exclude post-IPO rounds. Add an institutional-financing sensitivity that excludes `other` and `crowdfunding`.

### 8.5 Expected event counts pending executable reproduction

| Block | n | F18 | F36 | B+36 | C+36 | A36 |
|---|---:|---:|---:|---:|---:|---:|
| Model development, 2005–2007 | 1,811 | 643 | 986 | 698 | 199 | 136 |
| Probability calibration, 2008 | 902 | 211 | 372 | 190 | 49 | 72 |
| Explanation/risk calibration, 2009 | 733 | 275 | 396 | 174 | 49 | 59 |
| Untouched test, 1 Jan–1 Oct 2010 | 788 | 304 | 427 | 173 | 41 | 75 |
| Total | 4,234 | 1,433 | 2,181 | 1,235 | 338 | 342 |

C+36 is provisionally exploratory because its calibration and test blocks contain fewer than 50 positives. A36 provisionally passes the holdout event floor but has only 136 development positives and is a different estimand. Every number in this table is an expected value, not a reported experimental result, until the released deterministic event builder reproduces it exactly.

### 8.6 Competing and nested events

B+36 is nested within F36. Acquisition may compete with later financing. Report:

- the full label-overlap matrix;
- a sensitivity excluding companies acquired before the financing event;
- Broad36 as a union sensitivity;
- a descriptive multi-state table: no recorded event, financing only, acquisition only, both; and
- a cause-specific or multi-state survival sensitivity if the event-time data support stable estimation.

The primary claims concern recorded Crunchbase events among Crunchbase-visible early-stage firms, not all startups and not true failure.

### 8.7 Feature-time contract

Every feature must be assigned one approved availability class at \(t_0\): **event-timed**, **snapshot-static with uncertain historical availability**, or **prohibited outcome-accumulating**. A missing or unapproved class is an automated failure.

Event-timed and permitted in the strict set:

- exact landmark round type and amount;
- landmark syndicate edge count;
- each investor’s prior event count and network statistics computed exclusively from edges before \(t_0\);
- lagged overall market deal volume and round-size summaries;
- calendar year and quarter; and
- missingness, conflict, and source-quality indicators derived from those event-timed fields.

Snapshot-static and permitted only in the extended set:

- founding-to-first-financing age, with missing and invalid-date indicators;
- company industry/category and coarse geography;
- investor geography and breadth metadata; and
- category-specific or region-specific lagged activity whose category/geography tags lack historical observation timestamps.

Prohibited:

- snapshot total funding;
- snapshot funding-round count;
- last-funding date;
- last-milestone date;
- snapshot operating/acquired/IPO/closed status;
- all investments, network edges, milestones, or acquisitions after \(t_0\);
- acquisition price; and
- any field whose source-record availability cannot be placed at or before \(t_0\).

Observed constraints:

- exact landmark-round amount is missing for 659 of 4,234 companies (15.56%);
- founding date is missing for 607 of 4,234 (14.34%);
- 418 of 4,234 records have founding dates after first financing; and
- 1,074 have no matched landmark investor edge.

Invalid founding ages must become missing values with an invalid-date flag. They must never be repaired using later information. An automated lineage test must show that every landmark-round feature originates from the exact aggregated landmark event, never a later non-null row.

Two feature sets are mandatory:

1. **Strict event-timed set:** only variables reconstructible from dated financing/investment events plus calendar time.
2. **Snapshot-extended set:** adds founding age, company/investor metadata, category/geography, and metadata-dependent activity with an explicit historical-availability caveat.

All headline conclusions must survive the strict feature-set sensitivity or be labelled dependent on snapshot-static metadata.

### 8.8 Secondary data roles

#### Kaggle 66,368-company dataset

This file contains only snapshot/cumulative columns and lacks event histories for acquisition, IPO, and round-stage transitions. Funding totals, funding-round counts, last-funding dates, and status are outcome-accumulating leakage for an early-landmark design.

Permitted role: **leaky-snapshot negative control only**. It may demonstrate how leakage inflates predictive performance and changes explanatory rankings. It is not a replication dataset and does not mitigate Crunchbase ascertainment bias.

#### PHBench

[PHBench](https://github.com/ihlamury/phbench) contains 67,292 Product Hunt posts and 528 Series-A outcomes within 18 months. It has one outcome and cannot replicate RQ1.

Its timing limitations include:

- maker-follower values collected years after some launch dates;
- current API retrieval of engagement counts that may not represent a 24-hour frozen landmark;
- weekly/monthly ranks that are unavailable at launch-day decision time;
- unresolved or right-censored 2025 outcomes;
- unknown false negatives from exact-domain linkage; and
- private test labels that limit independent inferential analysis.

Permitted role: **optional secondary RQ2–RQ5 stress test**, subject to public/gated access, using progressively less strict feature sets:

1. static-clean: topic, text-shape, and launch-calendar variables;
2. add end-of-day rank if its timing is defensible;
3. add retrospective engagement/rank variables;
4. add explicitly leaky follower variables.

Report the performance and explanation changes across these sets as a leakage stress curve. Public train and validation contain only 425 positives and public test labels are unavailable, so do not promise independent test inference. The workflow must fail gracefully if gated access or private scoring is unavailable. Never describe PHBench as a leakage-clean independent replication.

### 8.9 Optional data-upgrade path

If legitimate academic access to a richer, event-dated Crunchbase, Dealroom, PitchBook, or author-shared panel becomes available, add:

- event-dated IPO;
- closure/survival with a recording-quality audit;
- patents and citations;
- richer founder/team histories; and
- modern post-2020 cohorts.

The richer dataset may replace the public mirror only after its license, provenance, landmark feature availability, censoring, positive-event counts, and reproducibility terms are documented. The public 2013 mirror remains a fully reproducible robustness dataset.

---

## 9. Experimental design

### 9.1 Phase 0: irreversible preflight gates

Complete before opening any final-test model-performance, attribution, faithfulness, decision, or gate result. Constructing the cohort and count table necessarily uses labels; if outcome blinding is desired, a data steward or automated sealing script must expose counts without exposing row-level 2010 labels to analysts.

1. reproduce source hashes and row counts;
2. freeze the administrative cutoff and duplicate-resolution rules;
3. create the company inclusion/exclusion flow;
4. generate the label-overlap and competing-event table;
5. freeze the feature-time ledger;
6. freeze company IDs in every time block;
7. run event-count and missingness gates;
8. run structural event-count and analytic-precision gates;
9. preregister primary endpoints, contrasts, metrics, thresholds, and multiplicity families; and
10. generate a signed configuration file that subsequent notebooks read without modification.

No model result may determine cohort, label, feature, or split inclusion.

### 9.2 Time partition

| Period | Permitted uses | Prohibited uses |
|---|---|---|
| 2005–2007 | fitting, blocked inner CV, hyperparameter and near-optimal-set selection | probability or risk calibration on later data |
| 2008 | probability calibration and cross-fitted explanation-risk meta-model development | model-family or hyperparameter selection |
| 2009 | split-conformal or learn-then-test/selective-risk calibration | redesigning the loss or candidate gates |
| 1 Jan–1 Oct 2010 | final evaluation only | any tuning, threshold choice, metric selection, or endpoint deletion |

Use four expanding-window folds entirely within 2005–2007: train 2005/validate 2006 H1; train through 2006 H1/validate 2006 H2; train through 2006/validate 2007 H1; and train through 2007 H1/validate 2007 H2. Pool out-of-time predictions for model comparison and obtain uncertainty by startup bootstrap stratified by validation calendar block; do not estimate the one-standard-error rule from four fold means alone. Within 2008, use cross-fitting whenever the same block supports probability calibration and a meta-score for explanation risk, so no case’s prediction or explanation loss is evaluated by a nuisance model trained on that case.

### 9.3 Preprocessing

Within every relevant training fold:

- convert impossible dates to missing and add flags;
- log-transform positive funding amounts;
- winsorize continuous features using training-only thresholds;
- median-impute numeric features and add missingness indicators;
- group rare categories using training-only frequencies;
- one-hot encode with an unknown-level channel;
- standardize only models that require it;
- construct network features from a graph truncated strictly before each \(t_0\); and
- construct market features from lagged events only.

Primary analysis: no SMOTE and no outcome-specific class weighting. Class weighting is a prespecified sensitivity and is treated as part of the configuration; it cannot vary silently across labels in RQ1. If SMOTE-NC is tested, it is confined to training folds and excluded from probability calibration, explanation-background construction, and evaluation.

### 9.4 Model families

Confirmatory:

1. elastic-net logistic regression;
2. explainable boosting machine;
3. random forest; and
4. XGBoost.

Secondary:

- LightGBM as an algorithm sensitivity;
- shallow MLP only if it passes the minimum-skill and calibration gates; and
- a survival or multi-state model for the censoring/competing-risk sensitivity.

The MLP must not be forced into the comparison merely to increase model diversity. With \(n=4,234\), a poorly performing MLP adds noise rather than scientific value.

### 9.5 Imbalance-aware predictive evaluation

Primary metric: average precision and AP lift over outcome prevalence.

Required secondary metrics:

- log loss;
- Brier score and Brier skill;
- calibration intercept and slope;
- reliability diagram with uncertainty;
- AUROC;
- precision and recall at confirmatory review budgets of 5%, 10%, and one stakeholder-frozen operational budget; 1% is descriptive only because it contains roughly eight test firms;
- lift at those budgets;
- decision-curve net benefit; and
- expected utility across a prespecified grid of false-positive/false-negative cost ratios.

No explanation conclusion is substantively interpreted for a label whose model fails the minimum predictive-skill gate.

### 9.6 Probability calibration

Fit calibrators on 2008 only. For the primary matched-label explanation contrast, use the same smooth calibration family—Platt scaling—fit separately with the identical procedure for every label/model. This prevents label-specific choices between smooth and stepwise mappings from becoming part of the apparent construct effect.

For predictive and decision-calibration sensitivity, compare Platt, isotonic, and beta calibration by cross-fitted 2008 log loss. Among methods within one bootstrap standard error of the best, choose Platt unless its cross-fitted calibration-slope 95% interval falls outside the frozen acceptable interval \([0.8,1.2]\); otherwise choose the simplest eligible alternative by the preregistered order beta then isotonic. Refit the selected mapping on all 2008 cases and freeze it before 2009. The explanation-risk meta-model uses only out-of-fold calibrated 2008 predictions and losses; the all-2008 calibrator is applied only to 2009–2010.

For explanation comparability, the common black-box function is the clipped calibrated log-odds:

\[
f(x)=\log\left\{\frac{\tilde p(x)}{1-\tilde p(x)}\right\},
\]

where \(\tilde p\) is clipped only by a preregistered numerical bound.

### 9.7 Constructing empirical near-optimal model sets

Do not use a fixed 0.01 or 0.02 test-AUROC band.

Procedure:

1. use at most 24 hyperparameter configurations per family, balanced across families, with three fitting seeds per configuration during selection;
2. evaluate every configuration using blocked inner cross-validation on 2005–2007;
3. define near-optimal membership using the one-standard-error rule on log loss;
4. require AP lift, positive Brier skill, and absence of gross calibration failure;
5. freeze membership before accessing 2008–2010 for its permitted roles;
6. retain both a global near-optimal set and family-specific near-optimal sets, capped by a frozen tie-breaking rule at 10 members per family for the exhaustive explanation audit; and
7. report if a model family has no globally near-optimal member.

For RQ1, the primary controlled analysis uses identical configurations and seeds across F18, F36, and B+36. A reference configuration selected on F36 is refit unchanged for the other outcomes. Label-specific tuning is a sensitivity analysis, because otherwise the label effect is confounded with model-selection choice.

For RQ2, use label-specific near-optimal sets, while identifying local comparisons that are prediction-equivalent.

The term “empirical near-optimal set” is preferred unless the sampled search can be shown to approximate a formal Rashomon set.

For uncertainty estimation, plan **100 company-bootstrap refits per headline configuration** using one frozen seed registry. The development compute pilot may revise this number only through a documented Monte Carlo-precision and power analysis completed before any 2010 result is generated; it may not be reduced merely for convenience or changed selectively by outcome/model.

### 9.8 Development-only design-freeze gate

After pilot models and explanations have run on 2005–2008 only, but before any 2010 model/explanation evaluation:

1. estimate explainer approximation error and the variance components of \(\Delta_{\text{spec}}\) from development-only pilot fits;
2. run and archive an RQ1 power simulation using the actual paired design, planned local-audit sample, model/configuration weighting, bootstrap-refit count, observed refit noise, and the same hierarchical-bootstrap decision rule intended for the final analysis;
3. validate the mixed-type conditional sampler and semi-synthetic controls;
4. benchmark runtime and freeze the model/refit/explanation budgets;
5. freeze every loss formula, risk budget, gate candidate, threshold, and multiplicity rule; and
6. sign a second design-lock file. Any later change is a labelled exploratory deviation.

The RQ1 power simulation must use at least 2,000 Monte Carlo datasets per effect/noise scenario over a grid containing the null, equivalence boundary, smallest meaningful effect, and larger effects. It must report rejection probability, equivalence-decision probability, inconclusive probability, confidence-interval width, Monte Carlo standard error, and sensitivity to 50/75/100 refits. No “80% power” statement is permitted until these results exist.

---

## 10. Common explanation engine

### 10.1 Semantic feature groups

Create approximately 8–12 groups before modelling:

- firm age and date quality;
- initial financing amount;
- initial round type;
- first-round syndicate size/diversity;
- investor experience;
- investor network position;
- industry/category;
- geography;
- lagged category-market activity;
- lagged regional-market activity;
- calendar/regime variables; and
- missingness/data-quality signals.

Group definitions must not change after explanation results are viewed.

### 10.2 Headline attribution method

Use **grouped permutation SHAP** as the one model-agnostic headline method across all model families, implemented by permuting prespecified semantic groups as coalitions. Freeze:

- the primary Platt-calibrated log-odds output;
- three independently sampled, stratified background sets of 100 companies from 2005–2007, with IDs released;
- all 788 test cases for frozen-deployment-model summaries and decision analyses;
- a model-result-blind, stratified local-audit subset of 300 test cases for the exhaustive model × outcome × refit multiverse, sampled by landmark and joint outcome pattern with released inclusion probabilities and design weights;
- at most 10 retained near-optimal members per family;
- 20 antithetic permutation orderings per explanation, or an equivalent implementation whose evaluation budget is documented; and
- three approximation seeds per background.

Run a development-only compute and Monte Carlo error benchmark before freezing these budgets. If the budget is changed, apply the same rule across all compared outcomes and model families before viewing 2010 results.

Native TreeSHAP, linear coefficients, EBM contributions, and global permutation importance are sensitivity analyses. They must not be compared as if they estimate identical objects.

### 10.3 Correlation and reference distributions

Primary explanation: grouped interventional attribution with a common background sampled from the development cohort.

Mandatory sensitivity:

- grouped conditional permutation using a development-fitted mixed-type Gower-k-nearest-neighbour donor sampler, with \(k\) chosen on 2005–2007 reconstruction/support diagnostics and then frozen;
- at least three independently drawn background samples;
- event-derived-only versus extended feature sets; and
- alternative reference strata for angel and Series-A landmarks.

Report held-out marginal reconstruction error, neighbour distances, effective donor counts, and categorical/continuous support violations. If the sampler fails the frozen diagnostics, label the result a conditional-imputation sensitivity—not conditional SHAP or “on-manifold” truth. Correlation-clustered PartitionSHAP is not accepted as a substitute for a conditional distribution. This analysis tests dependence on the reference distribution and correlated features.

### 10.4 Attribution normalization and distances

For case \(i\), model \(m\), outcome \(y\), and grouped attribution vector \(\phi_{i,m,y}\):

1. preserve signed log-odds attributions for directional analyses;
2. define absolute normalized importance

\[
u_{i,m,y,g}=
\frac{|\phi_{i,m,y,g}|}
{\sum_h|\phi_{i,m,y,h}|+\varepsilon};
\]

3. use square-root Jensen–Shannon divergence with base-2 logarithms as the primary magnitude-disagreement distance;
4. use rank-biased overlap as the primary top-weighted ranking metric; and
5. report top-3/top-5 Jaccard, Kendall \(\tau_b\), cosine distance, and top-feature sign agreement as secondary diagnostics.

If both attribution vectors have total absolute mass below the frozen numerical tolerance, define their distance as 0; if exactly one is null, define the distance as 1. Report the null-attribution rate by outcome/model. The primary metric, top-\(k\), sign-magnitude threshold, tolerance, and \(\varepsilon\) in NACF are frozen in Phase 0.

### 10.5 Local evaluation populations

All paired explanation contrasts use the same startups.

Report separately:

- all eligible cases;
- cases with the same true outcome under compared specifications;
- cases with the same predicted class;
- prediction-equivalent model pairs;
- correctly and incorrectly predicted cases; and
- angel versus Series-A landmarks.

These strata distinguish explanation disagreement caused by different model outputs from disagreement conditional on comparable predictions.

---

## 11. Crossed explanation-uncertainty decomposition

### 11.1 Design factors

For each startup and semantic feature group, retain long-form attributions indexed by:

- outcome specification;
- model family;
- configuration;
- bootstrap/refit seed;
- explainer background;
- explainer approximation seed; and
- test case.

### 11.2 Primary inference and descriptive decomposition

Primary inference remains the prespecified paired vector-distance contrasts \(\Delta_{\text{spec}}\) and \(\Delta_{\text{family}}\), estimated by a hierarchical bootstrap that preserves matched cases and refits. A distance-to-medoid summary discards attribution direction and is therefore descriptive only.

For decomposition, fit a feature-group-specific crossed hierarchical model to signed log-odds attributions after division by a common development-frozen robust scale for that group. The nesting/crossing is:

- outcome specification and model family are fixed effects, including their interaction;
- configuration is nested within model family;
- bootstrap/refit is nested within outcome × configuration;
- background and approximation replicate are crossed with outcome/model cells;
- startup is crossed with all design factors; and
- uncertainty uses startup- and refit-cluster bootstrap or a prespecified multi-membership model.

Repeat descriptively for logit-transformed normalized absolute importance with a frozen zero adjustment. Report adjusted contrasts, specification-by-family interaction, design-specific partial \(R^2\), stochastic variance components for refit/background/approximation, and startup heterogeneity. Freeze the exact formula and software in the design-lock file and verify recovery in simulation.

The finite outcomes and model families are not random samples from all possible constructs or algorithms. Their partial \(R^2\) and “variance shares” are design-specific descriptive quantities, not population-causal fractions. The paired bootstrap contrasts, not a variance component, answer the main scientific question.

### 11.3 Feature-level secondary decomposition

For each semantic group:

- estimate its probability of appearing in the top-\(k\);
- estimate rank and attribution-magnitude distributions;
- estimate the prespecified 75th-versus-25th percentile accumulated-local-effect contrast;
- quantify outcome, family, refit, and explainer components; and
- construct simultaneous 95% confidence intervals using max-\(T\), bootstrap, or a preregistered multiplicity procedure.

### 11.4 Construct-robust driver taxonomy

Classify a feature group only if its confidence bounds meet frozen criteria:

| State | Required evidence |
|---|---|
| Construct-robust | High top-\(k\) inclusion and stable ALE contrast across F18, F36, B+36 and eligible near-optimal models; passes faithfulness controls |
| Family-specific | Stable within a coherent outcome family but materially different outside it |
| Model-contingent | Stable within families but not across near-optimal model families |
| Unresolved/non-faithful | Wide intervals, inconsistent direction, background sensitivity, or failed sanity/faithfulness controls |

Nonlinear features may have no single global sign. In those cases report shape stability rather than forcing a positive/negative label.

---

## 12. Faithfulness and explanation sanity

### 12.1 Primary distribution-aware conditional-perturbation test

For each case and explanation:

1. rank semantic feature groups;
2. remove groups cumulatively;
3. replace them using a conditional distribution learned on development data only;
4. compare top-ranked, bottom-ranked, and random group-removal curves; and
5. summarize using raw paired AOPC and a development-scaled normalized AOPC with the scale and clipping frozen before test evaluation.

Naive zero, mean, or impossible-feature deletion is prohibited as a headline faithfulness test.

### 12.2 Local distribution-aware fidelity

Generate approximately supported local perturbations from the frozen development-only Gower-kNN donor model. Evaluate how well the attribution-based local surrogate reproduces the black-box calibrated log-odds in that neighbourhood. Call this conditional-imputation sensitivity, not proof that samples lie on the true data manifold.

### 12.3 Global reliance sensitivity

Conduct grouped remove-and-retrain or ROAR-style experiments as a separate global estimand. Label this “global predictive reliance under a changed feature set,” not local explanation faithfulness.

### 12.4 Mandatory controls

- random feature ranking;
- reversed/bottom-first ranking;
- model-parameter randomization;
- label randomization followed by refitting;
- explanation approximation-repeat stability;
- semi-synthetic outcomes constructed from the real covariates with known shared and outcome-specific signal groups; and
- an isolated injected post-landmark feature, `future_round_count_36`, as a positive control for leakage detection.

Run randomized-label, semi-synthetic, and injected-leakage controls only on development copies with separate matrix identifiers. The leakage column must never be merged into, or share a feature registry with, the approved analysis matrix.

### 12.5 Interpretation gate

Do not report “drivers” from real data unless:

- top-ranked conditional deletion outperforms random by the frozen meaningful margin;
- the semi-synthetic benchmark recovers known signals;
- attributions change under parameter and label randomization;
- explainer approximation error is smaller than the headline specification effect; and
- results are not reversed by the conditional-versus-interventional sensitivity.

---

## 13. Decision-consequence audit

For every outcome specification and eligible model:

1. rank the same 2010 startups by calibrated probability;
2. compare top-\(K\) lists confirmatorily at 5%, 10%, and one stakeholder-frozen operational review budget; show 1% descriptively only;
3. report Jaccard overlap, rank-biased overlap, rank reversals, and threshold crossings;
4. calculate precision, recall, lift, and net benefit;
5. measure the sector, country/region, and landmark-stage composition of selected lists after collapsing categories with training-only rules and enforcing frozen minimum reporting-cell counts;
6. report subgroup calibration and error where sample sizes permit;
7. calculate analyst workload under each accept/review policy; and
8. show case studies of high-confidence but construct-fragile startups.

Do not infer protected attributes that are absent from the source. Geography and industry analyses are coverage/selection audits, not substitutes for demographic fairness.

The primary DSS contribution is the relationship:

\[
\text{outcome specification}
\rightarrow
\text{explanatory conclusion}
\rightarrow
\text{screening-list change}
\rightarrow
\text{review workload or utility}.
\]

---

## 14. Uncertainty-gating benchmark

### 14.1 Why ordinary singleton conformal prediction is only a baseline

For binary probability-score conformal prediction with a fixed score quantile, singleton status is a deterministic thresholding rule in the two class probabilities—equivalently in maximum probability/margin, up to the frozen quantile, tie, and empty-set conventions. Mondrian calibration makes the thresholds label-specific. Singleton status may therefore be identical or nearly rank-equivalent to confidence gating.

If the fitted predictor and nonconformity rule are fixed without the calibration labels, and the new labelled point is exchangeable with the calibration observations, standard split conformal with the finite-sample quantile/tie convention provides marginal coverage:

\[
\Pr\{Y\in C_\alpha(X)\}\ge 1-\alpha.
\]

It does not provide:

- correctness of each singleton;
- accepted-subset error no greater than \(\alpha\);
- conditional coverage after singleton selection;
- explanation stability;
- explanation faithfulness; or
- validity under an unrestricted chronological distribution shift.

### 14.2 Frozen deployment predictor

RQ5 uses one frozen deployment model and one common explanation-risk evaluation population. Do not let each Rashomon member create its own accepted set for the headline comparison.

Freeze the base family, hyperparameters, features, and decision rule from 2005–2007 only. The 2008 block may select and fit only the probability-calibration mapping under Section 9.6 and the cross-fitted explanation-loss meta-model. Nothing in 2008 may change the base predictor.

### 14.3 Gates

Compare:

1. standard split-conformal singleton status;
2. class-conditional/Mondrian conformal status when calibration counts permit;
3. calibrated maximum probability;
4. probability margin;
5. predictive entropy;
6. bootstrap/ensemble epistemic variance;
7. selective-ensemble disagreement;
8. development-density or OOD score;
9. a cross-fitted predicted explanation-loss score;
10. random acceptance; and
11. an unattainable oracle ranked by realized explanation loss, shown only as an upper benchmark.

### 14.4 Matched coverage

Use two clearly separated evaluations.

**Primary deployable evaluation.** For gate score \(s_g\) and target calibration acceptance \(c\), freeze the threshold \(t_{g,c}\) as the corresponding 2009 risk-calibration quantile, using a frozen orientation and tie rule. Then apply \(g_c(x)=\mathbf 1\{s_g(x)\le t_{g,c}\}\), or the preregistered reverse inequality, to 2010 and report realized \(q_{2010}\) rather than forcing it to equal \(c\).

**Secondary retrospective ranking benchmark.** Accept the exact top-\(c\) fraction by unlabeled 2010 gate score to compare ranking ability at identical batch coverage. Label this transductive/ranking-only; it is not an online deployment policy or a population-risk guarantee.

The confirmatory grid is:

\[
c\in\{0.50,0.60,0.70,0.80\}.
\]

Show 0.90 and 0.95 descriptively only: with \(n=788\), they leave fewer than 100 reviewed cases. Conformal singleton gates may not attain every coverage because of set-valued ties; report the nearest attainable coverage without randomized tie-breaking in the primary analysis. Interpolate only for visualization.

### 14.5 Outcomes

Report:

- prediction error–coverage;
- log-loss–coverage;
- explanation-instability–coverage;
- conditional-faithfulness-loss–coverage;
- normalized partial area under each risk–coverage curve on \([0.50,0.80]\), lower being better;
- class-conditional acceptance and error;
- subgroup acceptance composition;
- review workload; and
- cross-fitted predictive improvement beyond probability margin, while reporting deterministic/near-deterministic collinearity for binary conformal singleton status.

Compare gates using paired 2010-startup bootstrap resamples because every frozen gate is evaluated on the same startups. These intervals condition on the trained models, meta-scores, and 2009 thresholds. Label them conditional inference. Use a nested full-pipeline bootstrap, including threshold recalibration, only for claims intended to cover training/calibration uncertainty.

---

## 15. Conditional extension: LTT/SCRC-calibrated explanation-risk policy

### 15.1 Purpose

The extension targets operational explanation losses rather than using label-set size as a proxy. Unless new theory is proved, it is an application of existing finite-family selective-risk-control machinery, not a new conformal method.

### 15.2 Prespecified bounded losses

Keep risks separate. Let \(u_0(x)\) be the deployed normalized absolute grouped-attribution vector. Before 2009, freeze a finite panel of \(K\) same-outcome model/refit explanations \(u_k(x)\), matched confirmatory construct explanations, all nuisance models, and all Monte Carlo draws.

1. **Same-outcome instability**

   \[
   L_S(x)=\frac{1}{K}\sum_{k=1}^{K}d\{u_0(x),u_k(x)\}\in[0,1].
   \]

   The estimand is conditional on this finite frozen panel; it is not uncertainty over every possible future model.

2. **Construct fragility**

   \[
   L_C(x)=\frac{1}{2}\left[d\{u_{F18}(x),u_{F36}(x)\}+d\{u_{F36}(x),u_{B+36}(x)\}\right]\in[0,1],
   \]

   using matched configurations/refits. This controls an operational sensitivity to the two confirmatory outcome contrasts, not disagreement that is necessarily erroneous.

3. **Distribution-aware faithfulness-proxy loss.** Compute top-ranked and mean random-ranking AOPC on the predicted-class probability using frozen perturbation draws, so each AOPC lies in \([0,1]\). Let \(s_F>0\) be a scale fixed from development data, with a preregistered positive floor. Define

   \[
   L_F(x)=1-\operatorname{clip}\left(\frac{A_{\text{top}}(x)-A_{\text{random}}(x)}{s_F},0,1\right)\in[0,1].
   \]

   Report raw paired AOPC and bottom-ranked performance separately. This loss controls only the quality proxy defined by the frozen perturbation test, not scientific truth.

4. **Prediction error**

   \[
   L_Y(x,y)=\mathbf 1\{\hat y(x)\ne y\}.
   \]

All predictors, explainers, backgrounds, conditional-imputation models, reference panels, candidate gates, meta-models, normalization constants, random rankings, perturbation draws, and Monte Carlo budgets must be fitted/frozen using data through 2008 only. Use fixed seeds and fixed Monte Carlo averages, making each loss a deterministic bounded functional at risk calibration and deployment. Missing/failed explanations receive loss 1 rather than being silently dropped.

### 15.3 Deployment-time gate score

The finite candidate family may use these scalar scores:

- calibrated probability and margin;
- entropy;
- ensemble variance;
- model disagreement;
- development-data density;
- explanation approximation variance;
- a cheaply estimated partial-refit attribution dispersion; and
- a cross-fitted meta-model predicting \(L_S\) and \(L_F\).

No candidate uses a 2009/2010 outcome, realized loss, or nuisance fit. If computing the full explanation loss is as expensive as review, the deployable gate must use a cheaper pre-frozen predicted-loss proxy.

Freeze exactly one specification of each score and one cross-fitted meta-score by the end of 2008. For each orientation, use thresholds defined by the 2008 score quantiles corresponding to target acceptances \(0.50,0.55,\ldots,0.95\). This produces a finite candidate set of at most 90 policies; thresholds may not be regenerated from 2009 losses. Any intersection or ensemble gate must replace, not augment, that frozen count unless the multiplicity correction is updated before 2009.

### 15.4 Calibration

Use 2008 to train the explanation-loss meta-score with internal cross-fitting and to freeze the finite candidate family. Use 2009 only as an independent risk-calibration block.

For a nested family of acceptance rules \(g_\theta\), estimate:

\[
R_S(\theta)=\mathbb{E}\{L_S\mid g_\theta(X)=1\},
\]

\[
R_C(\theta)=\mathbb{E}\{L_C\mid g_\theta(X)=1\},
\]

\[
R_F(\theta)=\mathbb{E}\{L_F\mid g_\theta(X)=1\},
\]

\[
R_Y(\theta)=\Pr\{\hat y(X)\ne Y\mid g_\theta(X)=1\},
\]

and

\[
q(\theta)=\Pr\{g_\theta(X)=1\}.
\]

All risks refer to a fresh draw and require \(q(\theta)>0\). Initial operational budgets are \(\rho_S=0.20\), \(\rho_C=0.25\), \(\rho_F=0.50\), \(\rho_Y=0.20\), \(q_{\min}=0.50\), and familywise \(\delta=0.05\). These values must be checked with stakeholder elicitation and development-only utility simulation; any change must be signed before 2009 and reported as a protocol revision, never optimized on 2009/2010.

For every fixed \(\theta\), let \(I_\theta=\{i:g_\theta(X_i)=1\}\) and \(N_\theta=|I_\theta|\). On independent 2009 risk calibration:

1. test each conditional mean using only \(\{L_{j,i}:i\in I_\theta\}\); conditional on \(N_\theta\), these are IID from the accepted population when \(g_\theta\) is a fixed function of \(X\);
2. use a valid one-sided bounded-mean test such as Hoeffding–Bentkus for \(R_j\le\rho_j\), requiring \(N_\theta\ge100\) as a research precision/certification rule;
3. test \(q(\theta)\ge q_{\min}\) separately using an exact binomial or valid one-sided lower confidence bound over all calibration acceptance indicators;
4. form the intersection–union p-value \(p_\theta=\max(p_S,p_C,p_F,p_Y,p_q)\) for the union null that at least one constraint fails;
5. apply the prespecified LTT/FWER procedure across all frozen \(\theta\); and
6. select maximum frozen utility, breaking ties by coverage then simplicity, only among simultaneously certified policies.

Rejected observations must not be coded as zero loss: that controls \(qR\), not \(R=\mathbb{E}[L\mid g=1]\). Ordinary CRC is not automatically a selective conditional-risk method, even when risk is monotone. If a continuous or adaptively generated candidate family is desired, discretize and freeze it or supply a separate uniform/selective-risk proof.

### 15.5 Honest guarantee

Conditional on every pre-calibration fitted object and candidate configuration, the intended statement is:

> If the risk-calibration observations are independent and identically distributed from the same population as fresh future deployment observations, then, with probability at least \(1-\delta\) over the independent risk-calibration sample, every policy certified by the simultaneous testing procedure—and therefore the selected certified policy—satisfies \(R_S\le\rho_S\), \(R_C\le\rho_C\), \(R_F\le\rho_F\), \(R_Y\le\rho_Y\), and \(q\ge q_{\min}\).

This is:

- aggregate rather than pointwise;
- conditional on the finite frozen reference panels and defined operational losses;
- dependent on the calibration assumptions and multiple-testing procedure;
- about a fresh draw from the same population, not the realized 2010 selected subset; and
- not a statement that an individual explanation is scientifically correct.

Generic finite exchangeability is not substituted for the independence required by this Hoeffding–Bentkus/LTT population-risk statement. Any transductive or exchangeability-only guarantee requires its own cited theorem and proof.

### 15.6 IID and temporal tracks

**Track A: theorem-behavior validation.** The cleanest check is 1,000 independent synthetic/semi-synthetic IID repetitions, never fewer than 500, with a large independent population for evaluating true constraint violations. Define a violation as the selected certified policy breaching any one simultaneous population constraint. Report an exact binomial Monte Carlo interval and whether its upper bound is compatible with the theorem’s familywise \(\delta\); empirical checks do not prove the theorem.

Use a 40% base-train / 15% probability-calibration / 15% meta-training / 15% risk-calibration / 15% test split for supplementary real-data benchmarks. Include at least Adult (OpenML 1590), Bank Marketing (OpenML 1461), credit-g (OpenML 31), and Phoneme (OpenML 1489), subject to license/provenance verification. Repeated splits of one real dataset are dependent and are reported only as supplementary stress tests. For Crunchbase, call random company splits an **IID working-model benchmark** because shared investors and calendar shocks induce dependence; add investor-cluster and time-cluster stress tests.

**Track B: deployment stress test.** Fit and calibrate using the frozen chronological blocks and test on 2010. Report empirical risk and coverage only. Do not attach a distribution-free guarantee unless a specific covariate-shift/non-exchangeable theorem and its assumptions are verified.

### 15.7 Method go/no-go

Promote the policy to a headline applied contribution only if:

1. the mathematical guarantee is independently checked;
2. the upper Monte Carlo confidence bound for the simultaneous any-constraint violation event is compatible with the theorem’s finite-sample familywise bound;
3. accepted-set sample sizes remain adequate;
4. it improves coverage or utility over margin, entropy, epistemic variance, and selective ensembles;
5. it controls both explanation and prediction risks without near-total abstention; and
6. it generalizes beyond startup data.

Call it a new **method** only if a non-inherited theorem or algorithm is added and independently checked. If any applied-policy condition fails, retain the component as exploratory and publish the core construct audit plus matched-coverage benchmark.

---

## 16. Statistical analysis plan

### 16.1 General principles

- Startup is the primary inferential unit.
- Hyperparameter configurations and random seeds are computational replicates, not independent observations.
- Pairwise model distances share models and cases and therefore are dependent.
- Accepted and reviewed cases are disjoint, not paired.
- Report effect sizes and confidence intervals before \(p\)-values.
- Use equivalence testing for robustness claims.
- Freeze primary and secondary families for multiplicity control.

### 16.2 RQ1

Use a hierarchical paired bootstrap:

1. resample startups;
2. resample full-pipeline refits/configurations within the frozen design;
3. calculate paired cross-specification and within-specification distances on the same cases; and
4. estimate signed \(\Delta_{\text{spec}}\) with its confidence interval and report NACF descriptively only when its stability denominator is valid.

Primary contrasts:

- F18 versus F36;
- F36 versus B+36.

Secondary:

- B+36 versus C+36;
- F36 versus A36;
- strict versus extended feature set.

### 16.3 RQ2

Use the equal-family-pair weighting in RQ2 with a model- and startup-cluster bootstrap or a multi-membership/dyadic mixed model. Do not treat all model pairs as independent or let a larger family dominate.

Report:

- all near-optimal pairs;
- prediction-equivalent pairs;
- identical-hard-prediction pairs; and
- matched-family-size sensitivity; and
- a common-case/matched-margin sensitivity for prediction-equivalent filtering.

### 16.4 RQ3

Use paired startup/model bootstrap differences between:

- top-ranked and random perturbation curves;
- top-ranked and bottom-ranked curves;
- real and randomized models; and
- recovered versus known semi-synthetic signals.

Report agreement among faithfulness metrics rather than selecting the most favourable result.

### 16.5 RQ4 and RQ5

For list changes, use paired bootstrap over startups.

Accepted-versus-reviewed regressions are optional descriptive associations, not the primary gate test. If shown, use an unpaired stratified bootstrap or clustered regression with prespecified covariates:

- true class;
- calibrated margin;
- cohort time;
- angel versus Series-A landmark;
- industry;
- geography; and
- missingness/data-quality indicators.

Avoid causal wording and state that adjustment can condition on selection-related variables. Primary gate comparison remains same-population deployable-threshold and matched-coverage risk.

For gate comparisons, bootstrap paired differences in normalized partial area under risk–coverage curves on \([0.50,0.80]\) and adjust the gate family using Holm or Benjamini–Hochberg as preregistered. Label test-only bootstrap inference conditional on the frozen pipeline.

### 16.6 Equivalence and inconclusive results

Use a three-way interpretation:

1. **material sensitivity:** confidence interval lies above the smallest meaningful effect;
2. **practical equivalence:** confidence interval lies entirely inside the equivalence region; or
3. **inconclusive:** neither criterion is met.

A non-significant difference is never labelled stability.

### 16.7 Provisional smallest meaningful effects

Freeze final values after decision-level simulation using development data only. Starting values:

| Estimand | Provisional meaningful threshold |
|---|---:|
| RQ1 excess normalized attribution distance | 0.10 beyond refit noise |
| RQ2 inter- minus intra-family distance | 0.10 |
| Conditional-deletion improvement over random | 0.10 normalized AOPC |
| Top-\(K\) portfolio overlap change | 0.10 absolute Jaccard change |
| Risk-gate improvement | 20% relative explanation-loss reduction at fixed coverage |

Thresholds must be justified through analyst decisions, not chosen after observing final results.

### 16.8 Multiplicity families

Separate families:

1. two confirmatory RQ1 contrasts;
2. model-family contrasts;
3. faithfulness controls;
4. gate comparisons;
5. secondary outcomes/datasets; and
6. subgroup analyses.

Use Holm for small confirmatory families and Benjamini–Hochberg for larger diagnostic families. Clearly label exploratory analyses.

---

## 17. Power, precision, and go/no-go gates

### 17.1 Endpoint gate

Before final analysis, each confirmatory outcome requires:

- at least 200 development positives;
- at least 50 positives in the probability-calibration block;
- at least 50 positives in the explanation/risk-calibration block;
- at least 50, preferably 80, final-test positives;
- AP lift of at least 1.5 over prevalence in blocked development validation; and
- positive Brier skill.

F18, F36, and B+36 pass the provisional count gate. C+36 does not and remains exploratory.

### 17.2 Explanation gate

Require:

- at least two model families in the global near-optimal set;
- explainer approximation error smaller than the target specification effect;
- a completed, archived development-only RQ1 power curve showing the operating characteristics at the frozen meaningful effect; if precision is inadequate, enlarge the audit/refit budget or relabel the contrast as estimation-focused/exploratory rather than asserting a numeric power target;
- semi-synthetic signal recovery better than random;
- appropriate model/label-randomization response; and
- no headline conclusion driven entirely by one background sample.

### 17.3 Selective-analysis gate

At every confirmatory coverage point:

- at least 100 accepted cases;
- at least 100 reviewed cases for direct group comparisons; 0.90/0.95 coverage may therefore be descriptive only;
- at least 20 positive events per reported class/group, preferably 50;
- finite and stable risk estimates; and
- no silent removal of empty or two-label conformal sets.

### 17.4 Method-extension gate

The risk-control extension requires:

- an independently risk-calibrated, finite pre-frozen candidate family;
- at least 100 accepted risk-calibration cases under the plan’s precision rule; smaller candidates are labelled uncertifiable, never safe;
- prespecified risk budgets;
- valid simultaneous bounds;
- empirical violation-frequency evaluation;
- nontrivial coverage; and
- superiority or clear complementarity relative to simpler gates.

---

## 18. Experiment matrix

| Experiment | Main comparison | Purpose | Status |
|---|---|---|---|
| E0 | Raw versus deduplicated/event-valid data | Source integrity | Mandatory |
| E1 | Strict versus extended feature-time set | Snapshot-metadata sensitivity | Mandatory |
| E2 | F18 versus F36 | Horizon multiverse | Confirmatory |
| E3 | F36 versus B+36 | Milestone-severity multiverse | Confirmatory |
| E3b | Positive-event-count-matched outcome pairs | Scarcity/prevalence envelope | Mandatory sensitivity |
| E4 | B+36 versus C+36 | Rare advanced-stage threshold | Exploratory |
| E5 | F36 versus A36/Broad36 | Cross-family construct difference | Secondary |
| E6 | Same configuration across labels | Isolate outcome specification | Confirmatory |
| E7 | Label-specific tuning | Realistic tuning sensitivity | Secondary |
| E8 | Inter- versus intra-family near-optimal models | Model multiplicity | Confirmatory |
| E9 | Prediction-equivalent model pairs | Remove output disagreement confounding | Confirmatory |
| E10 | Interventional grouped SHAP versus conditional grouped permutation | Correlation/reference sensitivity | Mandatory |
| E11 | Background sample repeats | Explainer uncertainty | Mandatory |
| E12 | Conditional deletion versus random/bottom | Faithfulness | Mandatory |
| E13 | Parameter and label randomization | Explanation sanity | Mandatory |
| E14 | Semi-synthetic known-signal study | Diagnostic validity | Mandatory |
| E15 | Injected leakage feature | Leakage-detection positive control | Mandatory |
| E16 | Top-\(K\) decision overlap and utility | Decision consequence | Confirmatory |
| E17 | Matched-coverage gate benchmark | Empirical selective reliability | Confirmatory |
| E18 | Synthetic IID explanation-risk repetitions | Theorem-behavior validation | Conditional |
| E19 | Chronological explanation-risk transport | Shift stress test | Conditional |
| E20 | PHBench clean-to-leaky feature curve | External leakage stress | Optional secondary |
| E21 | Kaggle snapshot model | Explicit leakage negative control | Optional |

---

## 19. Reproducible notebook and artifact workflow

Run in this order:

1. **00_environment_and_manifest**  
   Package versions, random-seed registry, source URLs, commit hashes, file hashes.

2. **01_source_schema_license_audit**  
   Schemas, row counts, licenses, missingness, impossible dates.

3. **02_event_deduplication_and_cutoff**  
   Duplicate rules, cutoff, event tables.

4. **03_landmark_cohort_and_competing_events**  
   \(t_0\), exclusions, overlap, censoring, multi-state table.

5. **04_feature_time_ledger**  
   Feature source, availability time, transformations, prohibited fields.

6. **05_split_freeze_and_structural_gates**  
   Frozen IDs, event counts, label overlap, analytic precision, and endpoint go/no-go; no explanation-effect power claim yet.

7. **06_blocked_nested_model_selection**  
   Model grids, blocked folds, primary risk, near-optimal membership.

8. **07_probability_calibration**  
   Cross-fitted calibrator selection and frozen calibrated functions.

9. **08_near_optimal_model_registry**  
   Configuration, family, seeds, fold metrics, membership reason.

10. **09_development_explanation_pilot**  
    Development-only groups, backgrounds, approximation-error/runtime pilot, conditional-sampler diagnostics, and control sensitivity.

11. **10_design_power_and_protocol_lock**  
    Development-only explanation-effect power, final budgets, loss formulas, risk budgets, candidate gates, and signed design-lock file.

12. **11_final_predictions_and_attributions**  
    Execute the locked pipeline on 2010; emit immutable long-form predictions and attributions.

13. **12_rq1_outcome_multiverse**  
    F18/F36/B+36 paired distances, refit noise, NACF.

14. **13_rq2_model_family_decomposition**  
    Inter/intra-family and prediction-equivalent analyses.

15. **14_rq3_faithfulness_and_sanity**  
    Conditional perturbation, randomization, semi-synthetic controls.

16. **15_decision_consequence_audit**  
    Portfolio overlap, utility, subgroup composition, case studies.

17. **16_conformal_and_selective_baselines**  
    Deployable 2009-threshold and retrospective matched-coverage gate curves.

18. **17_explanation_selective_risk_policy**  
    Frozen meta-score, LTT/SCRC risk calibration, synthetic IID theorem-behavior track.

19. **18_temporal_and_feature_robustness**  
    Chronological transport, strict/extended features, landmark strata.

20. **19_phbench_external_stress_optional**  
    Fail-graceful clean-to-leaky sensitivity when access permits.

21. **20_inference_figures_and_release**  
    Frozen tables, figures, supplementary results, provenance manifest.

Every notebook declares machine-readable input hashes, output schemas, permitted data blocks, and assertions. It writes a completion manifest; downstream notebooks refuse unmanifested or schema-incompatible inputs. Notebook 11 is impossible to run until the signed output from Notebook 10 exists.

### 19.1 Required frozen files

- source manifest and hashes;
- inclusion/exclusion ledger;
- event-cleaning log;
- feature-time ledger;
- label matrix and overlap table;
- split IDs;
- preprocessing specification;
- model configuration registry;
- near-optimal-set registry;
- long-form predictions;
- long-form grouped attributions;
- explanation-background IDs;
- faithfulness-control results;
- calibration and gate thresholds;
- both signed protocol-lock files;
- preregistered statistical-analysis configuration; and
- final figure/table source data.

### 19.2 Leakage protection

Automated tests must fail if:

- any feature timestamp exceeds \(t_0\);
- any feature lacks an approved availability class;
- any landmark feature is sourced from a row other than the exact aggregated landmark event;
- same-date/type round aggregation or multi-type-earliest-date exclusion is non-deterministic;
- final-test IDs appear in training, tuning, probability calibration, or risk calibration;
- a graph edge dated after \(t_0\) contributes to a feature;
- preprocessing is fitted outside its training fold;
- a test metric is used for model or gate selection; or
- a prohibited snapshot field enters the model matrix;
- a development-only randomized, semi-synthetic, or injected-leakage matrix shares an ID with an approved analysis matrix; or
- an injected-leakage feature appears outside its isolated positive-control registry.

---

## 20. Planned tables and figures

### Main tables

1. Literature-positioning matrix.
2. Dataset provenance, cohort flow, and feature-time ledger summary.
3. Outcome overlap, prevalence, and event counts by time block.
4. Predictive performance and calibration.
5. RQ1 noise-adjusted specification effects.
6. Crossed explanation-uncertainty decomposition.
7. Construct-robust driver taxonomy.
8. Decision consequences at fixed review budgets.
9. Matched-coverage gate comparison.
10. Risk-control results and empirical violation rates, if retained.

### Main figures

1. Construct-to-operationalization diagram and outcome grid.
2. Cohort timeline and four-block split.
3. Explanation multiverse specification curve.
4. Outcome versus model-family explanation-distance heatmap.
5. Variance/uncertainty decomposition.
6. Consensus partial-order or core/contingent driver map.
7. Top-\(K\) portfolio-overlap curves.
8. Prediction- and explanation-risk–coverage curves.
9. Semi-synthetic control results.
10. PHBench clean-to-leaky stress curve.

Avoid decorative architecture diagrams that do not encode a scientific relationship.

---

## 21. Threats to validity and mitigations

| Threat | Consequence | Mitigation and claim restriction |
|---|---|---|
| Crunchbase ascertainment/survivorship | Population excludes invisible firms | Restrict inference to recorded, early-stage Crunchbase firms |
| 2013 vintage | Limited contemporary policy relevance | Present as reproducible methodological proof; seek modern licensed replication |
| Static category/geography | Possible retrospective measurement | Strict event-derived feature sensitivity |
| Backfilled historical events | Prospective availability uncertain | State limitation; use event date as necessary but not sufficient availability |
| Rare C+ and acquisition events | Unstable estimates | Event gates, exploratory designation, wide CIs |
| Acquisition not necessarily success | Construct misinterpretation | Call it recorded acquisition; no value/return claim |
| Nested/competing outcomes | Dependence and changed risk sets | Overlap matrix, common cohort, competing-event sensitivities |
| Similar performance not prediction equivalence | Spurious explanation comparison | Local same-class and probability-tolerance strata |
| Cross-method explanation semantics | Confounded disagreement | Common model-agnostic headline method |
| Correlated features | Arbitrary credit allocation and OOD perturbations | Semantic grouping, conditional sensitivity |
| Faithfulness metrics disagree | Selective reporting risk | Freeze metrics and report discordance |
| Temporal shift | Invalid same-population IID risk guarantee | Separate theorem assumptions/IID working-model checks from empirical chronological transport |
| Shared investors and calendar shocks | Company rows are dependent | Treat random company splits as an IID working-model benchmark; add investor/time-cluster stress tests |
| Data-driven gate selection | Optimistic risk estimates | Independent risk calibration and multiple-testing correction |
| Accepted-set ratio risk | Zero-coding rejection controls the wrong estimand | Test bounded losses only among accepted cases and coverage separately |
| Composite explanation loss | Hidden tradeoffs | Control instability and faithfulness separately |
| No causal identification | Misleading “driver” language | Use predictor/attribution language and decision associations |

---

## 22. Ethics, governance, and responsible reporting

- Do not identify named startups in case studies unless the source license and ethical review clearly permit it; use anonymized IDs by default.
- Do not publish a lead-generation list or operational prospecting tool from historical data.
- Respect PHBench’s usage restrictions and avoid targeting or soliciting companies.
- Do not infer founder race, gender, religion, or other protected characteristics.
- Report geography and industry concentration as a selection-distribution audit.
- Emphasize that historical database coverage may reproduce ecosystem inequities.
- Explain that explanatory consistency does not establish causal truth or fairness.
- Include a model card, data statement, feature-time contract, and intended-use/non-use statement.
- Before finalizing DSS decision parameters, conduct structured elicitation with approximately 5–15 VC analysts, accelerator managers, or startup-screening professionals to validate the landmark, horizons, review budgets, risk tolerances, smallest useful effects, and audit-report format. Treat this as decision-grounding, not statistical proof or representativeness.

Intended use: research on the reliability of predictive explanations and decision-support policies.

Non-use: automated investment, lending, employment, founder ranking, solicitation, or claims about causal determinants of startup success.

---

## 23. Publication strategy

### 23.1 Primary paper: Decision Support Systems

Best fit if the manuscript foregrounds:

- the decision landmark and analyst-review budget;
- outcome-definition governance;
- portfolio/list consequences;
- calibrated uncertainty;
- review workload and utility; and
- actionable reporting of robust versus contingent evidence.

For a competitive DSS submission, include the structured practitioner grounding above. Without a modern event-dated replication, position the 2013 public cohort as a reproducible methodological audit/proof of concept rather than evidence of contemporary operational VC performance.

Recommended title:

> When Startup Success Changes Meaning: A Construct–Rashomon Audit of Predictive Explanations and Selective Venture Screening

### 23.2 Expert Systems with Applications

Strong fallback for the complete applied framework, especially with reproducible pipelines, several model families, faithfulness controls, and PHBench stress testing.

### 23.3 Machine Learning or Data Mining and Knowledge Discovery

Attempt only if the conditional risk-control extension supplies:

- a clearly new formal problem;
- a correct theorem and proof;
- comparison with LTT, conformal risk control, SCRC, confidence, uncertainty gating, and selective ensembles;
- at least 3–5 non-startup benchmark datasets;
- ablations and empirical violation-frequency analysis; and
- a reusable implementation.

### 23.4 FAccT

Possible only if the paper substantially develops:

- target-variable governance;
- stakeholder definitions of “success”;
- harms from proxy targets;
- selection disparities and review burdens; and
- participatory or expert validation of explanation usefulness.

The present data-only design is not yet sufficient for a strong FAccT submission.

---

## 24. Stop, pivot, and fallback rules

### Stop the confirmatory experiment if:

- source hashes or schemas cannot be reproduced;
- leakage-free features collapse below a scientifically meaningful set;
- fewer than two model families pass the skill gate;
- confirmatory endpoints fail event gates;
- explanation controls fail; or
- final-test performance/explanation information or row-level outcome-linked tuning has been used during development outside the sealed count audit.

### Pivot options

1. **Data-method paper:** public Crunchbase event-history audit plus the label-multiverse protocol, without strong contemporary VC claims.
2. **Measurement paper:** construct-validity and decision-consequence audit using fewer model families.
3. **General method paper:** validate Explanation Selective Risk Control on standard tabular datasets, with startup data as one case study.
4. **Leakage paper:** compare strict event-derived, snapshot-static, cumulative-leaky, and PHBench progressively leaky features.

### Null-result rule

A null difference is publishable only when:

- the meaningful-effect threshold was preregistered;
- equivalence testing supports practical similarity;
- intervals are sufficiently narrow;
- positive controls demonstrate audit sensitivity; and
- outcome event counts and predictive skill are adequate.

Otherwise report the result as inconclusive.

---

## 25. Final manuscript claim language

### Use

- “recorded follow-on financing within 36 months”
- “recorded acquisition event”
- “outcome specification”
- “construct-sensitive predictive attribution”
- “empirical near-optimal model set”
- “matched-coverage uncertainty gate”
- “aggregate accepted-set explanation risk”
- “conditional on frozen pre-calibration objects and independent IID risk-calibration/deployment draws”
- “no prior study was identified in the preliminary documented search; retain only if the archived search confirms it”
- “associational predictor attribution”

### Avoid

- “true startup success”
- “failure” for absence of a recorded event
- “drivers” without predictive/associational qualification
- “certified explanation”
- “conformal confidence”
- “guaranteed reliable individual explanation”
- “first in any domain”
- “nobody has done this”
- “both outcomes are equally publishable”
- “causal effect”

---

## 26. Final decision

Proceed with the core paper:

1. fixed early-stage cohort;
2. F18/F36 horizon contrast;
3. F36/B+36 milestone-severity contrast;
4. matched configurations and a common explainer;
5. retraining-noise-adjusted construct fragility;
6. crossed explanation-uncertainty decomposition;
7. faithfulness and semi-synthetic controls;
8. portfolio and review consequences; and
9. a matched-coverage uncertainty-gating benchmark.

Treat acquisition as a secondary different construct, C+ as exploratory, IPO as unavailable, Kaggle as a leakage negative control, and PHBench as a one-label external stress test.

Develop the LTT/SCRC-calibrated explanation-risk policy in parallel, but promote it only after the exact-loss, independence, sample-size, calibration, baseline, and multi-domain gates pass. Call it a new method only if it adds non-inherited theory or an algorithm.

This produces a paper whose central result remains meaningful whether the outcome-specification effect is large, practically equivalent, or heterogeneous—provided the relevant confidence, equivalence, and positive-control criteria are satisfied.

---

## 27. Curated key references

Status labels below distinguish peer-reviewed work from arXiv preprints. All links were checked on 26 August 2026; the submission bibliography must be exported in one consistent journal style with complete DOI/version metadata.

1. Jacobs, A. Z., and Wallach, H. [Measurement and Fairness](https://dl.acm.org/doi/10.1145/3442188.3445901). FAccT, 2021.
2. Steegen, S. et al. [Increasing Transparency Through a Multiverse Analysis](https://journals.sagepub.com/doi/abs/10.1177/1745691616658637). Perspectives on Psychological Science, 2016.
3. Simonsohn, U. et al. [Specification Curve Analysis](https://www.nature.com/articles/s41562-020-0912-z). Nature Human Behaviour, 2020.
4. Jafari, S. M. A. et al. [Deconstructing the Crystal Ball: The SAISE Framework](https://arxiv.org/abs/2508.05491). ArXiv preprint, 2025.
5. Arroyo, J. et al. [Assessment of Machine Learning Performance for Decision Support in Venture Capital Investments](https://doi.org/10.1109/ACCESS.2019.2938659). IEEE Access, 2019.
6. Ross, G., Das, S. R., Sciro, D., and Raza, H. [CapitalVX: A Machine Learning Model for Startup Selection and Exit Prediction](https://www.sciencedirect.com/science/article/pii/S2405918821000040), 2021.
7. Żbikowski, K., and Antosiuk, P. [A Machine Learning, Bias-Free Approach for Predicting Business Success Using Crunchbase Data](https://www.sciencedirect.com/science/article/pii/S0306457321000595), 2021.
8. Kim, J. et al. [How to Succeed in the Market?](https://doi.org/10.1016/j.techfore.2023.122614), 2023.
9. Razaghzadeh Bidgoli, M. et al. [Predicting the Success of Startups Using a Machine Learning Approach](https://link.springer.com/article/10.1186/s13731-024-00436-x), 2024.
10. Maarouf, A. et al. [A Fused Large Language Model for Predicting Startup Success](https://www.sciencedirect.com/science/article/pii/S0377221724007136), 2025.
11. Mashhadi, S. et al. [Interpretable Machine Learning for Predicting Startup Funding, Patenting, and Exits](https://arxiv.org/abs/2510.09465). ArXiv preprint, revised 2026.
12. Ihlamur, Y. et al. [PHBench](https://arxiv.org/abs/2605.02974). ArXiv preprint, 2026.
13. Fisher, A. et al. [Model Class Reliance](https://jmlr.org/papers/v20/18-760.html). JMLR, 2019.
14. Dong, J., and Rudin, C. [Variable Importance Clouds](https://arxiv.org/abs/1901.03209), 2019.
15. Laberge, G. et al. [Partial Order in Chaos](https://jmlr.org/papers/v24/23-0149.html). JMLR, 2023.
16. Donnelly, J., Katta, S., Rudin, C., and Browne, E. [The Rashomon Importance Distribution](https://arxiv.org/abs/2309.13775). ArXiv preprint.
17. Barr, B. et al. [The Disagreement Problem in Faithfulness Metrics](https://arxiv.org/abs/2311.07763), 2023.
18. Adebayo, J. et al. [Sanity Checks for Saliency Maps](https://papers.nips.cc/paper/8160-sanity-checks-for-saliency-maps). NeurIPS, 2018.
19. Black, E. et al. [Selective Ensembles for Consistent Predictions](https://openreview.net/forum?id=HfUyCRBeQc). ICLR, 2022.
20. Paes, L. M. et al. [Selective Explanations](https://proceedings.neurips.cc/paper_files/paper/2024/hash/647af5f6b2538524f6c047c1d9170fd9-Abstract-Conference.html). NeurIPS, 2024.
21. Stradiotti, L. et al. [Knowing What You Cannot Explain](https://arxiv.org/abs/2507.12900). ArXiv preprint, 2025.
22. Zhu, C., Bounia, L., Nguyen, V. L., Destercke, S., and Hoarau, A. [Robust Explanations Through Uncertainty Decomposition: A Path to Trustworthier AI](https://arxiv.org/abs/2507.12913). ArXiv preprint, 2025.
23. Mikriukov, G. et al. [Uncertainty Gating for Cost-Aware XAI](https://arxiv.org/abs/2603.29915). ArXiv preprint, 2026.
24. Marx, C. et al. [But Are You Sure?](https://proceedings.mlr.press/v206/marx23a.html). UAI, 2023.
25. Alkhatib, A., Boström, H., and Johansson, U. [Estimating Quality of Approximated Shapley Values Using Conformal Prediction](https://proceedings.mlr.press/v230/alkhatib24a.html). PMLR 230, 2024.
26. Löfström, T. et al. [Calibrated Explanations for Multi-class](https://proceedings.mlr.press/v230/lofstrom24a.html), 2024.
27. Johansson, U. et al. [Explaining Set-Valued Predictions](https://proceedings.mlr.press/v266/johansson25a.html), 2025.
28. Jin, Y., and Ren, Z. [Confidence on the Focal: Conformal Prediction with Selection-Conditional Coverage](https://doi.org/10.1093/jrsssb/qkaf016). JRSSB 87(4):1239–1259, 2025.
29. Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., and Lei, L. [Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control](https://arxiv.org/abs/2110.01052). ArXiv preprint, 2021.
30. Angelopoulos, A. N. et al. [Conformal Risk Control](https://openreview.net/forum?id=33XGfHLtZg). ICLR, 2024.
31. Xu, Y. et al. [Selective Conformal Risk Control](https://arxiv.org/abs/2512.12844). ArXiv preprint, revised 2026.
32. Tibshirani, R. et al. [Conformal Prediction Under Covariate Shift](https://papers.neurips.cc/paper/8522-conformal-prediction-under-covariate-shift). NeurIPS, 2019.
33. Barber, R. F. et al. [Conformal Prediction Beyond Exchangeability](https://projecteuclid.org/journals/annals-of-statistics/volume-51/issue-2/Conformal-prediction-beyond-exchangeability/10.1214/23-AOS2276.short). Annals of Statistics, 2023.
34. Opoku, J., and Banahene, D. [Signed Evidence Flow: Conflict-Aware and Stability-Calibrated Data Analysis](https://arxiv.org/abs/2606.21875). ArXiv preprint, revised July 2026.
35. Maalej, A., Sönströd, C., and Johansson, U. [Counterfactual Explanations for Conformal Prediction Sets](https://proceedings.mlr.press/v266/maalej25a.html). PMLR 266, 2025.
36. Jin, H., Xue, A., You, W., Goel, S., and Wong, E. [Probabilistic Stability Guarantees for Feature Attributions](https://papers.nips.cc/paper_files/paper/2025/file/67eee231405df68062cf9256a054118d-Paper-Conference.pdf). NeurIPS, 2025.
37. Edin, J. et al. [Normalized AOPC: Fixing Misleading Faithfulness Metrics for Feature Attribution Explainability](https://arxiv.org/abs/2408.08137). ACL, 2025.
38. Datahoarder. [Crunchbase October 2013 Public Mirror](https://github.com/datahoarder/crunchbase-october-2013). Exact execution commit to be recorded by Notebook 00.
39. Khamphukun, R., and Narkbunnum, W. [A Leakage-Controlled, Calibration-First Evaluation of Machine Learning Models for Startup-Outcome Prediction](https://www.mdpi.com/2078-2489/17/7/702). Information 17(7):702, 2026.
40. Meng, X., Liu, J., and Yuan, Y. [Uncertainty-Aware Multi-Task Learning for Startup Success and Valuation Prediction](https://doi.org/10.1016/j.aej.2026.03.030). Alexandria Engineering Journal 141:570–582, 2026.
41. Thackshanaramana B. [Hypothesis Class Determines Explanation: Why Accurate Models Disagree on Feature Attribution](https://arxiv.org/abs/2603.15821). ArXiv preprint submitted to TMLR, 2026.
42. Pan, C., Pan, X., and Sun, L. [A Multi-Layer AI Decision Support System for Startup Success Prediction and Risk Assessment Using Knowledge Graphs and Federated Learning](https://www.nature.com/articles/s41598-026-44162-8). Scientific Reports 16:20259, 2026.
