# CRUX-VC

**Construct–Rashomon Uncertainty Examination for Venture Capital**

CRUX-VC is the executable research workflow for the study **“When Startup Success Changes Meaning: A Construct–Rashomon Audit of Predictive Explanations and Selective Venture Screening.”** It implements the 21-stage protocol in the master research plan: source reconstruction, landmark cohort and outcome multiverse, leakage-controlled feature engineering, blocked model selection, calibration, grouped explanations, outcome/model uncertainty decomposition, faithfulness controls, decision consequences, matched-coverage uncertainty gating, and the conditional ESRC risk-control extension.

**Author/contact:** Md Anas Biswas — `up2082724@myport.ac.uk`  
**Repository:** `https://github.com/anasbiswas1/crux-vc`  
**Default Google Drive checkout:** `/content/drive/MyDrive/CRUX_Research/crux-vc`

## Status

The implementation is complete and organized as notebooks `00`–`20` plus reusable Python modules and tests. The repository intentionally contains **no empirical result claims and no Crunchbase source data**. Expected source/cohort counts in the configuration are executable assertions derived from the research plan; they become reproduced observations only after Notebooks `00`–`05` run successfully against hash-matching source files.

The core paper remains valid independently of the conditional ESRC extension. Notebook `17` now constructs the operational 2009 explanation-loss panel directly from frozen pre-calibration objects; it does not substitute conformal set size or a proxy score for explanation reliability. Any formal guarantee remains disabled unless the exact theorem/procedure has been independently checked.

## Repository structure

```text
config/       Frozen project, model, feature-group, statistical, and notebook contracts
notebooks/    21 Colab-ready notebooks, executed in numerical order
src/cruxvc/   Reusable scientific implementation
protocol/     Source expectations, preregistration template, and generated locks
results/      Generated manifests, models, predictions, attributions, controls, and release files
data/         Raw/interim/processed/external data locations; source data are git-ignored
docs/         Execution, methods, data-dictionary, and master-plan documentation
tests/        Unit and synthetic scientific-invariant tests
tools/        Notebook/repository validation and execution helpers
```

## Execution profiles

| Profile | Purpose | Paper claims |
|---|---|---|
| `smoke` | Fast code-path checking with reduced grids/refits/explanation budgets | Prohibited |
| `standard` | Development, debugging, and resource planning | Prohibited for locked headline results |
| `full` | Prespecified final analysis, including 100 headline bootstrap refits and 2,000 RQ1 power simulations | Required |

Do not run smoke and full experiments in the same generated-output workspace. Completion manifests are intentionally immutable. Use a separate checkout such as `crux-vc-smoke`, or archive the smoke workspace before starting the full execution.

## Colab setup

1. Place or clone the repository at:

   ```text
   MyDrive/CRUX_Research/crux-vc
   ```

2. Add a strong Colab secret named `CRUX_PROTOCOL_SIGNING_KEY`. Do not type the key into a notebook, commit it, or save it in Drive as a plain-text file.

3. For the final run, set:

   ```python
   import os
   os.environ["CRUX_PROFILE"] = "full"
   ```

4. Open and run Notebooks `00` through `10` in order. Notebook `10` issues the second, final-grade HMAC protocol lock only under the `full` profile.

5. Execute Notebook `11` in chunks. For example, with 20 chunks, run each chunk in a separate Colab session by setting:

   ```python
   import os
   os.environ["CRUX_PROFILE"] = "full"
   os.environ["CRUX_N_CHUNKS"] = "20"
   os.environ["CRUX_CHUNK_INDEX"] = "0"  # change through 19
   ```

   After every chunk manifest exists, run Notebook `11` once more with:

   ```python
   os.environ["CRUX_MERGE_ONLY"] = "1"
   ```

6. Run Notebooks `12`–`20` in numerical order. Notebook `17` builds the full 2009 ESRC operational-loss panel by default. `CRUX_ESRC_SKIP_FULL_PANEL=1` is only a diagnostic escape hatch and forces “not certifiable” output labelling.

## Data-source contract

Notebook `00` checks out the public October 2013 Crunchbase mirror, resolves the actual Git commit, verifies the four expected CSV hashes, and copies verified files into the stable raw-data directory. The configured expected SHA-256 values are:

| File | SHA-256 |
|---|---|
| `crunchbase-companies.csv` | `83217e8165db15124aec2799f5f28bda910ab09cc1ed9553b066eb97f60f195d` |
| `crunchbase-rounds.csv` | `1b27a2b561678fe6257556d729819edc98a9581b18d6d8fad82413658e055212` |
| `crunchbase-investments.csv` | `8f13f28d35915633c0a814dd22d49540ee115717d10d387ebc3a49183a35b65f` |
| `crunchbase-acquisitions.csv` | `c3fb008fa75c48cafea7bdbd4eca050c11a55383d558998d349433f201b9f779` |

A mismatch is a hard failure under the default strict configuration. Do not bypass the hash gate to make expected counts appear to match.

## Scientific safeguards

The implementation enforces the following controls:

- deterministic event collapse and exact-date horizon labels;
- a frozen feature-time ledger with prohibited-field checks;
- graph and market features truncated before each company landmark;
- disjoint 2005–2007 development, 2008 probability-calibration, 2009 risk-calibration, and 2010 test blocks;
- fold-local preprocessing and no outcome-specific silent resampling in the primary analysis;
- common grouped attribution semantics across model families;
- matched cases/configurations/backgrounds/seeds for the primary construct contrasts;
- frozen-stratum, inverse-inclusion-weighted hierarchical inference rather than pseudo-replication;
- parameter/label randomization, semi-synthetic recovery, and isolated leakage controls;
- cross-fitted 2009 conformal gate scores, same-family bootstrap epistemic variance, and separate cross-family disagreement;
- accepted-set losses evaluated among accepted cases, with coverage tested separately;
- HMAC protocol locks and a hash-chained final-test access log;
- secret-salted case-study identifiers and a release identifier scan.

## Local validation

Use Python 3.10–3.12 for the supported environment.

```bash
python -m pip install -e ".[dev,models,xai]"
python tools/validate_repo.py
pytest -q
```

The static validator checks repository inventory, notebook contracts, stable cell IDs, clean notebook outputs, and Python syntax. Tests use synthetic data and do not require the Crunchbase files.

## Generated release

Notebook `20` creates:

```text
results/release/CRUX_VC_reproducibility_release.zip
```

The release contains code, notebooks, tests, documentation, protocol metadata, frozen result tables, and figures. It excludes raw/intermediate data, signing keys, the final-test access log, and direct startup identifiers.

## Responsible use

CRUX-VC is for research on construct validity, predictive-explanation reliability, and decision-support auditing. It is not an automated investment, lending, employment, founder-ranking, solicitation, or causal-inference system. Historical Crunchbase coverage is incomplete and may reproduce ecosystem visibility biases.

## Licence

The CRUX-VC code is released under the MIT License. Third-party datasets retain their own terms. The Crunchbase mirror and any optional PHBench/Kaggle materials are not relicensed or bundled by this repository.
