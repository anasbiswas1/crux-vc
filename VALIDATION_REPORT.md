# CRUX-VC implementation validation report

**Validation date:** 25 August 2026  
**Scope:** Static, structural, unit, and synthetic scientific-invariant validation of the generated repository.  
**Empirical-data status:** The full Crunchbase pipeline was not executed in this build environment because the raw source bytes were unavailable and the runtime had no usable external download path. No source/cohort count is therefore represented here as independently reproduced.

## Validated inventory

- 21 notebooks are present with the exact required names from `00_environment_and_manifest.ipynb` through `20_inference_figures_and_release.ipynb`.
- Every notebook is nbformat 4, has stable cell IDs, has no committed execution output, and has syntactically valid Python code cells.
- Every notebook embeds the permitted data blocks, declared inputs, declared outputs, scientific assertions, protocol version, and central contract reference.
- Reusable implementation is contained in `src/cruxvc`; notebooks orchestrate rather than duplicate most scientific logic.
- The notebook dependency graph, manifest writer, two protocol locks, HMAC verification, and hash-chained test-access log are implemented.

## Test result

The current suite contains **22 passing tests** covering:

1. deterministic event-type normalization, exact duplicate handling, same-date conflict aggregation, administrative cutoff enforcement, and exact 18/36-month labels;
2. landmark eligibility, competing acquisition rules, and nested outcome invariants;
3. strict/extended feature construction, pre-landmark investor features, feature registry enforcement, and prohibited future-feature failure;
4. deterministic model fitting, calibrated-probability wrappers, and configuration IDs;
5. grouped explanation distances, noise-adjusted RQ1 calculations, equivalence/material/inconclusive interpretation, and risk–coverage integration;
6. finite-sample conformal sets and deterministic aligned cross-fitted standard/Mondrian gate scores;
7. accepted-set finite-policy calibration with separate coverage constraints;
8. ESRC finite-panel instability, construct-fragility losses, and worst-loss treatment of a missing panel explanation;
9. HMAC protocol-lock tamper detection and final-test access-log chain/tamper detection;
10. exact-size proportional stratified sampling, frozen audit-design weighting, and release identifier scanning;
11. non-extrapolating partial risk–coverage integration and collapsed-cell composition arithmetic;
12. machine-readable notebook-contract consistency.

Executed commands:

```bash
python -m compileall -q src tools tests
python tools/validate_notebooks.py
python tools/validate_repo.py
PYTHONPATH=src pytest -q
```

Observed result:

```text
Validated 21 CRUX-VC notebooks: inventory, syntax, metadata, and clean outputs.
CRUX-VC repository validation passed.
22 passed
```

The host validation environment used Python 3.13.5 and did not include `pyarrow`; therefore it was suitable for source compilation and synthetic tests, but not for executing the Parquet-based end-to-end notebook workflow. The repository pins its supported execution stack in `requirements.txt`; the final Colab run must install that file before execution.

## What remains deliberately unverified until execution

The following are not claimed by this report:

- resolution of the actual checked-out source commit;
- successful reproduction of the four source SHA-256 values;
- raw source row counts, deterministic clean event counts, the 4,234-company cohort, or time-block event counts;
- endpoint skill gates, near-optimal model-family membership, calibration performance, explanation power, or runtime adequacy;
- any 2010 prediction, attribution, RQ1/RQ2 effect, faithfulness, decision, uncertainty-gating, or ESRC result;
- PHBench or Kaggle external-control availability; or
- validity of a new theorem for ESRC.

Notebooks `00`–`05` are responsible for source/count reproduction and fail when verified-source expectations differ. Notebook `10` cannot create a final-grade lock outside the full profile. Notebook `11` cannot access the final test without both protocol locks. Notebook `17` withholds any theorem-backed claim unless the full operational panel passes and an independent theorem check is explicitly recorded.

## Release assessment

The implementation package is ready to upload to the repository and execute. Scientific submission readiness remains contingent on the full locked run, all go/no-go gates, review of generated manifests and deviations, and manuscript claims being populated only from frozen outputs.
