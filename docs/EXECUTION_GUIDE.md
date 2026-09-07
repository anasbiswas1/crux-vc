# CRUX-VC execution guide

## One-time Colab setup

Place this repository at:

`/content/drive/MyDrive/CRUX_Research/crux-vc`

The notebooks also honor the environment variable `CRUX_REPO_ROOT`. Run notebooks strictly from `00` to `20`. Notebook 11 is deliberately locked until Notebook 10 creates a valid design lock.

## Compute profiles

- `smoke`: code-path validation only; never suitable for paper claims.
- `standard`: development and resource planning; never suitable for locked headline claims.
- `full`: planned final analysis (up to 24 configurations per family, 3 selection seeds, development-locked explanation budgets, and 2,000 power simulations).

Set `CRUX_PROFILE=full` before final protocol locking. Notebook 10 will not issue a final-grade lock under a non-final profile.

## Chunked final explanation execution

Notebook 11 supports independent chunks:

- `CRUX_N_CHUNKS`: total number of model/refit task chunks.
- `CRUX_CHUNK_INDEX`: zero-based chunk index.
- `CRUX_MERGE_ONLY=1`: merge completed chunks after all expected chunk manifests exist.

Each chunk writes immutable Parquet outputs and a hash manifest. Rerunning a completed chunk with different inputs fails unless the output directory is deliberately archived first.

## Protocol signing

Store a strong passphrase in the Colab secret `CRUX_PROTOCOL_SIGNING_KEY` or export it as an environment variable. Do not place it in the notebook or repository.

## GitHub update

After extracting the suite into the existing repository, validate locally or in Colab:

```bash
python -m pip install -e .
python tools/validate_repo.py
pytest
```

Then review `git diff`, commit, and push through your authenticated GitHub workflow.

## Separate smoke and final workspaces

Completion manifests are immutable. Do not run a smoke profile and then overwrite it with a full run in the same generated-output workspace. Use a separate checkout such as `crux-vc-smoke`, or archive all generated data/results/protocol locks before the full run.

## Notebook 17 operational panel

Notebook 17 builds the 2009 operational explanation-loss panel by default. It refits the finite frozen F36 reference panel, computes matched F18/F36/B+36 explanations, runs the fixed conditional-deletion audit, assigns loss 1 to missing or failed explanations, and then calibrates the finite policy family. `CRUX_ESRC_SKIP_FULL_PANEL=1` is diagnostic only. `CRUX_ESRC_THEOREM_CHECKED=1` must not be set until an independent mathematical review confirms the exact implemented theorem and assumptions.
