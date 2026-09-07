# Protocol controls

The pipeline creates two immutable protocol locks:

1. `phase0_lock.json` after source, cohort, labels, feature-time ledger, split IDs, gates, and preregistration are frozen.
2. `design_lock.json` after development-only explanation pilots, runtime/precision studies, power simulation, losses, risk budgets, and candidate gates are frozen.

For final execution, set `CRUX_PROTOCOL_SIGNING_KEY` as a Colab secret or environment variable. The notebooks create HMAC-SHA256 signatures without writing the key. Notebook 11 refuses access to the final-test analysis when the configured final profile is not active, either lock is invalid, or HMAC signing is required but unavailable.

`test_access_log.jsonl` is a hash-chained audit log. It records every code path that requests final-test labels, predictions, or explanations.
