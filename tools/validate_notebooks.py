#!/usr/bin/env python
"""Validate notebook inventory, metadata, outputs, and Python syntax."""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = [
    "00_environment_and_manifest.ipynb",
    "01_source_schema_license_audit.ipynb",
    "02_event_deduplication_and_cutoff.ipynb",
    "03_landmark_cohort_and_competing_events.ipynb",
    "04_feature_time_ledger.ipynb",
    "05_split_freeze_and_structural_gates.ipynb",
    "06_blocked_nested_model_selection.ipynb",
    "07_probability_calibration.ipynb",
    "08_near_optimal_model_registry.ipynb",
    "09_development_explanation_pilot.ipynb",
    "10_design_power_and_protocol_lock.ipynb",
    "11_final_predictions_and_attributions.ipynb",
    "12_rq1_outcome_multiverse.ipynb",
    "13_rq2_model_family_decomposition.ipynb",
    "14_rq3_faithfulness_and_sanity.ipynb",
    "15_decision_consequence_audit.ipynb",
    "16_conformal_and_selective_baselines.ipynb",
    "17_explanation_selective_risk_policy.ipynb",
    "18_temporal_and_feature_robustness.ipynb",
    "19_phbench_external_stress_optional.ipynb",
    "20_inference_figures_and_release.ipynb",
]


def main() -> None:
    notebook_dir = ROOT / "notebooks"
    contracts = yaml.safe_load((ROOT / "config" / "notebook_contracts.yaml").read_text(encoding="utf-8"))
    actual = sorted(path.name for path in notebook_dir.glob("*.ipynb"))
    errors = []
    if actual != EXPECTED:
        errors.append(f"Notebook inventory mismatch. Expected {EXPECTED}, found {actual}")
    for name in EXPECTED:
        path = notebook_dir / name
        if not path.exists():
            continue
        notebook = nbformat.read(path, as_version=4)
        if notebook.nbformat != 4:
            errors.append(f"{name}: nbformat is not 4")
        if not notebook.cells or notebook.cells[0].cell_type != "markdown":
            errors.append(f"{name}: first cell must be markdown")
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        if len(code_cells) < 3:
            errors.append(f"{name}: too few executable cells ({len(code_cells)})")
        for index, cell in enumerate(code_cells):
            if cell.get("outputs"):
                errors.append(f"{name}: code cell {index} contains committed outputs")
            if cell.get("execution_count") is not None:
                errors.append(f"{name}: code cell {index} has execution_count")
            source = cell.source
            if re.search(r"(^|\n)\s*[!%]", source):
                errors.append(f"{name}: code cell {index} contains a shell/IPython magic")
                continue
            try:
                ast.parse(source, filename=f"{name}:cell{index}")
            except SyntaxError as exc:
                errors.append(f"{name}: code cell {index} syntax error: {exc}")
        metadata = notebook.metadata.get("cruxvc", {})
        expected_stage = name[:2]
        if metadata.get("stage_id") != expected_stage:
            errors.append(f"{name}: missing/incorrect cruxvc.stage_id")
        expected_contract = contracts[expected_stage]
        checks = {
            "permitted_blocks": expected_contract["permitted_blocks"],
            "declared_inputs": expected_contract["inputs"],
            "declared_outputs": expected_contract["outputs"],
            "assertions": expected_contract["assertions"],
        }
        for field, expected_value in checks.items():
            if list(metadata.get(field, [])) != list(expected_value):
                errors.append(f"{name}: cruxvc.{field} does not match notebook contract")
        if metadata.get("contract_ref") != f"config/notebook_contracts.yaml#{expected_stage}":
            errors.append(f"{name}: missing/incorrect contract_ref")
        for index, cell in enumerate(notebook.cells):
            if not cell.get("id"):
                errors.append(f"{name}: cell {index} has no stable nbformat id")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(f"Validated {len(EXPECTED)} CRUX-VC notebooks: inventory, syntax, metadata, and clean outputs.")


if __name__ == "__main__":
    main()
