"""Cross-notebook registries and chunk-safe final execution helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from .calibration import CalibratedModel, PlattCalibrator
from .explanations import active_feature_groups, explain_dataset
from .hashing import hash_jsonable
from .io import read_json, read_table, write_json, write_table
from .models import fit_model, predict_positive


def seed_registry(base_seed: int, counts: Mapping[str, int]) -> pd.DataFrame:
    sequence = np.random.SeedSequence(base_seed)
    total = sum(int(value) for value in counts.values())
    children = sequence.spawn(total)
    rows = []
    cursor = 0
    for purpose, count in counts.items():
        for index in range(int(count)):
            rows.append(
                {
                    "purpose": purpose,
                    "index": index,
                    "seed": int(children[cursor].generate_state(1, dtype=np.uint32)[0]),
                }
            )
            cursor += 1
    return pd.DataFrame(rows)


def build_refit_registry(
    selected_configs: pd.DataFrame,
    outcomes: Sequence[str],
    bootstrap_refits: int,
    seed_table: pd.DataFrame,
) -> pd.DataFrame:
    refit_seeds = seed_table[seed_table["purpose"].eq("bootstrap_refit")].sort_values("index")
    if len(refit_seeds) < bootstrap_refits:
        raise ValueError("Seed registry is smaller than requested bootstrap-refit count")
    rows = []
    for config in selected_configs.itertuples(index=False):
        for outcome in outcomes:
            for refit_slot, seed in enumerate(refit_seeds["seed"].head(bootstrap_refits)):
                rows.append(
                    {
                        "task_id": f"{outcome}__{config.family}__{config.config_id}__r{refit_slot:03d}",
                        "outcome": outcome,
                        "family": config.family,
                        "config_id": config.config_id,
                        "parameters_json": config.parameters_json,
                        "refit_id": f"bootstrap_{refit_slot:03d}",
                        "refit_slot": refit_slot,
                        "seed": int(seed),
                    }
                )
    return pd.DataFrame(rows)


def assign_task_chunks(registry: pd.DataFrame, n_chunks: int) -> pd.DataFrame:
    if n_chunks < 1:
        raise ValueError("n_chunks must be positive")
    result = registry.sort_values("task_id").reset_index(drop=True).copy()
    result["chunk_index"] = np.arange(len(result)) % n_chunks
    return result


def bootstrap_indices(case_ids: Sequence[int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.asarray(case_ids)
    return rng.choice(values, size=len(values), replace=True)


def fit_refit_calibrated_model(
    training: pd.DataFrame,
    probability_calibration: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    outcome: str,
    family: str,
    parameters: Mapping[str, Any],
    seed: int,
) -> CalibratedModel:
    sampled_ids = bootstrap_indices(training["case_id"], seed)
    sampled = training.set_index("case_id").loc[sampled_ids].reset_index()
    model = fit_model(sampled, sampled[outcome], feature_columns, family, parameters, seed)
    raw = predict_positive(model, probability_calibration, feature_columns)
    calibrator = PlattCalibrator().fit(raw, probability_calibration[outcome].to_numpy(dtype=int))
    return CalibratedModel(model, calibrator, tuple(feature_columns))


def run_final_task_chunk(
    tasks: pd.DataFrame,
    training: pd.DataFrame,
    probability_calibration: pd.DataFrame,
    test_cases: pd.DataFrame,
    audit_cases: pd.DataFrame,
    backgrounds: Mapping[str, pd.DataFrame],
    groups: Mapping[str, Sequence[str]],
    feature_columns: Sequence[str],
    approximation_seeds: Sequence[int],
    n_orderings: int,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_records = []
    attribution_parts = []
    active_groups = active_feature_groups(groups, feature_columns)
    for task in tasks.itertuples(index=False):
        task_type = getattr(task, "task_type", "bootstrap")
        parameters = json.loads(task.parameters_json)
        if task_type == "deployment":
            calibrated_path = Path(getattr(task, "calibrated_model_path"))
            if not calibrated_path.exists():
                raise FileNotFoundError(calibrated_path)
            model = joblib.load(calibrated_path)
            # Only the small matched-reference deployment panel is explained on
            # every 2010 case. The larger label-specific near-optimal set is the
            # RQ2 local-audit multiverse and therefore uses the frozen weighted
            # audit sample, preventing an unnecessary all-test compute explosion.
            role = getattr(task, "analysis_role", "deployment")
            explanation_cases = (
                test_cases if role == "matched_reference_deployment" else audit_cases
            )
        else:
            model = fit_refit_calibrated_model(
                training,
                probability_calibration,
                feature_columns=feature_columns,
                outcome=task.outcome,
                family=task.family,
                parameters=parameters,
                seed=int(task.seed),
            )
            explanation_cases = audit_cases
        probability = model.predict_proba(test_cases)[:, 1]
        prediction_records.append(
            pd.DataFrame(
                {
                    "case_id": test_cases["case_id"].to_numpy(),
                    "outcome": task.outcome,
                    "family": task.family,
                    "config_id": task.config_id,
                    "model_id": task.task_id,
                    "refit_id": task.refit_id,
                    "task_type": task_type,
                    "analysis_role": getattr(task, "analysis_role", task_type),
                    "probability": probability,
                    "predicted_class": (probability >= 0.5).astype(int),
                    "y_true": test_cases[task.outcome].to_numpy(dtype=int),
                }
            )
        )
        for background_id, background in backgrounds.items():
            for approximation_seed in approximation_seeds:
                explained = explain_dataset(
                        model,
                        explanation_cases,
                        background,
                        active_groups,
                        feature_columns,
                        model_metadata={
                            "outcome": task.outcome,
                            "family": task.family,
                            "config_id": task.config_id,
                            "model_id": task.task_id,
                            "refit_id": task.refit_id,
                            "task_type": task_type,
                            "analysis_role": getattr(task, "analysis_role", task_type),
                        },
                        background_id=background_id,
                        approximation_seed=int(approximation_seed),
                        n_orderings=n_orderings,
                    )
                # Preserve the frozen audit design information for weighted RQ1
                # and RQ2 inference. Full-test deployment explanations receive
                # unit weights and a dedicated stratum label.
                case_metadata_columns = [
                    column
                    for column in [
                        "case_id",
                        "joint_outcome_pattern",
                        "audit_stratum",
                        "design_weight",
                        "audit_inclusion_probability",
                    ]
                    if column in explanation_cases.columns
                ]
                if "case_id" in case_metadata_columns:
                    metadata = explanation_cases[case_metadata_columns].drop_duplicates("case_id")
                    explained = explained.merge(metadata, on="case_id", how="left", validate="many_to_one")
                if "design_weight" not in explained.columns:
                    explained["design_weight"] = 1.0
                if "audit_inclusion_probability" not in explained.columns:
                    explained["audit_inclusion_probability"] = 1.0
                if "audit_stratum" not in explained.columns:
                    explained["audit_stratum"] = "full_test"
                attribution_parts.append(explained)
    predictions = pd.concat(prediction_records, ignore_index=True) if prediction_records else pd.DataFrame()
    attributions = pd.concat(attribution_parts, ignore_index=True) if attribution_parts else pd.DataFrame()
    task_hash = hash_jsonable(tasks.to_dict("records"))[:16]
    prediction_path = output_dir / f"predictions_{task_hash}.parquet"
    attribution_path = output_dir / f"attributions_{task_hash}.parquet"
    write_table(predictions, prediction_path, metadata={"task_hash": task_hash})
    write_table(attributions, attribution_path, metadata={"task_hash": task_hash})
    write_json(
        {
            "task_hash": task_hash,
            "task_ids": tasks["task_id"].tolist(),
            "prediction_path": str(prediction_path),
            "attribution_path": str(attribution_path),
        },
        output_dir / f"chunk_{task_hash}.json",
    )
    return prediction_path, attribution_path


def merge_chunk_outputs(chunk_dir: str | Path, output_predictions: str | Path, output_attributions: str | Path) -> tuple[Path, Path]:
    chunk_dir = Path(chunk_dir)
    prediction_files = sorted(chunk_dir.glob("predictions_*.parquet"))
    attribution_files = sorted(chunk_dir.glob("attributions_*.parquet"))
    if not prediction_files or not attribution_files:
        raise RuntimeError(f"No chunk outputs found in {chunk_dir}")
    predictions = pd.concat([read_table(path) for path in prediction_files], ignore_index=True)
    attributions = pd.concat([read_table(path) for path in attribution_files], ignore_index=True)
    if predictions.duplicated(["case_id", "model_id", "outcome"]).any():
        raise RuntimeError("Duplicate prediction keys across chunks")
    key = [
        "case_id",
        "model_id",
        "outcome",
        "refit_id",
        "background_id",
        "approximation_seed",
        "feature_group",
        "explanation_method",
    ]
    key = [column for column in key if column in attributions.columns]
    if attributions.duplicated(key).any():
        raise RuntimeError("Duplicate attribution keys across chunks")
    write_table(predictions, output_predictions)
    write_table(attributions, output_attributions)
    return Path(output_predictions), Path(output_attributions)
