"""Strictly align model predictions to benchmark labels and calculate scores."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pv_metrics import evaluate_forecast

from .protocol import sha256_file, validate_benchmark
from .spec import TaskSpec, get_task_spec


def validate_prediction_frame(frame: pd.DataFrame, spec: TaskSpec) -> None:
    missing = sorted({"row_id", "prediction"} - set(frame.columns))
    if missing:
        raise ValueError(f"prediction data is missing columns: {missing}")
    if frame.empty:
        raise ValueError("prediction data must not be empty")
    if frame["row_id"].isna().any() or frame["row_id"].astype(str).duplicated().any():
        raise ValueError("prediction row_id must be non-empty and unique")
    try:
        arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in frame["prediction"]]
    except (TypeError, ValueError) as error:
        raise ValueError("prediction column contains invalid numeric arrays") from error
    bad = [index for index, value in enumerate(arrays) if len(value) != spec.horizon_points]
    if bad:
        raise ValueError(
            f"prediction arrays must contain {spec.horizon_points} points; invalid rows: {bad[:5]}"
        )
    if not np.isfinite(np.concatenate(arrays)).all():
        raise ValueError("prediction arrays contain NaN or infinite values")
    if "model_id" in frame:
        model_ids = frame["model_id"].dropna().astype(str).unique()
        if len(model_ids) > 1:
            raise ValueError("one prediction file must contain at most one model_id")


def score_benchmark(
    benchmark_dir: str | Path,
    predictions: str | Path | pd.DataFrame,
    *,
    split: str = "test",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Score predictions after exact row_id alignment with a locked split."""

    benchmark_root = Path(benchmark_dir)
    manifest = validate_benchmark(benchmark_root)
    if split not in {"validation", "test"}:
        raise ValueError("score split must be 'validation' or 'test'")
    spec = get_task_spec(manifest["task"]["name"])
    label_path = benchmark_root / manifest["files"][split]["file"]
    labels = pd.read_parquet(label_path)
    if isinstance(predictions, pd.DataFrame):
        prediction_frame = predictions.copy()
        prediction_hash = None
    else:
        prediction_path = Path(predictions)
        prediction_frame = pd.read_parquet(prediction_path)
        prediction_hash = sha256_file(prediction_path)
    validate_prediction_frame(prediction_frame, spec)

    label_ids = labels["row_id"].astype(str)
    prediction_ids = prediction_frame["row_id"].astype(str)
    missing = sorted(set(label_ids) - set(prediction_ids))
    extra = sorted(set(prediction_ids) - set(label_ids))
    if missing or extra:
        raise ValueError(
            f"prediction row_id set does not match benchmark {split} split: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    prediction_lookup = prediction_frame.assign(row_id=prediction_ids).set_index("row_id")
    ordered_prediction = np.stack(
        prediction_lookup.loc[label_ids, "prediction"].map(
            lambda value: np.asarray(value, dtype=np.float64)
        )
    )
    groundtruth = np.stack(labels["target"].map(lambda value: np.asarray(value, dtype=np.float64)))
    result = evaluate_forecast(
        spec.name,
        groundtruth,
        ordered_prediction,
        labels["capacity"].to_numpy(dtype=np.float64),
        capacity_floor_ratio=spec.capacity_floor_ratio,
    )
    details = labels[
        ["row_id", "station", "timestamp_win", "target_start", "target_end", "capacity"]
    ].copy()
    details["normalized_error"] = result.per_sample_normalized_error
    details["accuracy"] = result.per_sample_accuracy
    by_station = details.groupby("station", as_index=False)["accuracy"].mean()
    model_id = None
    if "model_id" in prediction_frame:
        values = prediction_frame["model_id"].dropna().astype(str).unique()
        model_id = values[0] if len(values) else None
    summary: dict[str, Any] = {
        "protocol_version": manifest["protocol_version"],
        "benchmark_id": manifest["benchmark_id"],
        "task": spec.name,
        "split": split,
        "model_id": model_id,
        "primary_metric": "accuracy",
        "aggregation": spec.aggregation,
        "capacity_floor_ratio": spec.capacity_floor_ratio,
        "samples": len(labels),
        "normalized_error": result.normalized_error,
        "accuracy": result.accuracy,
        "accuracy_percent": result.accuracy_percent,
        "station_macro_accuracy": float(by_station["accuracy"].mean()),
        "worst_station_accuracy": float(by_station["accuracy"].min()),
        "data_sha256": manifest["files"][split]["sha256"],
        "predictions_sha256": prediction_hash,
    }
    return summary, details
