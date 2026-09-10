"""Write, load, hash, and validate immutable benchmark datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .spec import (
    ARRAY_LENGTHS_BASE,
    PROTOCOL_VERSION,
    TASK_COLUMNS,
    TaskName,
    TaskSpec,
    get_task_spec,
)


MANIFEST_NAME = "benchmark.json"
SPLITS = ("train", "validation", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stack_numeric(frame: pd.DataFrame, column: str, expected_length: int) -> None:
    try:
        values = [np.asarray(value, dtype=np.float64).reshape(-1) for value in frame[column]]
    except (TypeError, ValueError) as error:
        raise ValueError(f"column {column!r} contains invalid numeric arrays") from error
    invalid_lengths = [index for index, value in enumerate(values) if len(value) != expected_length]
    if invalid_lengths:
        raise ValueError(
            f"column {column!r} must contain {expected_length} points; invalid rows: {invalid_lengths[:5]}"
        )
    if values and not np.isfinite(np.concatenate(values)).all():
        raise ValueError(f"column {column!r} contains NaN or infinite values")


def validate_task_frame(frame: pd.DataFrame, spec: TaskSpec, *, split: str | None = None) -> None:
    missing = sorted(set(TASK_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"benchmark data is missing columns: {missing}")
    if frame.empty:
        raise ValueError("benchmark split must not be empty")
    if frame["row_id"].isna().any() or frame["row_id"].astype(str).duplicated().any():
        raise ValueError("row_id must be non-empty and unique")
    if not frame["task"].eq(spec.name).all():
        raise ValueError(f"task column must contain only {spec.name!r}")
    if split is not None and not frame["split"].eq(split).all():
        raise ValueError(f"split column must contain only {split!r}")

    capacity = frame["capacity"].to_numpy(dtype=np.float64)
    if not np.isfinite(capacity).all() or (capacity <= 0).any():
        raise ValueError("capacity must contain finite values greater than zero")
    for column, length in ARRAY_LENGTHS_BASE.items():
        _stack_numeric(frame, column, length)
    for column in [
        "ghi_forecast",
        "temperature_forecast",
        "wind_speed_forecast",
        "wind_direction_forecast",
        "target",
    ]:
        _stack_numeric(frame, column, spec.horizon_points)

    origin = pd.to_datetime(frame["timestamp_win"], errors="coerce")
    start = pd.to_datetime(frame["target_start"], errors="coerce")
    end = pd.to_datetime(frame["target_end"], errors="coerce")
    if origin.isna().any() or start.isna().any() or end.isna().any():
        raise ValueError("timestamp_win, target_start, and target_end must be valid timestamps")
    if task_schedule_mask(spec, origin, start, end).all():
        return
    raise ValueError(f"one or more rows violate the {spec.name} issue/target schedule")


def task_schedule_mask(
    spec: TaskSpec,
    origin: pd.Series,
    start: pd.Series,
    end: pd.Series,
) -> pd.Series:
    if spec.name == "ultra_short":
        return (origin.dt.minute % 15 == 0) & (start == origin + pd.Timedelta(minutes=15)) & (
            end == origin + pd.Timedelta(hours=4)
        )
    next_day = origin.dt.normalize() + pd.Timedelta(days=1)
    return (
        (origin.dt.hour == 10)
        & (origin.dt.minute == 0)
        & (start == next_day)
        & (end == next_day + pd.Timedelta(hours=23, minutes=45))
    )


def _split_frame(
    frame: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    origin = pd.to_datetime(frame["timestamp_win"])
    masks = {
        "train": origin <= train_end,
        "validation": (origin > train_end) & (origin <= validation_end),
        "test": origin > validation_end,
    }
    result: dict[str, pd.DataFrame] = {}
    for split, mask in masks.items():
        selected = frame.loc[mask].copy()
        if selected.empty:
            raise ValueError(f"time boundaries produce an empty {split!r} split")
        selected["split"] = split
        result[split] = selected.reset_index(drop=True)
    return result


def write_benchmark(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    task: TaskName,
    train_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
) -> dict[str, Any]:
    """Write versioned train/validation/test splits plus a hash manifest."""

    spec = get_task_spec(task)
    train_boundary = pd.Timestamp(train_end)
    validation_boundary = pd.Timestamp(validation_end)
    if train_boundary >= validation_boundary:
        raise ValueError("train_end must be earlier than validation_end")
    validate_task_frame(frame, spec)
    splits = _split_frame(
        frame,
        train_end=train_boundary,
        validation_end=validation_boundary,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    all_row_ids: set[str] = set()
    for split, selected in splits.items():
        validate_task_frame(selected, spec, split=split)
        row_ids = set(selected["row_id"].astype(str))
        if all_row_ids.intersection(row_ids):
            raise ValueError("row_id values overlap between splits")
        all_row_ids.update(row_ids)
        path = root / f"{split}.parquet"
        selected.to_parquet(path, index=False)
        files[split] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "rows": len(selected),
            "stations": sorted(selected["station"].astype(str).unique().tolist()),
            "origin_start": pd.Timestamp(selected["timestamp_win"].min()).isoformat(),
            "origin_end": pd.Timestamp(selected["timestamp_win"].max()).isoformat(),
        }
    benchmark_identity = json.dumps(
        {
            "protocol_version": PROTOCOL_VERSION,
            "task": task,
            "files": {name: item["sha256"] for name, item in files.items()},
        },
        sort_keys=True,
    )
    benchmark_id = hashlib.sha256(benchmark_identity.encode("utf-8")).hexdigest()[:24]
    manifest: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_id": benchmark_id,
        "task": spec.as_dict(),
        "schema": {
            "columns": TASK_COLUMNS,
            "array_lengths": {
                **ARRAY_LENGTHS_BASE,
                "ghi_forecast": spec.horizon_points,
                "temperature_forecast": spec.horizon_points,
                "wind_speed_forecast": spec.horizon_points,
                "wind_direction_forecast": spec.horizon_points,
                "target": spec.horizon_points,
            },
            "prediction_columns": ["row_id", "prediction", "model_id"],
        },
        "split_policy": {
            "type": "forecast_origin_time",
            "train_end_inclusive": train_boundary.isoformat(),
            "validation_end_inclusive": validation_boundary.isoformat(),
            "test_rule": "timestamp_win > validation_end_inclusive",
        },
        "files": files,
    }
    manifest_path = root / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_benchmark(benchmark_dir: str | Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    root = Path(benchmark_dir)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"benchmark manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            f"unsupported protocol version {manifest.get('protocol_version')!r}; expected {PROTOCOL_VERSION!r}"
        )
    splits = {
        split: pd.read_parquet(root / manifest["files"][split]["file"])
        for split in SPLITS
    }
    return manifest, splits


def validate_benchmark(benchmark_dir: str | Path) -> dict[str, Any]:
    """Validate file hashes, schemas, schedules, and split isolation."""

    root = Path(benchmark_dir)
    manifest, splits = load_benchmark(root)
    spec = get_task_spec(manifest["task"]["name"])
    seen: set[str] = set()
    for split, frame in splits.items():
        entry = manifest["files"][split]
        path = root / entry["file"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"benchmark file hash mismatch: {path}")
        if len(frame) != entry["rows"]:
            raise ValueError(f"benchmark row count mismatch: {path}")
        validate_task_frame(frame, spec, split=split)
        row_ids = set(frame["row_id"].astype(str))
        if seen.intersection(row_ids):
            raise ValueError("row_id values overlap between benchmark splits")
        seen.update(row_ids)
    return manifest
