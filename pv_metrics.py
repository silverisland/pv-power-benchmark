"""Metrics for 15-minute photovoltaic power forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


TaskType = Literal["ultra_short", "short_term"]
HORIZON_POINTS: dict[str, int] = {"ultra_short": 16, "short_term": 96}
DEFAULT_CAPACITY_FLOOR_RATIO = 0.2


@dataclass(frozen=True)
class MetricResult:
    """Metric values for one or more equally weighted forecast origins."""

    task: TaskType
    horizon_points: int
    sample_count: int
    normalized_error: float
    accuracy: float
    accuracy_percent: float
    per_sample_normalized_error: np.ndarray
    per_sample_accuracy: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "horizon_points": self.horizon_points,
            "sample_count": self.sample_count,
            "normalized_error": self.normalized_error,
            "accuracy": self.accuracy,
            "accuracy_percent": self.accuracy_percent,
            "per_sample_normalized_error": self.per_sample_normalized_error.tolist(),
            "per_sample_accuracy": self.per_sample_accuracy.tolist(),
        }


def _as_2d(values: Any, *, name: str, horizon_points: int) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular numeric array") from error
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape ({horizon_points},) or (B, {horizon_points})")
    if array.shape[0] == 0 or array.shape[1] != horizon_points:
        raise ValueError(
            f"{name} must contain exactly {horizon_points} points per sample; got shape {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _capacity_column(capacity: Any, *, sample_count: int) -> np.ndarray:
    try:
        values = np.asarray(capacity, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("capacity must be a positive scalar or one value per sample") from error
    if values.ndim == 0:
        values = np.full(sample_count, float(values), dtype=np.float64)
    else:
        values = values.reshape(-1)
        if values.size != sample_count:
            raise ValueError(
                f"capacity must be scalar or contain {sample_count} values; got {values.size}"
            )
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("capacity values must be finite and greater than zero")
    return values.reshape(-1, 1)


def _normalized_residuals(
    groundtruth: Any,
    prediction: Any,
    capacity: Any,
    *,
    horizon_points: int,
    capacity_floor_ratio: float,
) -> np.ndarray:
    if not np.isfinite(capacity_floor_ratio) or capacity_floor_ratio < 0:
        raise ValueError("capacity_floor_ratio must be finite and non-negative")
    truth = _as_2d(groundtruth, name="groundtruth", horizon_points=horizon_points)
    forecast = _as_2d(prediction, name="prediction", horizon_points=horizon_points)
    if forecast.shape != truth.shape:
        raise ValueError(
            f"prediction and groundtruth must have the same shape; got {forecast.shape} and {truth.shape}"
        )
    capacity_values = _capacity_column(capacity, sample_count=truth.shape[0])
    denominator = np.maximum(truth, capacity_floor_ratio * capacity_values)
    if (denominator <= 0).any():
        raise ValueError("normalization denominator must be greater than zero")
    return (forecast - truth) / denominator


def _result(task: TaskType, per_sample_error: np.ndarray) -> MetricResult:
    per_sample_error = np.asarray(per_sample_error, dtype=np.float64)
    mean_error = float(np.mean(per_sample_error))
    per_sample_accuracy = 1.0 - per_sample_error
    accuracy = 1.0 - mean_error
    return MetricResult(
        task=task,
        horizon_points=HORIZON_POINTS[task],
        sample_count=len(per_sample_error),
        normalized_error=mean_error,
        accuracy=accuracy,
        accuracy_percent=100.0 * accuracy,
        per_sample_normalized_error=per_sample_error,
        per_sample_accuracy=per_sample_accuracy,
    )


def ultra_short_metrics(
    groundtruth: Any,
    prediction: Any,
    capacity: Any,
    *,
    capacity_floor_ratio: float = DEFAULT_CAPACITY_FLOOR_RATIO,
) -> MetricResult:
    """Evaluate continuous four-hour forecasts with 16-point normalized MAE."""

    residuals = _normalized_residuals(
        groundtruth,
        prediction,
        capacity,
        horizon_points=HORIZON_POINTS["ultra_short"],
        capacity_floor_ratio=capacity_floor_ratio,
    )
    return _result("ultra_short", np.mean(np.abs(residuals), axis=1))


def short_term_metrics(
    groundtruth: Any,
    prediction: Any,
    capacity: Any,
    *,
    capacity_floor_ratio: float = DEFAULT_CAPACITY_FLOOR_RATIO,
) -> MetricResult:
    """Evaluate next-day forecasts with 96-point normalized RMSE."""

    residuals = _normalized_residuals(
        groundtruth,
        prediction,
        capacity,
        horizon_points=HORIZON_POINTS["short_term"],
        capacity_floor_ratio=capacity_floor_ratio,
    )
    return _result("short_term", np.sqrt(np.mean(np.square(residuals), axis=1)))


def evaluate_forecast(
    task: TaskType,
    groundtruth: Any,
    prediction: Any,
    capacity: Any,
    *,
    capacity_floor_ratio: float = DEFAULT_CAPACITY_FLOOR_RATIO,
) -> MetricResult:
    """Dispatch to the metric defined for the requested forecast task."""

    if task == "ultra_short":
        return ultra_short_metrics(
            groundtruth,
            prediction,
            capacity,
            capacity_floor_ratio=capacity_floor_ratio,
        )
    if task == "short_term":
        return short_term_metrics(
            groundtruth,
            prediction,
            capacity,
            capacity_floor_ratio=capacity_floor_ratio,
        )
    raise ValueError(f"unsupported task {task!r}; expected one of {sorted(HORIZON_POINTS)}")


def evaluate_prediction_frame(
    frame: Any,
    task: TaskType,
    *,
    groundtruth_column: str = "groundtruth",
    prediction_column: str = "prediction",
    capacity_column: str = "capacity",
    timestamp_column: str = "timestamp_win",
    target_start_column: str = "target_start",
    target_end_column: str = "target_end",
    validate_schedule: bool = True,
    capacity_floor_ratio: float = DEFAULT_CAPACITY_FLOOR_RATIO,
) -> MetricResult:
    """Evaluate a DataFrame-like object containing one array per row."""

    required = [groundtruth_column, prediction_column, capacity_column]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"prediction frame is missing columns: {missing}")
    if validate_schedule:
        _validate_frame_schedule(
            frame,
            task,
            timestamp_column=timestamp_column,
            target_start_column=target_start_column,
            target_end_column=target_end_column,
        )
    try:
        groundtruth = np.stack(frame[groundtruth_column].to_numpy())
        prediction = np.stack(frame[prediction_column].to_numpy())
        capacity = frame[capacity_column].to_numpy(dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("prediction frame contains invalid array columns") from error
    return evaluate_forecast(
        task,
        groundtruth,
        prediction,
        capacity,
        capacity_floor_ratio=capacity_floor_ratio,
    )


def _validate_frame_schedule(
    frame: Any,
    task: TaskType,
    *,
    timestamp_column: str,
    target_start_column: str,
    target_end_column: str,
) -> None:
    """Validate issue time and target interval when schedule columns are present."""

    if timestamp_column not in frame:
        raise ValueError(f"schedule validation requires column {timestamp_column!r}")
    try:
        import pandas as pd

        origin = pd.to_datetime(frame[timestamp_column], errors="coerce")
    except (ImportError, TypeError, ValueError) as error:
        raise ValueError("unable to parse forecast-origin timestamps") from error
    if origin.isna().any():
        raise ValueError(f"{timestamp_column} contains invalid timestamps")
    if task == "short_term" and not ((origin.dt.hour == 10) & (origin.dt.minute == 0)).all():
        raise ValueError("short-term forecasts must be issued at 10:00")

    has_start = target_start_column in frame
    has_end = target_end_column in frame
    if has_start != has_end:
        raise ValueError("target_start and target_end must either both be present or both be absent")
    if not has_start:
        return
    target_start = pd.to_datetime(frame[target_start_column], errors="coerce")
    target_end = pd.to_datetime(frame[target_end_column], errors="coerce")
    if target_start.isna().any() or target_end.isna().any():
        raise ValueError("target interval contains invalid timestamps")
    if task == "ultra_short":
        valid = (target_start == origin + pd.Timedelta(minutes=15)) & (
            target_end == origin + pd.Timedelta(hours=4)
        )
        if not valid.all():
            raise ValueError("ultra-short target interval must run from T0+15min through T0+4h")
    else:
        next_day = origin.dt.normalize() + pd.Timedelta(days=1)
        valid = (target_start == next_day) & (
            target_end == next_day + pd.Timedelta(hours=23, minutes=45)
        )
        if not valid.all():
            raise ValueError("short-term target interval must cover the next day from 00:00 through 23:45")
