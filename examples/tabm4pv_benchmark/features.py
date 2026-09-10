"""Feature engineering extracted from the original tabm4pv.py demo."""

from __future__ import annotations

import numpy as np
import pandas as pd


WEATHER_COLUMNS = [
    "ghi_forecast",
    "temperature_forecast",
    "wind_speed_forecast",
    "wind_direction_forecast",
]


def _stack(frame: pd.DataFrame, column: str) -> np.ndarray:
    try:
        values = np.stack(frame[column].map(lambda value: np.asarray(value, dtype=np.float32)))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid benchmark array column: {column}") from error
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"benchmark column {column!r} must be a finite rectangular array")
    return values


def feature_names(history_length: int = 96) -> list[str]:
    names = [f"{column}_target" for column in WEATHER_COLUMNS]
    names.extend(f"power_lag_{lag}" for lag in range(history_length, 0, -1))
    names.extend(["predict_hour", "predict_month"])
    return names


def build_horizon_features(
    frame: pd.DataFrame,
    horizon_index: int,
    *,
    history_length: int = 96,
) -> np.ndarray:
    """Build original-demo features for one zero-based forecast horizon."""

    if horizon_index < 0:
        raise ValueError("horizon_index must be non-negative")
    power_history = _stack(frame, "power_history")
    if power_history.shape[1] < history_length:
        raise ValueError(
            f"power_history has {power_history.shape[1]} points; requires {history_length}"
        )
    future_weather = []
    for column in WEATHER_COLUMNS:
        values = _stack(frame, column)
        if values.shape[1] <= horizon_index:
            raise ValueError(f"{column} does not contain horizon index {horizon_index}")
        future_weather.append(values[:, horizon_index : horizon_index + 1])

    origin = pd.to_datetime(frame["timestamp_win"], errors="coerce")
    if origin.isna().any():
        raise ValueError("timestamp_win contains invalid timestamps")
    target_time = origin + pd.to_timedelta((horizon_index + 1) * 15, unit="min")
    time_features = np.column_stack(
        [target_time.dt.hour.to_numpy(), target_time.dt.month.to_numpy()]
    ).astype(np.float32)
    return np.concatenate(
        [*future_weather, power_history[:, -history_length:], time_features],
        axis=1,
    ).astype(np.float32)


def build_horizon_target(frame: pd.DataFrame, horizon_index: int) -> np.ndarray:
    target = _stack(frame, "target")
    if horizon_index < 0 or target.shape[1] <= horizon_index:
        raise ValueError(f"target does not contain horizon index {horizon_index}")
    return target[:, horizon_index].astype(np.float32)
