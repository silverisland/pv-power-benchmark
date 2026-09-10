"""Build task-specific benchmark rows from the canonical 192-point source table."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from .spec import HISTORY_POINTS, PROTOCOL_VERSION, TASK_COLUMNS, TaskName, get_task_spec


SOURCE_COLUMNS = {
    "timestamp": "timestamp_win",
    "station": "station",
    "capacity": "cap_power_on",
    "power_history": "observe_power",
    "target_future": "observe_power_future",
    "ghi_history": "GHI_solargis",
    "temperature_history": "TEMP_solargis",
    "ghi_future": "GHI_solargis_future",
    "temperature_future": "TEMP_solargis_future",
    "wind_speed_future": "WS_solargis_future",
    "wind_direction_future": "WD_solargis_future",
}


def make_row_id(task: str, station: str, timestamp: pd.Timestamp) -> str:
    identity = f"{PROTOCOL_VERSION}|{task}|{station}|{timestamp.isoformat()}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _numeric_array(value: Any, *, column: str, minimum_length: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError(f"column {column!r} contains a non-numeric array") from error
    if len(array) < minimum_length:
        raise ValueError(
            f"column {column!r} requires at least {minimum_length} points; got {len(array)}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"column {column!r} contains NaN or infinite values")
    return array


def _future_indices(origin: pd.Timestamp, future_length: int, task: TaskName) -> np.ndarray:
    spec = get_task_spec(task)
    future_times = pd.date_range(
        origin + pd.Timedelta(minutes=spec.cadence_minutes),
        periods=future_length,
        freq=f"{spec.cadence_minutes}min",
    )
    if task == "ultra_short":
        return np.arange(spec.horizon_points, dtype=np.int64)
    next_day = origin.normalize() + pd.Timedelta(days=1)
    mask = (future_times >= next_day) & (future_times < next_day + pd.Timedelta(days=1))
    return np.flatnonzero(np.asarray(mask))


def build_task_dataset(
    source: pd.DataFrame,
    task: TaskName,
    *,
    columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Convert canonical full-horizon rows into one benchmark task table."""

    spec = get_task_spec(task)
    names = {**SOURCE_COLUMNS, **(columns or {})}
    required = list(names.values())
    missing = sorted(set(required) - set(source.columns))
    if missing:
        raise ValueError(f"source data is missing columns: {missing}")
    current = source.copy()
    current[names["timestamp"]] = pd.to_datetime(current[names["timestamp"]], errors="coerce")
    if current[names["timestamp"]].isna().any():
        raise ValueError("source data contains invalid forecast-origin timestamps")
    if task == "short_term":
        origin = current[names["timestamp"]]
        current = current.loc[(origin.dt.hour == 10) & (origin.dt.minute == 0)].copy()
        if current.empty:
            raise ValueError("short-term task requires source rows issued at 10:00")

    rows: list[dict[str, Any]] = []
    for source_row in current.to_dict(orient="records"):
        origin = pd.Timestamp(source_row[names["timestamp"]])
        station = str(source_row[names["station"]])
        capacity = float(source_row[names["capacity"]])
        if not np.isfinite(capacity) or capacity <= 0:
            raise ValueError(f"invalid capacity for station {station!r} at {origin}")
        power_history = _numeric_array(
            source_row[names["power_history"]],
            column=names["power_history"],
            minimum_length=HISTORY_POINTS,
        )[-HISTORY_POINTS:]
        ghi_history = _numeric_array(
            source_row[names["ghi_history"]],
            column=names["ghi_history"],
            minimum_length=HISTORY_POINTS,
        )[-HISTORY_POINTS:]
        temperature_history = _numeric_array(
            source_row[names["temperature_history"]],
            column=names["temperature_history"],
            minimum_length=HISTORY_POINTS,
        )[-HISTORY_POINTS:]
        target_future = _numeric_array(
            source_row[names["target_future"]],
            column=names["target_future"],
            minimum_length=spec.horizon_points,
        )
        indices = _future_indices(origin, len(target_future), task)
        if len(indices) != spec.horizon_points:
            raise ValueError(
                f"{task} row for {station!r} at {origin} resolves to {len(indices)} target points; "
                f"expected {spec.horizon_points}"
            )

        future_values = {}
        for output_name, source_key in [
            ("ghi_forecast", "ghi_future"),
            ("temperature_forecast", "temperature_future"),
            ("wind_speed_forecast", "wind_speed_future"),
            ("wind_direction_forecast", "wind_direction_future"),
        ]:
            values = _numeric_array(
                source_row[names[source_key]],
                column=names[source_key],
                minimum_length=int(indices[-1]) + 1,
            )
            future_values[output_name] = values[indices]

        if task == "ultra_short":
            target_start = origin + pd.Timedelta(minutes=15)
            target_end = origin + pd.Timedelta(hours=4)
        else:
            target_start = origin.normalize() + pd.Timedelta(days=1)
            target_end = target_start + pd.Timedelta(hours=23, minutes=45)
        rows.append(
            {
                "row_id": make_row_id(task, station, origin),
                "task": task,
                "split": "unassigned",
                "timestamp_win": origin,
                "target_start": target_start,
                "target_end": target_end,
                "station": station,
                "capacity": capacity,
                "power_history": power_history,
                "ghi_history": ghi_history,
                "temperature_history": temperature_history,
                **future_values,
                "target": target_future[indices],
            }
        )
    result = pd.DataFrame(rows, columns=TASK_COLUMNS)
    if result["row_id"].duplicated().any():
        raise ValueError("source data produces duplicate benchmark row_id values")
    return result.sort_values(["timestamp_win", "station"]).reset_index(drop=True)
