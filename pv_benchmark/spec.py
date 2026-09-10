"""Versioned task definitions and table contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


PROTOCOL_VERSION = "1.0.0"
CADENCE_MINUTES = 15
HISTORY_POINTS = 7 * 24 * 4
CAPACITY_FLOOR_RATIO = 0.2

TaskName = Literal["ultra_short", "short_term"]


@dataclass(frozen=True)
class TaskSpec:
    name: TaskName
    description: str
    horizon_points: int
    cadence_minutes: int
    issue_time: str
    target_rule: str
    metric: str
    aggregation: str = "equal_weight_forecast_origins"
    capacity_floor_ratio: float = CAPACITY_FLOOR_RATIO

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


TASKS: dict[str, TaskSpec] = {
    "ultra_short": TaskSpec(
        name="ultra_short",
        description="Continuous four-hour photovoltaic power forecast",
        horizon_points=16,
        cadence_minutes=CADENCE_MINUTES,
        issue_time="any_15_minute_boundary",
        target_rule="T0+15min through T0+4h",
        metric="normalized_mae_accuracy",
    ),
    "short_term": TaskSpec(
        name="short_term",
        description="10:00 issue for the whole next calendar day",
        horizon_points=96,
        cadence_minutes=CADENCE_MINUTES,
        issue_time="10:00 Asia/Shanghai",
        target_rule="next day 00:00 through 23:45",
        metric="normalized_rmse_accuracy",
    ),
}

TASK_COLUMNS = [
    "row_id",
    "task",
    "split",
    "timestamp_win",
    "target_start",
    "target_end",
    "station",
    "capacity",
    "power_history",
    "ghi_history",
    "temperature_history",
    "ghi_forecast",
    "temperature_forecast",
    "wind_speed_forecast",
    "wind_direction_forecast",
    "target",
]

ARRAY_LENGTHS_BASE = {
    "power_history": HISTORY_POINTS,
    "ghi_history": HISTORY_POINTS,
    "temperature_history": HISTORY_POINTS,
}


def get_task_spec(task: str) -> TaskSpec:
    try:
        return TASKS[task]
    except KeyError as error:
        raise ValueError(f"unsupported task {task!r}; expected one of {sorted(TASKS)}") from error
