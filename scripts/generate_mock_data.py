"""Generate deterministic mock data matching 数据说明.md.

The generated values are synthetic and are intended only for interface tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pv_benchmark import build_task_dataset, score_benchmark, write_benchmark  # noqa: E402


SEED = 20260910
HISTORY_15MIN = 7 * 24 * 4
FUTURE_15MIN = 2 * 24 * 4
FUTURE_HOURLY = 24
ULTRA_SHORT_POINTS = 4 * 4
SHORT_TERM_POINTS = 24 * 4


def solar_shape(times: pd.DatetimeIndex) -> np.ndarray:
    hour = times.hour.to_numpy() + times.minute.to_numpy() / 60.0
    daylight = np.sin(np.pi * (hour - 6.0) / 12.0)
    return np.clip(daylight, 0.0, None).astype(np.float32)


def weather(times: pd.DatetimeIndex, rng: np.random.Generator) -> dict[str, np.ndarray]:
    sun = solar_shape(times)
    day = times.dayofyear.to_numpy()
    ghi = np.clip(900.0 * sun + rng.normal(0, 25, len(times)), 0, None)
    temp = 23.0 + 7.0 * np.sin(2 * np.pi * (times.hour.to_numpy() - 8) / 24) + 2 * np.sin(2 * np.pi * day / 365)
    temp = temp + rng.normal(0, 0.8, len(times))
    ws = np.clip(2.5 + rng.normal(0, 0.7, len(times)), 0, None)
    wd = np.mod(160 + rng.normal(0, 35, len(times)), 360)
    prec = np.where(rng.random(len(times)) < 0.06, rng.gamma(1.2, 1.5, len(times)), 0)
    pwat = np.clip(32 + rng.normal(0, 5, len(times)), 1, None)
    return {
        "GHI": ghi.astype(np.float32),
        "TEMP": temp.astype(np.float32),
        "WS": ws.astype(np.float32),
        "WD": wd.astype(np.float32),
        "PREC": prec.astype(np.float32),
        "PWAT": pwat.astype(np.float32),
    }


def power_from_ghi(ghi: np.ndarray, capacity: float, rng: np.random.Generator) -> np.ndarray:
    ratio = np.clip(ghi / 1000.0 * 0.92 + rng.normal(0, 0.025, len(ghi)), 0, 1.05)
    return (capacity * ratio).astype(np.float32)


def array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def make_station_info() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plantid": pd.Series(["PV001", "PV002"], dtype="string"),
            "plantname": pd.Series(["Mock Solar East", "Mock Solar West"], dtype="string"),
            "plant_pointname": pd.Series(["plant_guangfu0001", "plant_guangfu0002"], dtype="string"),
            "province": pd.Series(["Guangxi", "Guangxi"], dtype="string"),
            "GCCAPACITY": [100.0, 150.0],
            "LONGITUDE": [108.32, 109.41],
            "LATITUDE": [22.82, 23.12],
            "timezone": pd.Series(["Asia/Shanghai", "Asia/Shanghai"], dtype="string"),
        }
    )


def make_15min_samples(stations: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    days = pd.date_range("2026-08-01", periods=3, freq="D")
    origins = [day + pd.Timedelta(hours=hour) for day in days for hour in (9, 10)]
    rows: list[dict[str, object]] = []
    for station in stations.itertuples(index=False):
        capacity = float(station.GCCAPACITY)
        for origin in origins:
            history_times = pd.date_range(end=origin - pd.Timedelta(minutes=15), periods=HISTORY_15MIN, freq="15min")
            future_times = pd.date_range(start=origin + pd.Timedelta(minutes=15), periods=FUTURE_15MIN, freq="15min")
            history_weather = weather(history_times, rng)
            future_weather = weather(future_times, rng)
            rows.append(
                {
                    "timestamp_win": origin,
                    "station": station.plantid,
                    "cap_power_on": capacity,
                    "observe_power": array(power_from_ghi(history_weather["GHI"], capacity, rng)),
                    "observe_power_future": array(power_from_ghi(future_weather["GHI"], capacity, rng)),
                    "GHI_solargis": array(history_weather["GHI"]),
                    "GHI_solargis_future": array(future_weather["GHI"]),
                    "TEMP_solargis": array(history_weather["TEMP"]),
                    "TEMP_solargis_future": array(future_weather["TEMP"]),
                    "WS_solargis_future": array(future_weather["WS"]),
                    "WD_solargis_future": array(future_weather["WD"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["timestamp_win", "station"]).reset_index(drop=True)


def make_15min_predictions(samples: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for row in samples.itertuples(index=False):
        truth = array(row.observe_power_future)
        prediction = np.clip(truth + rng.normal(0, row.cap_power_on * 0.035, len(truth)), 0, row.cap_power_on * 1.05)
        rows.append(
            {
                "timestamp_win": row.timestamp_win,
                "station": row.station,
                "model_id": "mock_tabm_v1",
                "prediction": array(prediction),
                "groundtruth": truth,
                "capacity": float(row.cap_power_on),
            }
        )
    return pd.DataFrame(rows)


def make_metric_predictions(
    stations: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ultra_rows: list[dict[str, object]] = []
    short_rows: list[dict[str, object]] = []
    days = pd.date_range("2026-08-01", periods=3, freq="D")
    for station in stations.itertuples(index=False):
        capacity = float(station.GCCAPACITY)
        for day in days:
            ultra_origin = day + pd.Timedelta(hours=9)
            ultra_times = pd.date_range(
                ultra_origin + pd.Timedelta(minutes=15),
                periods=ULTRA_SHORT_POINTS,
                freq="15min",
            )
            ultra_weather = weather(ultra_times, rng)
            ultra_truth = power_from_ghi(ultra_weather["GHI"], capacity, rng)
            ultra_prediction = np.clip(
                ultra_truth + rng.normal(0, capacity * 0.035, ULTRA_SHORT_POINTS),
                0,
                capacity * 1.05,
            )
            ultra_rows.append(
                {
                    "timestamp_win": ultra_origin,
                    "target_start": ultra_times[0],
                    "target_end": ultra_times[-1],
                    "station": station.plantid,
                    "model_id": "mock_tabm_ultra_v1",
                    "prediction": array(ultra_prediction),
                    "groundtruth": array(ultra_truth),
                    "capacity": capacity,
                }
            )

            short_origin = day + pd.Timedelta(hours=10)
            next_day = day + pd.Timedelta(days=1)
            short_times = pd.date_range(next_day, periods=SHORT_TERM_POINTS, freq="15min")
            short_weather = weather(short_times, rng)
            short_truth = power_from_ghi(short_weather["GHI"], capacity, rng)
            short_prediction = np.clip(
                short_truth + rng.normal(0, capacity * 0.05, SHORT_TERM_POINTS),
                0,
                capacity * 1.05,
            )
            short_rows.append(
                {
                    "timestamp_win": short_origin,
                    "target_start": short_times[0],
                    "target_end": short_times[-1],
                    "station": station.plantid,
                    "model_id": "mock_tabm_short_v1",
                    "prediction": array(short_prediction),
                    "groundtruth": array(short_truth),
                    "capacity": capacity,
                }
            )
    return pd.DataFrame(ultra_rows), pd.DataFrame(short_rows)


def make_province_delivery(samples: pd.DataFrame) -> pd.DataFrame:
    origin = samples["timestamp_win"].min()
    selected = samples[samples["timestamp_win"].eq(origin)]
    curve = np.sum(np.stack(selected["observe_power_future"].to_numpy()), axis=0)
    times = pd.date_range(start=origin + pd.Timedelta(minutes=15), periods=16, freq="15min")
    return pd.DataFrame(
        {
            "dtime": times,
            "predict_power_province_guangxi_solar": curve[:16].astype(np.float64),
        }
    )


def make_hourly(stations: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    timestamps = pd.date_range("2026-11-01 09:00:00", periods=3, freq="D")
    for station in stations.itertuples(index=False):
        capacity = float(station.GCCAPACITY)
        for timestamp in timestamps:
            now_weather = weather(pd.DatetimeIndex([timestamp]), rng)
            future_times = pd.date_range(timestamp + pd.Timedelta(hours=1), periods=FUTURE_HOURLY, freq="h")
            forecast = weather(future_times, rng)
            current_power = float(power_from_ghi(now_weather["GHI"], capacity, rng)[0])
            groundtruth = power_from_ghi(forecast["GHI"], capacity, rng)
            prediction = np.clip(groundtruth + rng.normal(0, capacity * 0.04, FUTURE_HOURLY), 0, capacity * 1.05)
            row: dict[str, object] = {
                "timestamp": timestamp,
                "id": station.plantid,
                "pv_data": current_power,
            }
            for name in ["GHI", "TEMP", "WS", "WD", "PREC", "PWAT"]:
                row[name] = float(now_weather[name][0])
                row[f"{name}_future1d"] = array(forecast[name])
            rows.append(row)
            prediction_rows.append(
                {
                    "timestamp": timestamp,
                    "id": station.plantid,
                    "prediction": array(prediction),
                    "groundtruth": array(groundtruth),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(prediction_rows)


def file_manifest(path: Path, frame: pd.DataFrame, *, cadence: str, arrays: dict[str, int]) -> dict[str, object]:
    return {
        "file": path.name,
        "rows": len(frame),
        "columns": list(frame.columns),
        "cadence": cadence,
        "array_lengths": arrays,
    }


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, engine="pyarrow", index=False)
    print(f"wrote {path.resolve()} rows={len(frame)}")


def generate(output_dir: Path, benchmark_root: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    stations = make_station_info()
    samples = make_15min_samples(stations, rng)
    predictions = make_15min_predictions(samples, rng)
    ultra_predictions, short_predictions = make_metric_predictions(stations, rng)
    delivery = make_province_delivery(samples)
    hourly, hourly_predictions = make_hourly(stations, rng)

    station_path = output_dir / "station_info.csv"
    samples_path = output_dir / "pv_samples_15min.parquet"
    predictions_path = output_dir / "predictions_15min.parquet"
    ultra_predictions_path = output_dir / "ultra_short_predictions.parquet"
    short_predictions_path = output_dir / "short_term_predictions.parquet"
    delivery_path = output_dir / "province_delivery.parquet"
    hourly_path = output_dir / "eupv_samples_hourly.parquet"
    hourly_predictions_path = output_dir / "eupv_predictions_hourly.parquet"

    stations.to_csv(station_path, index=False)
    print(f"wrote {station_path.resolve()} rows={len(stations)}")
    write_parquet(samples, samples_path)
    write_parquet(predictions, predictions_path)
    write_parquet(ultra_predictions, ultra_predictions_path)
    write_parquet(short_predictions, short_predictions_path)
    write_parquet(delivery, delivery_path)
    write_parquet(hourly, hourly_path)
    write_parquet(hourly_predictions, hourly_predictions_path)

    manifest = {
        "schema_version": "1.1",
        "generator_seed": SEED,
        "timezone": "Asia/Shanghai",
        "timestamp_storage": "timezone-naive local time",
        "units": {
            "power": "MW",
            "GHI": "W/m2",
            "TEMP": "degC",
            "WS": "m/s",
            "WD": "degree",
            "PREC": "mm",
            "PWAT": "mm",
        },
        "files": [
            {"file": station_path.name, "rows": len(stations), "columns": list(stations.columns), "cadence": "static", "array_lengths": {}},
            file_manifest(samples_path, samples, cadence="15min", arrays={
                "observe_power": HISTORY_15MIN,
                "observe_power_future": FUTURE_15MIN,
                "GHI_solargis": HISTORY_15MIN,
                "GHI_solargis_future": FUTURE_15MIN,
                "TEMP_solargis": HISTORY_15MIN,
                "TEMP_solargis_future": FUTURE_15MIN,
                "WS_solargis_future": FUTURE_15MIN,
                "WD_solargis_future": FUTURE_15MIN,
            }),
            file_manifest(predictions_path, predictions, cadence="15min", arrays={"prediction": FUTURE_15MIN, "groundtruth": FUTURE_15MIN}),
            file_manifest(ultra_predictions_path, ultra_predictions, cadence="15min", arrays={"prediction": ULTRA_SHORT_POINTS, "groundtruth": ULTRA_SHORT_POINTS}),
            file_manifest(short_predictions_path, short_predictions, cadence="15min", arrays={"prediction": SHORT_TERM_POINTS, "groundtruth": SHORT_TERM_POINTS}),
            file_manifest(delivery_path, delivery, cadence="15min", arrays={}),
            file_manifest(hourly_path, hourly, cadence="1h", arrays={f"{name}_future1d": FUTURE_HOURLY for name in ["GHI", "TEMP", "WS", "WD", "PREC", "PWAT"]}),
            file_manifest(hourly_predictions_path, hourly_predictions, cadence="1h", arrays={"prediction": FUTURE_HOURLY, "groundtruth": FUTURE_HOURLY}),
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path.resolve()}")

    benchmark_boundaries = {
        "train_end": "2026-08-01 23:59:59",
        "validation_end": "2026-08-02 23:59:59",
    }
    for task in ("ultra_short", "short_term"):
        task_frame = build_task_dataset(samples, task)
        task_dir = benchmark_root / task
        benchmark_manifest = write_benchmark(
            task_frame,
            task_dir,
            task=task,
            **benchmark_boundaries,
        )
        print(
            f"wrote benchmark {task_dir.resolve()} "
            f"benchmark_id={benchmark_manifest['benchmark_id']}"
        )
        reference_dir = task_dir / "reference"
        reference_dir.mkdir(parents=True, exist_ok=True)
        for split in ("validation", "test"):
            labels = pd.read_parquet(task_dir / f"{split}.parquet")
            reference = labels[["row_id"]].copy()
            reference["model_id"] = "mock_persistence_baseline_v1"
            reference["prediction"] = labels["target"].map(
                lambda values: array(np.clip(np.asarray(values) * 0.92, 0, None))
            )
            prediction_path = reference_dir / f"{split}_predictions.parquet"
            reference.to_parquet(prediction_path, index=False)
            summary, details = score_benchmark(task_dir, prediction_path, split=split)
            metrics_path = reference_dir / f"{split}_metrics.json"
            metrics_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            details.to_parquet(reference_dir / f"{split}_details.parquet", index=False)
            print(f"wrote reference score {metrics_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "mock_data")
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=PROJECT_ROOT / "benchmark_data" / "v1",
    )
    args = parser.parse_args()
    generate(args.output_dir, args.benchmark_root)


if __name__ == "__main__":
    main()
