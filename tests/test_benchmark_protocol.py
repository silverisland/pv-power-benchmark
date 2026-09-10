import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pv_benchmark import (
    build_task_dataset,
    initialize_codex_project,
    score_benchmark,
    validate_benchmark,
    write_benchmark,
)


def source_frame() -> pd.DataFrame:
    rows = []
    for day in pd.date_range("2026-08-01", periods=3, freq="D"):
        for station, capacity in [("PV001", 100.0), ("PV002", 200.0)]:
            origin = day + pd.Timedelta(hours=10)
            base = np.arange(192, dtype=np.float32)
            rows.append(
                {
                    "timestamp_win": origin,
                    "station": station,
                    "cap_power_on": capacity,
                    "observe_power": np.arange(672, dtype=np.float32),
                    "observe_power_future": base,
                    "GHI_solargis": np.arange(672, dtype=np.float32),
                    "GHI_solargis_future": base + 100,
                    "TEMP_solargis": np.arange(672, dtype=np.float32),
                    "TEMP_solargis_future": base + 200,
                    "WS_solargis_future": base + 300,
                    "WD_solargis_future": base + 400,
                }
            )
    return pd.DataFrame(rows)


class BenchmarkProtocolTests(unittest.TestCase):
    def test_short_term_builder_uses_next_calendar_day(self):
        frame = build_task_dataset(source_frame(), "short_term")
        self.assertTrue(frame["target"].map(len).eq(96).all())
        # 10:00 origin, first future point is 10:15. Next-day 00:00 is index 55.
        self.assertEqual(float(frame.iloc[0]["target"][0]), 55.0)
        self.assertEqual(float(frame.iloc[0]["target"][-1]), 150.0)
        self.assertTrue((frame["target_start"].dt.hour == 0).all())
        self.assertTrue((frame["target_end"].dt.hour == 23).all())
        self.assertTrue((frame["target_end"].dt.minute == 45).all())

    def test_perfect_predictions_score_one(self):
        frame = build_task_dataset(source_frame(), "short_term")
        with tempfile.TemporaryDirectory() as directory:
            benchmark_dir = Path(directory) / "short_term"
            manifest = write_benchmark(
                frame,
                benchmark_dir,
                task="short_term",
                train_end="2026-08-01 23:59:59",
                validation_end="2026-08-02 23:59:59",
            )
            self.assertEqual(validate_benchmark(benchmark_dir)["benchmark_id"], manifest["benchmark_id"])
            test = pd.read_parquet(benchmark_dir / "test.parquet")
            predictions = pd.DataFrame(
                {
                    "row_id": test["row_id"],
                    "model_id": "perfect_model",
                    "prediction": test["target"],
                }
            )
            summary, details = score_benchmark(benchmark_dir, predictions)
            self.assertEqual(summary["accuracy"], 1.0)
            self.assertEqual(summary["model_id"], "perfect_model")
            self.assertEqual(len(details), len(test))

    def test_missing_prediction_row_is_rejected(self):
        frame = build_task_dataset(source_frame(), "ultra_short")
        with tempfile.TemporaryDirectory() as directory:
            benchmark_dir = Path(directory) / "ultra_short"
            write_benchmark(
                frame,
                benchmark_dir,
                task="ultra_short",
                train_end="2026-08-01 23:59:59",
                validation_end="2026-08-02 23:59:59",
            )
            test = pd.read_parquet(benchmark_dir / "test.parquet")
            predictions = pd.DataFrame(
                {"row_id": test["row_id"], "prediction": test["target"]}
            ).iloc[:-1]
            with self.assertRaisesRegex(ValueError, "does not match"):
                score_benchmark(benchmark_dir, predictions)

    def test_codex_project_integration_is_repeatable(self):
        frame = build_task_dataset(source_frame(), "ultra_short")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_dir = root / "benchmark"
            project_dir = root / "model"
            project_dir.mkdir()
            (project_dir / "AGENTS.md").write_text("# Existing guidance\n", encoding="utf-8")
            manifest = write_benchmark(
                frame,
                benchmark_dir,
                task="ultra_short",
                train_end="2026-08-01 23:59:59",
                validation_end="2026-08-02 23:59:59",
            )
            initialize_codex_project(
                project_dir,
                benchmark_dir,
                model_id="model_v1",
            )
            initialize_codex_project(
                project_dir,
                benchmark_dir,
                model_id="model_v1",
            )
            agents = (project_dir / "AGENTS.md").read_text(encoding="utf-8")
            self.assertIn("# Existing guidance", agents)
            self.assertEqual(agents.count("pv-benchmark:managed-start"), 1)
            config = pd.read_json(project_dir / "pv-benchmark.json", typ="series")
            self.assertEqual(config["benchmark_id"], manifest["benchmark_id"])
            self.assertTrue((project_dir / "BENCHMARK.md").exists())


if __name__ == "__main__":
    unittest.main()
