import unittest

import numpy as np
import pandas as pd

from pv_metrics import (
    evaluate_forecast,
    evaluate_prediction_frame,
    short_term_metrics,
    ultra_short_metrics,
)


class MetricTests(unittest.TestCase):
    def test_perfect_forecasts_have_accuracy_one(self):
        ultra = np.linspace(0, 100, 16)
        short = np.linspace(0, 100, 96)
        self.assertEqual(ultra_short_metrics(ultra, ultra, 100).accuracy, 1.0)
        self.assertEqual(short_term_metrics(short, short, 100).accuracy, 1.0)

    def test_ultra_short_uses_normalized_absolute_error(self):
        truth = np.zeros(16)
        prediction = np.full(16, 2.0)
        result = ultra_short_metrics(truth, prediction, capacity=100.0)
        self.assertAlmostEqual(result.normalized_error, 0.1)
        self.assertAlmostEqual(result.accuracy, 0.9)

    def test_short_term_uses_normalized_rmse(self):
        truth = np.zeros(96)
        prediction = np.full(96, 2.0)
        result = short_term_metrics(truth, prediction, capacity=100.0)
        self.assertAlmostEqual(result.normalized_error, 0.1)
        self.assertAlmostEqual(result.accuracy_percent, 90.0)

    def test_batch_uses_one_capacity_per_origin_and_macro_average(self):
        truth = np.zeros((2, 16))
        prediction = np.vstack([np.full(16, 2.0), np.full(16, 6.0)])
        result = evaluate_forecast("ultra_short", truth, prediction, [100.0, 200.0])
        np.testing.assert_allclose(result.per_sample_normalized_error, [0.1, 0.15])
        self.assertAlmostEqual(result.accuracy, 0.875)

    def test_wrong_horizon_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly 16 points"):
            ultra_short_metrics(np.zeros(15), np.zeros(15), 100.0)

    def test_non_finite_input_is_rejected(self):
        truth = np.zeros(96)
        truth[2] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            short_term_metrics(truth, np.zeros(96), 100.0)

    def test_non_positive_capacity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            ultra_short_metrics(np.zeros(16), np.zeros(16), 0.0)

    def test_short_term_frame_validates_schedule(self):
        frame = pd.DataFrame(
            {
                "timestamp_win": [pd.Timestamp("2026-08-01 09:00:00")],
                "target_start": [pd.Timestamp("2026-08-02 00:00:00")],
                "target_end": [pd.Timestamp("2026-08-02 23:45:00")],
                "groundtruth": [np.zeros(96)],
                "prediction": [np.zeros(96)],
                "capacity": [100.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "issued at 10:00"):
            evaluate_prediction_frame(frame, "short_term")


if __name__ == "__main__":
    unittest.main()
