import unittest

import numpy as np
import pandas as pd

from examples.tabm4pv_benchmark.features import (
    build_horizon_features,
    build_horizon_target,
    feature_names,
)


class TabMBenchmarkFeatureTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "timestamp_win": [pd.Timestamp("2026-08-01 09:00:00")],
                "power_history": [np.arange(672, dtype=np.float32)],
                "ghi_forecast": [np.arange(16, dtype=np.float32) + 100],
                "temperature_forecast": [np.arange(16, dtype=np.float32) + 200],
                "wind_speed_forecast": [np.arange(16, dtype=np.float32) + 300],
                "wind_direction_forecast": [np.arange(16, dtype=np.float32) + 400],
                "target": [np.arange(16, dtype=np.float32) + 500],
            }
        )

    def test_features_preserve_original_order(self):
        features = build_horizon_features(self.frame, 15)
        self.assertEqual(features.shape, (1, 102))
        np.testing.assert_array_equal(features[0, :4], [115, 215, 315, 415])
        np.testing.assert_array_equal(features[0, 4:100], np.arange(576, 672))
        np.testing.assert_array_equal(features[0, -2:], [13, 8])
        self.assertEqual(len(feature_names()), 102)

    def test_target_selects_requested_horizon(self):
        target = build_horizon_target(self.frame, 15)
        np.testing.assert_array_equal(target, [515])


if __name__ == "__main__":
    unittest.main()
