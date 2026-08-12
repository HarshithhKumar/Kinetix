import sys
import unittest
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.config import PROJECT_ROOT  # noqa: E402
from app.data_loader import get_raw_sample, load_raw_uci_test_windows, load_uci_test_labels
from app.model import ACTIVITY_NAMES, predict_activity
from app.preprocessing import preprocess_sensor_data


class PipelineTests(unittest.TestCase):
    def test_raw_sample_shape(self):
        sample = get_raw_sample()
        data = np.array(sample["sensor_data"], dtype=np.float32)

        self.assertEqual(data.shape, (128, 6))
        self.assertEqual(sample["shape"], [128, 6])
        self.assertEqual(len(sample["feature_names"]), 6)
        self.assertIn(sample["ground_truth"], ACTIVITY_NAMES)

    def test_ground_truth_matches_uci_label(self):
        sample = get_raw_sample()
        labels = load_uci_test_labels()
        expected = ACTIVITY_NAMES[int(labels[0])]
        self.assertEqual(sample["ground_truth"], expected)

    def test_scaler_applied_once(self):
        raw = load_raw_uci_test_windows()[0]
        scaled_once = preprocess_sensor_data(raw)
        scaled_twice_input = scaled_once.reshape(128, 6)
        scaled_twice = preprocess_sensor_data(scaled_twice_input)

        self.assertFalse(np.allclose(scaled_once, scaled_twice))
        self.assertEqual(scaled_once.shape, (1, 128, 6))

    def test_preprocessing_rejects_invalid_shape(self):
        with self.assertRaises(ValueError):
            preprocess_sensor_data(np.zeros((64, 6)).tolist())

        with self.assertRaises(ValueError):
            preprocess_sensor_data(np.zeros((128, 3)).tolist())

    def test_prediction_output(self):
        raw = load_raw_uci_test_windows()[0]
        processed = preprocess_sensor_data(raw)
        activity, confidence, probabilities = predict_activity(processed)

        self.assertIn(activity, ACTIVITY_NAMES)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertEqual(len(probabilities), 6)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=4)

    def test_raw_sample_not_pre_scaled(self):
        raw = load_raw_uci_test_windows()[0]
        x_test_scaled = np.load(PROJECT_ROOT / "models" / "X_test.npy")[0]

        # Raw UCI window must differ from the old pre-scaled test artifact.
        self.assertFalse(np.allclose(raw, x_test_scaled, atol=1e-5))

    def test_model_input_shape(self):
        raw = load_raw_uci_test_windows()[0]
        processed = preprocess_sensor_data(raw)

        self.assertEqual(raw.shape, (128, 6))
        self.assertEqual(processed.shape, (1, 128, 6))


if __name__ == "__main__":
    unittest.main()
