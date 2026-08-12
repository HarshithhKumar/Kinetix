import numpy as np

from .config import DATA_DIR
from .model import ACTIVITY_NAMES

UCI_TEST_INERTIAL_DIR = DATA_DIR / "test" / "Inertial Signals"
UCI_Y_TEST_PATH = DATA_DIR / "test" / "y_test.txt"

FEATURE_NAMES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]

UCI_TEST_SIGNAL_FILES = [
    "body_acc_x_test.txt",
    "body_acc_y_test.txt",
    "body_acc_z_test.txt",
    "body_gyro_x_test.txt",
    "body_gyro_y_test.txt",
    "body_gyro_z_test.txt",
]

DEFAULT_SAMPLE_INDEX = 0


def load_raw_uci_test_windows() -> np.ndarray:
    """Load all raw UCI HAR test windows. Shape: (n_windows, 128, 6)."""
    signals = [
        np.loadtxt(UCI_TEST_INERTIAL_DIR / filename)
        for filename in UCI_TEST_SIGNAL_FILES
    ]
    return np.stack(signals, axis=-1).astype(np.float32)


def load_uci_test_labels() -> np.ndarray:
    """Load UCI test labels as 0-based indices matching ACTIVITY_NAMES."""
    labels = np.loadtxt(UCI_Y_TEST_PATH, dtype=int)
    return (labels - 1).astype(int)


def get_activity_name(label_index: int) -> str:
    return ACTIVITY_NAMES[label_index]


def get_raw_test_window(index: int = DEFAULT_SAMPLE_INDEX) -> np.ndarray:
    """Return one raw UCI test window with shape (128, 6)."""
    windows = load_raw_uci_test_windows()
    if index < 0 or index >= len(windows):
        raise IndexError(
            f"Sample index {index} out of range for UCI test set (0..{len(windows) - 1})"
        )
    return windows[index]


def get_raw_sample(index: int = DEFAULT_SAMPLE_INDEX) -> dict:
    """Build the /sample response payload from raw UCI test data."""
    window = get_raw_test_window(index)
    labels = load_uci_test_labels()
    ground_truth = get_activity_name(int(labels[index]))

    return {
        "sensor_data": window.tolist(),
        "ground_truth": ground_truth,
        "shape": [128, 6],
        "feature_names": FEATURE_NAMES,
    }
