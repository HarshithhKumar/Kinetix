import pickle

import numpy as np

from .config import SCALER_PATH

with open(SCALER_PATH, "rb") as file:
    scaler = pickle.load(file)


def preprocess_sensor_data(sensor_data):
    """
    Scale raw UCI-equivalent sensor windows for LSTM inference.

    Expects raw body acceleration (g) and body gyroscope (rad/s) values
    with shape (128, 6). Applies the training-fitted StandardScaler once.
    """
    data = np.array(sensor_data, dtype=np.float32)

    if data.shape != (128, 6):
        raise ValueError(
            f"Expected sensor data shape (128, 6), but received {data.shape}"
        )

    data_2d = data.reshape(-1, 6)
    data_scaled = scaler.transform(data_2d)
    return data_scaled.reshape(1, 128, 6).astype(np.float32)
