"""
Integration test for the Kinetix inference API (correct raw-data pipeline).

Usage:
    python test_api.py

Requires the FastAPI server at http://localhost:8000
or set KINETIX_API_URL.
"""

import json
import os
import sys
import urllib.error
import urllib.request

API_URL = os.environ.get("KINETIX_API_URL", "http://localhost:8000")

EXPECTED_FEATURES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]

VALID_ACTIVITIES = {
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
}


def request_json(method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    url = f"{API_URL}{path}"
    data = None
    headers = {"Content-Type": "application/json"}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            detail = json.loads(body)
        except json.JSONDecodeError:
            detail = {"detail": body}
        return exc.code, detail


def main() -> int:
    print(f"Testing Kinetix API at {API_URL}\n")

    status, health = request_json("GET", "/health")
    print(f"GET /health -> {status} {health}")
    if status != 200 or health.get("status") != "ok":
        print("Health check failed.")
        return 1

    status, sample = request_json("GET", "/sample")
    print(f"GET /sample -> {status}")
    if status != 200:
        print("Sample request failed:", sample)
        return 1

    sensor_data = sample.get("sensor_data")
    ground_truth = sample.get("ground_truth")
    shape = sample.get("shape")
    feature_names = sample.get("feature_names")

    if not sensor_data or ground_truth is None or shape != [128, 6]:
        print("Invalid /sample response:", sample)
        return 1

    if feature_names != EXPECTED_FEATURES:
        print("Unexpected feature_names:", feature_names)
        return 1

    if len(sensor_data) != 128 or any(len(row) != 6 for row in sensor_data):
        print("Sample shape validation failed.")
        return 1

    if ground_truth not in VALID_ACTIVITIES:
        print("Invalid ground truth label:", ground_truth)
        return 1

    print(f"Ground truth: {ground_truth}")
    print(f"Sample shape: {shape}")
    print(f"Feature names: {feature_names}")

    status, prediction = request_json(
        "POST",
        "/predict",
        {"sensor_data": sensor_data},
    )
    print(f"\nPOST /predict -> {status}")
    if status != 200:
        print("Prediction failed:", prediction)
        return 1

    predicted_activity = prediction.get("activity")
    confidence = prediction.get("confidence")
    probabilities = prediction.get("probabilities", {})

    print(f"Predicted activity: {predicted_activity}")
    print(f"Confidence: {confidence:.4f}" if confidence is not None else "Confidence: n/a")
    print("Probabilities:")
    for name, value in probabilities.items():
        print(f"  {name}: {value:.4f}")

    if predicted_activity not in VALID_ACTIVITIES:
        print("Invalid predicted activity.")
        return 1

    if not (0.0 <= confidence <= 1.0):
        print("Confidence out of range.")
        return 1

    if len(probabilities) != 6:
        print("Expected six probabilities.")
        return 1

    prob_sum = sum(probabilities.values())
    if not (0.99 <= prob_sum <= 1.01):
        print(f"Probabilities do not sum to ~1 (sum={prob_sum:.4f}).")
        return 1

    print(f"\nGround truth vs prediction: {ground_truth} -> {predicted_activity}")
    print("Pipeline test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
