"""
Evaluate the saved Kinetix LSTM on raw UCI HAR test inertial signals.

Pipeline: raw UCI windows -> StandardScaler.transform (once) -> LSTM -> metrics
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import DATA_DIR, PROJECT_ROOT  # noqa: E402
from app.data_loader import (  # noqa: E402
    UCI_TEST_SIGNAL_FILES,
    get_activity_name,
    load_raw_uci_test_windows,
    load_uci_test_labels,
)
from app.model import ACTIVITY_NAMES, model  # noqa: E402
from app.preprocessing import preprocess_sensor_data, scaler  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
FRONTEND_ASSETS_DIR = PROJECT_ROOT / "frontend" / "public" / "assets"

UCI_TRAIN_INERTIAL_DIR = DATA_DIR / "train" / "Inertial Signals"

UCI_TRAIN_SIGNAL_FILES = [
    name.replace("_test", "_train") for name in UCI_TEST_SIGNAL_FILES
]


def load_raw_uci_train_windows() -> np.ndarray:
    signals = [
        np.loadtxt(UCI_TRAIN_INERTIAL_DIR / filename)
        for filename in UCI_TRAIN_SIGNAL_FILES
    ]
    return np.stack(signals, axis=-1)


def verify_scaler_fitted_on_training_data_only() -> None:
    train_raw = load_raw_uci_train_windows()
    reference = StandardScaler()
    reference.fit(train_raw.reshape(-1, 6))

    if not np.allclose(reference.mean_, scaler.mean_):
        raise RuntimeError("Saved scaler mean_ does not match training-data fit.")
    if not np.allclose(reference.scale_, scaler.scale_):
        raise RuntimeError("Saved scaler scale_ does not match training-data fit.")

    print("Scaler leakage check: PASSED (fitted on training inertial data only)")


def save_confusion_matrix_plot(matrix: np.ndarray, output_path: str) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=ACTIVITY_NAMES,
        yticklabels=ACTIVITY_NAMES,
        cbar=False,
    )
    plt.title("Kinetix LSTM — Confusion Matrix (UCI HAR Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def copy_frontend_assets(metrics_path: Path, matrix_path: Path) -> None:
    FRONTEND_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    frontend_metrics = FRONTEND_ASSETS_DIR / "metrics.json"
    frontend_matrix = FRONTEND_ASSETS_DIR / "confusion_matrix.png"

    frontend_metrics.write_text(Path(metrics_path).read_text(encoding="utf-8"), encoding="utf-8")

    frontend_matrix.write_bytes(Path(matrix_path).read_bytes())

    print(f"Copied metrics to {frontend_metrics}")
    print(f"Copied confusion matrix to {frontend_matrix}")


def evaluate() -> dict:
    verify_scaler_fitted_on_training_data_only()

    raw_test = load_raw_uci_test_windows()
    y_true = load_uci_test_labels()

    print(f"Raw test shape: {raw_test.shape}")
    print(f"Raw sample shape: {raw_test[0].shape}")
    print(f"Scaled sample shape: {preprocess_sensor_data(raw_test[0]).shape[1:]}")
    print(f"Model input shape: {preprocess_sensor_data(raw_test[0]).shape}")

    x_test = np.vstack([preprocess_sensor_data(window) for window in raw_test])
    probabilities = model.predict(x_test, verbose=0)
    y_pred = probabilities.argmax(axis=1)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=ACTIVITY_NAMES,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred)

    per_class = {}
    for index, name in enumerate(ACTIVITY_NAMES):
        per_class[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1_score": float(f1[index]),
            "support": int(support[index]),
        }

    metrics_payload = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "raw UCI HAR test inertial -> StandardScaler (once) -> LSTM",
        "test_samples": int(len(y_true)),
        "accuracy": accuracy,
        "accuracy_percentage": round(accuracy * 100, 2),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "activity_labels": ACTIVITY_NAMES,
    }

    metrics_path = MODELS_DIR / "metrics.json"
    matrix_path = MODELS_DIR / "confusion_matrix.png"

    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    save_confusion_matrix_plot(matrix, str(matrix_path))
    copy_frontend_assets(metrics_path, matrix_path)

    print("\nTest Accuracy")
    print(f"{accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nMacro Precision / Recall / F1")
    print(f"{macro_precision:.4f} / {macro_recall:.4f} / {macro_f1:.4f}")

    print("\nClassification Report")
    print(report)

    print("Confusion Matrix")
    header = "             " + " ".join(f"{name[:8]:>8}" for name in ACTIVITY_NAMES)
    print(header)
    for index, row in enumerate(matrix):
        row_values = " ".join(f"{value:8d}" for value in row)
        print(f"{ACTIVITY_NAMES[index][:12]:12} {row_values}")

    first_truth = get_activity_name(int(y_true[0]))
    first_pred = ACTIVITY_NAMES[int(y_pred[0])]
    first_conf = float(probabilities[0][y_pred[0]])

    print("\nFirst sample")
    print(f"Ground truth: {first_truth}")
    print(f"Prediction:   {first_pred}")
    print(f"Confidence:   {first_conf:.4f}")

    return metrics_payload


if __name__ == "__main__":
    evaluate()
