"""
Reproducible Kinetix LSTM training on raw UCI HAR inertial signals.

Pipeline:
  raw UCI train/test inertial (128, 6)
  -> StandardScaler fit on TRAIN only
  -> stratified train/validation split (from official train set)
  -> LSTM training with early stopping
  -> single final evaluation on official test set
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from datetime import datetime, timezone
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

RANDOM_SEED = 42
VALIDATION_SIZE = 0.15
BATCH_SIZE = 64
MAX_EPOCHS = 80
LEARNING_RATE = 0.001
L2_REG = 0.001

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

FEATURE_NAMES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "UCI HAR Dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FRONTEND_ASSETS_DIR = os.path.join(BASE_DIR, "frontend", "public", "assets")

TRAIN_SIGNAL_FILES = [
    "body_acc_x_train.txt",
    "body_acc_y_train.txt",
    "body_acc_z_train.txt",
    "body_gyro_x_train.txt",
    "body_gyro_y_train.txt",
    "body_gyro_z_train.txt",
]

TEST_SIGNAL_FILES = [name.replace("_train", "_test") for name in TRAIN_SIGNAL_FILES]


def set_seeds(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_inertial_windows(split: str) -> np.ndarray:
    folder = os.path.join(DATA_DIR, split, "Inertial Signals")
    files = TRAIN_SIGNAL_FILES if split == "train" else TEST_SIGNAL_FILES
    signals = [np.loadtxt(os.path.join(folder, filename)) for filename in files]
    return np.stack(signals, axis=-1).astype(np.float32)


def load_labels(split: str) -> np.ndarray:
    path = os.path.join(DATA_DIR, split, f"y_{split}.txt")
    labels = np.loadtxt(path, dtype=int)
    return (labels - 1).astype(int)


def scale_windows(raw: np.ndarray, scaler: StandardScaler, fit: bool = False) -> np.ndarray:
    flat = raw.reshape(-1, 6)
    if fit:
        scaled = scaler.fit_transform(flat)
    else:
        scaled = scaler.transform(flat)
    return scaled.reshape(raw.shape).astype(np.float32)


def inspect_dataset(y_train: np.ndarray, y_test: np.ndarray) -> None:
    print("\n=== Dataset inspection ===")
    print(f"Train windows: {len(y_train)}")
    print(f"Test windows:  {len(y_test)}")
    print("\nTrain class distribution:")
    for index, name in enumerate(ACTIVITY_NAMES):
        count = int(np.sum(y_train == index))
        pct = 100.0 * count / len(y_train)
        print(f"  {index} {name:22} {count:5} ({pct:5.2f}%)")
    print("\nTest class distribution:")
    for index, name in enumerate(ACTIVITY_NAMES):
        count = int(np.sum(y_test == index))
        pct = 100.0 * count / len(y_test)
        print(f"  {index} {name:22} {count:5} ({pct:5.2f}%)")


def build_model() -> tf.keras.Model:
    """
    Bidirectional first LSTM to capture forward/backward temporal context.
    Targeted at static-activity separation (SITTING/STANDING/LAYING), which
    share similar low-motion accelerometer/gyroscope patterns.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(128, 6)),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    128,
                    return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
                )
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(
                64,
                kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
            ),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
            ),
            tf.keras.layers.Dense(len(ACTIVITY_NAMES), activation="softmax"),
        ],
        name="kinetix_lstm",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_balanced_class_weights(y: np.ndarray) -> dict[int, float]:
    classes = np.arange(len(ACTIVITY_NAMES))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def save_training_history(history: tf.keras.callbacks.History, models_dir: str) -> dict:
    hist = {key: [float(v) for v in values] for key, values in history.history.items()}
    history_path = os.path.join(models_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(hist, file, indent=2)

    epochs = range(1, len(hist["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, hist["loss"], label="train")
    axes[0].plot(epochs, hist["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, hist["accuracy"], label="train")
    axes[1].plot(epochs, hist["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(models_dir, "training_history.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return hist


def save_model_summary(model: tf.keras.Model, models_dir: str, meta: dict) -> None:
    buffer = StringIO()
    model.summary(print_fn=lambda line: buffer.write(line + "\n"))
    summary_text = buffer.getvalue()

    lines = [
        summary_text,
        "",
        "Training metadata",
        f"Optimizer: Adam",
        f"Learning rate: {meta['learning_rate']}",
        f"Batch size: {meta['batch_size']}",
        f"Epochs trained: {meta['epochs_trained']}",
        f"Best validation epoch: {meta['best_epoch']}",
        f"Parameters: {model.count_params()}",
        f"Input shape: (128, 6)",
        f"Output classes: {len(ACTIVITY_NAMES)}",
        f"Class weights used: {meta['used_class_weights']}",
    ]

    with open(os.path.join(models_dir, "model_summary.txt"), "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


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


def copy_frontend_assets(metrics_path: str, matrix_path: str, history_path: str) -> None:
    os.makedirs(FRONTEND_ASSETS_DIR, exist_ok=True)
    with open(metrics_path, "r", encoding="utf-8") as src:
        payload = src.read()
    with open(os.path.join(FRONTEND_ASSETS_DIR, "metrics.json"), "w", encoding="utf-8") as dst:
        dst.write(payload)
    with open(matrix_path, "rb") as src, open(
        os.path.join(FRONTEND_ASSETS_DIR, "confusion_matrix.png"), "wb"
    ) as dst:
        dst.write(src.read())
    with open(history_path, "r", encoding="utf-8") as src:
        history_payload = src.read()
    with open(
        os.path.join(FRONTEND_ASSETS_DIR, "training_history.json"), "w", encoding="utf-8"
    ) as dst:
        dst.write(history_payload)


def evaluate_on_test(model: tf.keras.Model, x_test: np.ndarray, y_test: np.ndarray, meta: dict) -> dict:
    probabilities = model.predict(x_test, verbose=0)
    y_pred = probabilities.argmax(axis=1)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=ACTIVITY_NAMES, digits=4, zero_division=0
    )

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
        "test_samples": int(len(y_test)),
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
        "training": meta,
    }

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    matrix_path = os.path.join(MODELS_DIR, "confusion_matrix.png")

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

    save_confusion_matrix_plot(matrix, matrix_path)
    history_path = os.path.join(MODELS_DIR, "training_history.json")
    copy_frontend_assets(metrics_path, matrix_path, history_path)

    print("\n=== Official test set evaluation ===")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro P/R/F1: {macro_precision:.4f} / {macro_recall:.4f} / {macro_f1:.4f}")
    print("\nClassification report:")
    print(report)
    print("Confusion matrix:")
    print(matrix)

    return metrics_payload


def train() -> dict:
    set_seeds()
    os.makedirs(MODELS_DIR, exist_ok=True)

    x_train_raw = load_inertial_windows("train")
    x_test_raw = load_inertial_windows("test")
    y_train_full = load_labels("train")
    y_test = load_labels("test")

    inspect_dataset(y_train_full, y_test)

    scaler = StandardScaler()
    x_train_scaled_full = scale_windows(x_train_raw, scaler, fit=True)
    x_test_scaled = scale_windows(x_test_raw, scaler, fit=False)

    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as file:
        pickle.dump(scaler, file)
    print(f"\nSaved scaler to {scaler_path}")

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_scaled_full,
        y_train_full,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_train_full,
    )

    print(f"\nTrain split: {len(x_train)} | Validation: {len(x_val)} | Test: {len(x_test_scaled)}")

    class_weights = compute_balanced_class_weights(y_train)
    print("\nBalanced class weights (train split only):")
    for index, name in enumerate(ACTIVITY_NAMES):
        print(f"  {name}: {class_weights[index]:.4f}")

    model = build_model()
    checkpoint_path = os.path.join(MODELS_DIR, "kinetix_lstm_best.keras")
    final_model_path = os.path.join(MODELS_DIR, "kinetix_lstm.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\n=== Training ===")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    hist = save_training_history(history, MODELS_DIR)
    epochs_trained = len(hist["loss"])
    best_epoch = int(np.argmin(hist["val_loss"]) + 1)

    model.save(final_model_path)
    print(f"\nSaved model to {final_model_path}")

    training_meta = {
        "architecture": "BiLSTM(128) -> BN -> Dropout -> LSTM(64) -> Dropout -> Dense(64) -> Dense(6)",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs_trained": epochs_trained,
        "best_epoch": best_epoch,
        "max_epochs": MAX_EPOCHS,
        "parameters": int(model.count_params()),
        "used_class_weights": True,
        "validation_size": VALIDATION_SIZE,
        "random_seed": RANDOM_SEED,
        "final_train_accuracy": hist["accuracy"][-1],
        "final_val_accuracy": hist["val_accuracy"][-1],
        "final_train_loss": hist["loss"][-1],
        "final_val_loss": hist["val_loss"][-1],
    }

    save_model_summary(model, MODELS_DIR, training_meta)

    metrics = evaluate_on_test(model, x_test_scaled, y_test, training_meta)
    return metrics


if __name__ == "__main__":
    result = train()
    print("\nTraining complete.")
    print(f"Final test accuracy: {result['accuracy_percentage']}%")
    sys.exit(0)
