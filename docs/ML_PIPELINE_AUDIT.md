# Kinetix ML Pipeline Audit

**Audit date:** 2026-08-12  
**Scope:** Existing training artifacts, UCI HAR dataset files, backend inference code, and notebooks. No code was modified during this audit.

**Sources inspected:**
- `notebooks/01_data_exploration.ipynb`, `02_model_training.ipynb`, `03_model_evaluation.ipynb`
- `backend/app/preprocessing.py`, `model.py`, `main.py`, `schemas.py`
- `models/kinetix_lstm.keras`, `models/scaler.pkl`, `models/X_train.npy`, `models/X_test.npy`, `models/X_test_raw.npy`, `models/y_train.npy`, `models/y_test.npy`, `models/metrics.json`
- `data/UCI HAR Dataset/README.txt`, `features_info.txt`, `activity_labels.txt`, inertial signal files under `train/Inertial Signals/` and `test/Inertial Signals/`

---

## 1. Dataset

| Item | Finding |
|------|---------|
| Dataset | UCI Human Activity Recognition Using Smartphones Dataset (Version 1.0) |
| Location | `data/UCI HAR Dataset/` |
| Subjects | 30 volunteers |
| Train/test split | 70% of subjects for training, 30% for test (per `README.txt`) |
| Saved arrays | `models/X_train.npy`: (7352, 128, 6); `models/X_test.npy`: (2947, 128, 6); labels `y_train.npy` / `y_test.npy` match UCI window counts |
| Window count verification | UCI `body_acc_x_test.txt` has 2947 rows; matches `X_test.npy` first dimension |

**UCI dataset preprocessing (from `README.txt`):**
- Smartphone: Samsung Galaxy S II, worn on the waist
- Sampling rate: **50 Hz**
- Noise filtering applied to raw signals
- Fixed-width sliding windows: **2.56 seconds**, **50% overlap**
- **128 readings per window**
- Total acceleration separated into **body acceleration** (gravity removed via 0.3 Hz Butterworth low-pass) and gravity acceleration
- The 561-feature vectors in `X_train.txt` / `X_test.txt` are **not** used by the Kinetix LSTM pipeline

---

## 2. Input Features

### A. Exactly what six input features were used during training?

Verified by reconstructing `X_train.npy` from UCI inertial files:

| Index | Feature | UCI source file (train) |
|-------|---------|-------------------------|
| 0 | Body acceleration X | `train/Inertial Signals/body_acc_x_train.txt` |
| 1 | Body acceleration Y | `train/Inertial Signals/body_acc_y_train.txt` |
| 2 | Body acceleration Z | `train/Inertial Signals/body_acc_z_train.txt` |
| 3 | Body gyroscope X | `train/Inertial Signals/body_gyro_x_train.txt` |
| 4 | Body gyroscope Y | `train/Inertial Signals/body_gyro_y_train.txt` |
| 5 | Body gyroscope Z | `train/Inertial Signals/body_gyro_z_train.txt` |

Feature order in each timestep: `[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]`.

### B. Raw inertial values or derived UCI features?

**Raw inertial time-series values** from the UCI *Inertial Signals* folders — **not** the 561 derived time/frequency domain features in `train/X_train.txt`.

These are **not** unprocessed phone sensor dumps. UCI applies dataset-level preprocessing before publishing:
- Median filter + 3rd-order low-pass Butterworth (20 Hz corner) on accelerometer and gyroscope
- Gravity separation from acceleration (0.3 Hz cutoff) to produce **body acceleration**

**Not used:**
- `total_acc_*` (total acceleration including gravity)
- `train/X_train.txt` / `test/X_test.txt` (561-feature vectors)
- Jerk, magnitude, FFT, or other derived features listed in `features_info.txt`

### Units (from `README.txt`)

| Signal | Unit |
|--------|------|
| Body acceleration | **g** (standard gravity, 9.80665 m/s²) |
| Gyroscope | **rad/s** |

---

## 3. Windowing

| Item | Value | Source |
|------|-------|--------|
| Timesteps per window | **128** | UCI README; confirmed in all `(N, 128, 6)` arrays |
| Window duration | **2.56 s** | UCI README (128 ÷ 50 Hz) |
| Overlap | **50%** | UCI README |
| Sampling frequency | **50 Hz** | UCI README, `features_info.txt` |
| Window creation code | Not determined from the current project files | Notebooks are stubs; no training script found in repo |

Each row in the UCI inertial `.txt` files is one pre-segmented 128-sample window aligned with an activity label.

---

## 4. Preprocessing

### E. Normalization / standardization during training

**Standardization (z-score)** using `sklearn.preprocessing.StandardScaler`.

Evidence:
- `models/scaler.pkl` unpickles to `StandardScaler` with `n_features_in_=6`
- Re-fitting `StandardScaler` on flattened UCI training inertial data reproduces `X_train.npy` exactly (`np.allclose(..., atol=1e-5)`)

`X_train.npy` per-feature statistics after saving: mean ≈ 0, std ≈ 1 across all timesteps.

### F. Which scaler was used?

`sklearn.preprocessing.StandardScaler` saved at `models/scaler.pkl`.

Saved parameters (6 features):

| Feature | mean_ | scale_ (std) |
|---------|-------|--------------|
| body_acc_x | -0.0006363 | 0.19484634 |
| body_acc_y | -0.0002923 | 0.12242748 |
| body_acc_z | -0.0002753 | 0.10687881 |
| body_gyro_x | 0.00050646 | 0.40681506 |
| body_gyro_y | -0.00082378 | 0.38185432 |
| body_gyro_z | 0.00011295 | 0.25574314 |

Scaler was pickled with scikit-learn **1.9.0** (version warning observed when loading with 1.7.0).

### G. How the scaler was fitted

Verified reconstruction:

```text
uci_train = stack(body_acc_x/y/z, body_gyro_x/y/z)   # shape (7352, 128, 6)
scaler.fit(uci_train.reshape(-1, 6))                 # 941,056 rows × 6 columns
X_train = scaler.transform(...).reshape(7352, 128, 6)
```

The scaler was fit on **all timesteps from all training windows** (flattened 2D), not per-window statistics.

### H. Scaler application during inference (`preprocessing.py`)

Current inference code:

```python
data_2d = data.reshape(-1, 6)
data_scaled = scaler.transform(data_2d)
data_scaled = data_scaled.reshape(1, 128, 6)
```

This is **correct** if the API receives **raw UCI-equivalent inertial values** (pre-scaler).

**Important:** `models/X_test.npy` already contains **post-scaler** values (`scaler.transform(uci_test)` matches `X_test.npy` exactly). The `/sample` endpoint serves `X_test.npy[0]`, so the live API path applies the scaler **a second time**. See Section 10.

---

## 5. LSTM Architecture

Extracted from `models/kinetix_lstm.keras` (loaded with TensorFlow/Keras 3.15.1).

### I. Exact LSTM input shape

**(batch, 128, 6)** — confirmed `model.input_shape = (None, 128, 6)`.

### J. Architecture details

| Layer | Type | Output shape | Key config |
|-------|------|--------------|------------|
| 0 | LSTM | (None, 128, 64) | 64 units, `return_sequences=True`, activation `tanh`, recurrent_activation `sigmoid`, **L2(0.001)** on kernel, dropout **0.0**, recurrent_dropout **0.0** |
| 1 | Dropout | (None, 128, 64) | rate **0.3** |
| 2 | LSTM | (None, 64) | 64 units, `return_sequences=False`, activation `tanh`, recurrent_activation `sigmoid`, **L2(0.001)** on kernel, dropout **0.0**, recurrent_dropout **0.0** |
| 3 | Dropout | (None, 64) | rate **0.3** |
| 4 | Dense | (None, 64) | 64 units, activation **relu**, **L2(0.001)** on kernel |
| 5 | Dense | (None, 6) | 6 units, activation **softmax**, no L2 on kernel |

**Summary:**
- **2 LSTM layers**, 64 units each
- **Dropout 0.3** after each LSTM (inference-time Dropout layers are present but inactive without training mode)
- **No recurrent dropout** (both LSTM layers: `recurrent_dropout=0.0`)
- **L2 regularization (0.001)** on LSTM kernels and first Dense kernel
- **Total parameters:** 55,750

**Output shape:** `(None, 6)` softmax probabilities.

---

## 6. Training Configuration

| Item | Value | Source |
|------|-------|--------|
| Optimizer (N) | **Adam** | `compile_config` in `kinetix_lstm.keras` |
| Learning rate | **0.001** | Same |
| Adam betas | beta_1=0.9, beta_2=0.999, epsilon=1e-7 | Same |
| Loss (O) | **sparse_categorical_crossentropy** | Same |
| Training metric | accuracy | Same |
| Epochs (L) | Not determined from the current project files | Not in model metadata or notebooks |
| Batch size (M) | Not determined from the current project files | Not in model metadata or notebooks |
| Training notebook | `notebooks/02_model_training.ipynb` contains only a title cell — no training code | |
| Evaluation notebook | `notebooks/03_model_evaluation.ipynb` contains only a title cell — no evaluation code | |
| Model saved | 2026-08-10 @ 20:23:11 | `metadata.json` inside `.keras` file |
| Keras version at save | 3.15.1 | Same |

Training data fed to the model: **`X_train.npy` (already scaled)** with **`y_train.npy`** (0-based labels). No additional scaling inside the training loop is evidenced in the repository.

---

## 7. Activity Labels

### K. Label index mapping

UCI `activity_labels.txt` uses **1-based** labels. Saved `y_*.npy` uses **0-based** (`UCI_label - 1`).

| Model output index | Saved `y_*` value | UCI label ID | Activity name (`model.py` / UCI) |
|--------------------|-------------------|--------------|----------------------------------|
| 0 | 0 | 1 | WALKING |
| 1 | 1 | 2 | WALKING_UPSTAIRS |
| 2 | 2 | 3 | WALKING_DOWNSTAIRS |
| 3 | 3 | 4 | SITTING |
| 4 | 4 | 5 | STANDING |
| 5 | 5 | 6 | LAYING |

`model.py` `ACTIVITY_NAMES` list order matches softmax output index order above.

---

## 8. Evaluation Results

### P. Test accuracy

From `models/metrics.json` and re-evaluation of saved model on `X_test.npy`:

| Metric | Value |
|--------|-------|
| **Test accuracy** | **59.96%** (0.5995928062) |

Verified by running `model.predict(X_test)` against `y_test.npy`.

**Note:** A ~97% test accuracy figure is **not** present in any project file inspected (`metrics.json`, notebooks, or saved artifacts).

### Q. Precision, recall, F1

Not stored in project files. Computed during audit from saved model + `X_test.npy` / `y_test.npy`:

| Activity | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| WALKING | 0.8525 | 0.9435 | 0.8957 | 496 |
| WALKING_UPSTAIRS | 0.9494 | 0.7962 | 0.8661 | 471 |
| WALKING_DOWNSTAIRS | 0.8864 | 0.9286 | 0.9070 | 420 |
| SITTING | 0.0000 | 0.0000 | 0.0000 | 491 |
| STANDING | 0.0000 | 0.0000 | 0.0000 | 532 |
| LAYING | 0.3417 | 0.9944 | 0.5086 | 537 |
| **Accuracy** | | | **0.5996** | 2947 |
| Macro avg | 0.5050 | 0.6105 | 0.5295 | 2947 |
| Weighted avg | 0.4838 | 0.5996 | 0.5111 | 2947 |

The model shows strong performance on walking classes and LAYING recall, but **never predicts SITTING or STANDING** on the test set with the correct single-scaling inference path.

Train accuracy (audit): **62.35%** on `X_train.npy`.

---

## 9. Inference Pipeline

End-to-end path as implemented today:

```text
GET /sample
  → loads models/X_test.npy[0]   (already StandardScaler-transformed)
  → returns as sensor_data

POST /predict { sensor_data }
  → preprocessing.py: validate shape (128, 6)
  → StandardScaler.transform (again)
  → reshape (1, 128, 6)
  → model.predict → argmax → ACTIVITY_NAMES[index]
```

**Expected correct path for raw inertial input:**

```text
raw window (128, 6) in UCI body_acc/gyro units
  → StandardScaler.transform once
  → reshape (1, 128, 6)
  → model.predict
```

**Alternative correct path for pre-scaled input:**

```text
pre-scaled window (128, 6) identical to X_train/X_test
  → model.predict directly (no scaler)
```

---

## 10. Training vs Inference Consistency

### R. Mismatch analysis

| Aspect | Training | Current API inference | Consistent? |
|--------|----------|----------------------|-------------|
| Feature type | UCI body acc + gyro inertial | Assumed same (documented as acc/gyro) | Intended yes |
| Feature order | acc_xyz, gyro_xyz | Same order in API schema/docs | Yes |
| Scaler | Fit on train raw inertial; applied once before model | Applied in `preprocessing.py` | Yes **if input is raw** |
| `/sample` data | N/A | Serves **pre-scaled** `X_test.npy[0]` | **No — double scaling** |
| Model input | Pre-scaled `X_train` / `X_test` | Scaler applied to already-scaled sample | **No** |

**Verified with window index 0:**

| Path | Prediction | Confidence | True label (`y_test[0]`) |
|------|------------|------------|--------------------------|
| API path: `X_test[0]` → scaler → model | WALKING (0) | 81.01% | STANDING (4) |
| Correct: `X_test[0]` → model (no scaler) | LAYING (5) | 36.32% | STANDING (4) |
| Correct: UCI raw window 0 → scaler → model | LAYING (5) | 36.32% | STANDING (4) |

Double-scaling reduces test accuracy from **59.96%** to **49.44%** across the full test set.

**Additional note:** `models/X_test_raw.npy` is **byte-identical** to `X_test.npy` (both are scaled). The filename suggests raw data but the file is not raw.

---

## 11. Requirements for Live Smartphone Sensor Input

To convert live smartphone data into the exact **(128, 6)** tensor the LSTM expects, the following must match the UCI training pipeline.

### 11.1 Required signals (6 features)

Provide, in this order, for each of 128 consecutive timesteps:

1. Body acceleration X (**g**)
2. Body acceleration Y (**g**)
3. Body acceleration Z (**g**)
4. Gyroscope angular velocity X (**rad/s**)
5. Gyroscope angular velocity Y (**rad/s**)
6. Gyroscope angular velocity Z (**rad/s**)

### 11.2 What phone sensors typically provide vs what the model needs

| Phone output | Compatible as-is? | Required processing |
|--------------|-------------------|---------------------|
| Raw accelerometer (includes gravity) | **No** | Gravity separation to body acceleration (UCI uses 0.3 Hz Butterworth low-pass); optionally median + 20 Hz low-pass denoising as in UCI |
| Linear acceleration / "user acceleration" (OS-provided gravity removal) | **Partially** | Must verify axis orientation, units converted to **g**, and comparable filtering |
| Raw gyroscope | **Partially** | Convert to **rad/s** if device reports deg/s; apply UCI-equivalent denoising if matching training domain |
| Total acceleration (`total_acc_*`) | **No** | Correlation with body acc X is ~0.45; values differ by ~1 g bias |
| Derived features (jerk, FFT, magnitude) | **No** | Not used in this model |
| 561-feature UCI vectors | **No** | Different representation entirely |

### 11.3 Sampling and windowing

| Requirement | Value |
|-------------|-------|
| Sample rate | **50 Hz** (or resample/interpolate to 50 Hz) |
| Window length | **128 samples** (= 2.56 seconds at 50 Hz) |
| Overlap | 50% overlap if sliding windows are used (matches UCI segmentation) |
| Shape per request | `(128, 6)` — list of 128 rows, each with 6 floats |

### 11.4 Standardization at inference

1. Build raw/window `(128, 6)` in UCI-equivalent units and feature order.
2. Apply **`models/scaler.pkl`** exactly once: `scaler.transform(data.reshape(-1, 6)).reshape(1, 128, 6)`.
3. Pass result to the LSTM.

Do **not** apply the scaler to data that is already standardized like `X_train.npy` / `X_test.npy`.

### 11.5 Coordinate frame and placement

UCI data was collected with a Samsung Galaxy S II **on the waist**. Live data should use a **consistent device orientation and body frame** relative to training, or predictions may degrade. Specific axis alignment procedure is **not determined from the current project files**.

### 11.6 Compatibility verdict for direct raw smartphone readings

**The current model is not compatible with direct, unprocessed smartphone accelerometer + gyroscope readings.**

Minimum gap between a typical phone API and model input:

1. **Gravity removal** — model trained on body acceleration, not total acceleration.
2. **Units** — acceleration in **g**, gyroscope in **rad/s**.
3. **Sample rate** — **50 Hz**, **128-sample** windows.
4. **UCI-equivalent filtering** — training data was denoised and gravity-separated per UCI protocol.
5. **StandardScaler** — must use the saved scaler fit on UCI training inertial data, applied **once**.
6. **Current API bug** — `/sample` returns pre-scaled data; live integration must send **raw-equivalent** values into `preprocessing.py`, or `/sample` must be changed to serve raw UCI windows (out of scope for this audit).

### 11.7 Recommended live-data pipeline (future work)

```text
Phone IMU (acc + gyro)
  → unit conversion (g, rad/s)
  → axis/orientation normalization
  → gravity separation (body acceleration)
  → optional UCI-matched filtering
  → buffer at 50 Hz
  → extract 128-sample window
  → StandardScaler.transform (models/scaler.pkl)
  → LSTM predict
```

---

## Audit Limitations

The following could **not** be determined because training/evaluation code is not present in the repository (notebooks are placeholders):

- Exact training script that produced `X_*.npy` and `scaler.pkl`
- Number of training epochs
- Batch size
- Validation split strategy during training
- Device placement / sensor axis calibration documentation
- Why reported project accuracy (~97%) differs from saved `metrics.json` (~60%)

All architectural, preprocessing, and evaluation findings above are derived from **saved artifacts**, **UCI dataset documentation**, **backend inference code**, and **reproducibility checks** run against those artifacts.

---

## Corrected Inference Pipeline

### Previous incorrect behavior (fixed)

The original `/sample` endpoint loaded `models/X_test.npy[0]`, which contains **already StandardScaler-transformed** test data. `preprocessing.py` then applied `StandardScaler.transform()` again before inference. This **double scaling** produced misleading demo predictions (for example, WALKING at ~81% on a window whose UCI ground truth is STANDING).

### Correct pipeline (current backend)

```text
Raw UCI HAR body acceleration + body gyroscope (128 × 6)
  → shape validation (128, 6)
  → saved StandardScaler.transform() exactly once (models/scaler.pkl)
  → reshape to (1, 128, 6)
  → 2-layer LSTM
  → six-class softmax prediction
```

**`/sample` source:** raw UCI test inertial files under `data/UCI HAR Dataset/test/Inertial Signals/`:

- `body_acc_x_test.txt`, `body_acc_y_test.txt`, `body_acc_z_test.txt`
- `body_gyro_x_test.txt`, `body_gyro_y_test.txt`, `body_gyro_z_test.txt`

Ground truth comes from `data/UCI HAR Dataset/test/y_test.txt` mapped through `activity_labels.txt`.

**`/predict` input:** raw UCI-equivalent windows only. Do not send pre-scaled `X_test.npy` values.

**Evaluation:** run `python backend/evaluate_model.py` to measure test accuracy on the full raw UCI test set using this pipeline.

**Measured test accuracy with correct inference:** **59.96%** (macro F1: **52.95%**, weighted F1: **51.11%**). Fixing double scaling did not improve accuracy; it made inference mathematically correct.

Evaluation artifacts (generated by `backend/evaluate_model.py`):

- `models/metrics.json` — accuracy, macro/weighted precision/recall/F1, per-class metrics
- `models/confusion_matrix.png` — confusion matrix plot
- Copies synced to `frontend/public/assets/` for the dashboard
