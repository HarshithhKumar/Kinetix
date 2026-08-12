# Live Sensor Pipeline

**Status:** Experimental  
**Last updated:** 2026-08-12

Kinetix live phone mode collects motion data in the browser, builds a 128×6 window, and sends it to the same `POST /predict` endpoint used by the Dataset Demo. The trained BiLSTM and saved `StandardScaler` are unchanged.

---

## Overview

```
Phone browser (DeviceMotionEvent)
  → gravity removal + unit conversion (frontend, experimental)
  → 128 × 6 window (acc_x/y/z, gyro_x/y/z)
  → POST /predict (raw window, no frontend scaling)
  → StandardScaler (train-fit, backend)
  → BiLSTM
  → 6-class softmax
```

Live mode is explicitly marked **Experimental**. Browser preprocessing approximates UCI HAR body signals; it does **not** exactly reproduce the dataset pipeline unless validated against labeled phone data.

---

## Browser Sensor APIs

| API | Usage |
|-----|--------|
| `DeviceMotionEvent` | Primary source for accelerometer and gyroscope |
| `DeviceMotionEvent.requestPermission()` | Required on iOS 13+ Safari before motion events fire |

**Not used:** geolocation, camera, microphone, DeviceOrientation alone.

### Accelerometer

| Source field | Description |
|--------------|-------------|
| `acceleration` | Gravity-free acceleration (m/s²) when provided |
| `accelerationIncludingGravity` | Fallback; gravity estimated and subtracted |

### Gyroscope

| Source field | Maps to |
|--------------|---------|
| `rotationRate.beta` | gyro_x |
| `rotationRate.gamma` | gyro_y |
| `rotationRate.alpha` | gyro_z |

Values are in **degrees per second** in most browsers.

---

## Sampling

| Parameter | Value |
|-----------|-------|
| Target rate | ~50 Hz |
| Interval | 20 ms (`setInterval`) |
| Window size | 128 samples (~2.56 s) |
| Stop condition | Auto-stop at 128 samples |

Latest `devicemotion` reading is sampled on each tick. Sensor events update refs; React state updates are throttled (every 8 samples during capture) to avoid rendering on every event.

---

## Gravity Handling (Experimental)

UCI HAR uses **body acceleration** with gravity removed (0.3 Hz Butterworth low-pass on total acceleration in the original dataset pipeline).

Live mode first-pass approach:

1. **Prefer** `event.acceleration` when non-null → divide by 9.80665 to convert m/s² → **g**.
2. **Fallback** when only `accelerationIncludingGravity` is available:
   - Estimate gravity with exponential low-pass filter (α ≈ 0.04, ~0.3 Hz at 50 Hz).
   - Body acceleration = (including_gravity − gravity_estimate) / 9.80665.

This is a simplified approximation. It does not replicate UCI’s median filter, 20 Hz low-pass, or exact gravity separation.

---

## Gyroscope Unit Conversion

| Browser | Model expects |
|---------|---------------|
| deg/s (`rotationRate.*`) | rad/s (UCI gyroscope unit) |

Conversion: `rad/s = deg/s × (π / 180)`

---

## Feature Order (128 × 6)

Each timestep:

```
[body_acc_x, body_acc_y, body_acc_z, body_gyro_x, body_gyro_y, body_gyro_z]
```

Matches UCI inertial signal order and backend training pipeline.

---

## Backend / StandardScaler

- Frontend sends **unscaled** 128×6 windows.
- `backend/app/preprocessing.py` applies the training-fitted `StandardScaler` once.
- Optional request field `input_source: "live" | "dataset"` is accepted for tracing; preprocessing is identical.

---

## Mobile Testing

1. Run backend on all interfaces:
   ```bash
   cd backend
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
2. Run frontend with network access:
   ```bash
   cd frontend
   npm run dev
   ```
3. Open `http://<LAN-IP>:5173` on the phone (HTTPS may be required on some iOS versions for motion APIs outside localhost).
4. Tap **Enable Motion Sensors** → **Start Capture** → **Run Prediction**.

The frontend API base URL defaults to `http://<same-hostname>:8000`.

---

## Current Limitations

| Limitation | Detail |
|------------|--------|
| Not UCI-identical | Browser gravity removal ≠ UCI Butterworth pipeline |
| Axis alignment | Phone orientation may differ from UCI waist-mounted Samsung Galaxy S II |
| No live accuracy claim | Predictions are not evaluated against labeled live data |
| Desktop browsers | Typically lack motion hardware; mode shows “not supported” |
| iOS permission | User must grant motion access; denied permission is handled gracefully |
| HTTPS | Some devices require secure context for motion sensors |

---

## Files

| File | Role |
|------|------|
| `frontend/src/services/liveSensors.js` | Sensor detection, transforms, validation |
| `frontend/src/hooks/useLiveSensorCapture.js` | Capture state machine |
| `frontend/src/components/LiveSensorPanel.jsx` | Live UI |
| `frontend/src/components/InputModeSelector.jsx` | Demo / Live toggle |
| `backend/app/schemas.py` | Optional `input_source` on predict request |
