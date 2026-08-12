# Demo Sample Validation

Validated: 2026-08-12

## Summary

The `/sample` endpoint returns **UCI HAR test window index 0**. The inference pipeline is correct. The model predicts **SITTING** for a window whose ground truth is **STANDING** — this is a normal individual misclassification, not a pipeline bug.

## Checks

| Check | Result |
|-------|--------|
| Raw window matches UCI inertial files | Yes |
| `y_test.txt` label (1-based) | 5 → STANDING |
| Label mapping (0-based index) | 4 → STANDING |
| Feature order | body_acc_x/y/z, body_gyro_x/y/z |
| StandardScaler applied once | Yes |
| Model input shape | (1, 128, 6) |
| Full-test eval for index 0 | SITTING (same as API) |
| Overall test accuracy | 86.90% |

## Demo sample probabilities

| Activity | Probability |
|----------|------------:|
| WALKING | 0.0001% |
| WALKING_UPSTAIRS | 0.0032% |
| WALKING_DOWNSTAIRS | 0.0002% |
| SITTING | 91.31% |
| STANDING | 6.32% |
| LAYING | 2.36% |

## Context

On the full test set, **59 of 532 STANDING windows** are misclassified as SITTING. Static activities (SITTING/STANDING/LAYING) share similar low-motion IMU patterns; the demo sample falls into this known confusion region.

No model or pipeline changes required.
