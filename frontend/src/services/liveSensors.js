/**
 * Browser motion sensor utilities for Kinetix live mode.
 *
 * Transforms DeviceMotionEvent readings into UCI HAR–like feature units:
 *   body acceleration (g) + body gyroscope (rad/s)
 *
 * This is an experimental approximation — see docs/LIVE_SENSOR_PIPELINE.md.
 */

export const WINDOW_SIZE = 128;
export const TARGET_HZ = 50;
export const SAMPLE_INTERVAL_MS = 1000 / TARGET_HZ;
export const GRAVITY_MS2 = 9.80665;
export const DEG_TO_RAD = Math.PI / 180;

/** Low-pass coefficient (~0.3 Hz cutoff at 50 Hz) for gravity estimation. */
export const GRAVITY_LP_ALPHA = 0.04;

export const FEATURE_NAMES = [
  "body_acc_x",
  "body_acc_y",
  "body_acc_z",
  "body_gyro_x",
  "body_gyro_y",
  "body_gyro_z",
];

export function detectSensorSupport() {
  if (typeof window === "undefined") {
    return {
      supported: false,
      needsPermission: false,
      reason: "Not running in a browser.",
    };
  }

  if (!("DeviceMotionEvent" in window)) {
    return {
      supported: false,
      needsPermission: false,
      reason: "DeviceMotionEvent is not available on this browser.",
    };
  }

  const needsPermission =
    typeof DeviceMotionEvent.requestPermission === "function";

  return {
    supported: true,
    needsPermission,
    reason: null,
  };
}

/**
 * @returns {{ x: number, y: number, z: number }}
 */
export function createGravityState() {
  return { x: 0, y: 0, z: 0, initialized: false };
}

/**
 * Convert a DeviceMotionEvent into one 6-feature sample in UCI-like units.
 *
 * Accelerometer:
 *   - Prefer gravity-free `acceleration` when present (m/s² → g).
 *   - Otherwise high-pass `accelerationIncludingGravity` via low-pass gravity estimate.
 *
 * Gyroscope:
 *   - `rotationRate` alpha/beta/gamma (deg/s) → rad/s mapped to z/x/y axes.
 */
export function motionEventToSample(event, gravityState) {
  const acc = event.acceleration;
  const accGrav = event.accelerationIncludingGravity;
  const rot = event.rotationRate;

  let ax = 0;
  let ay = 0;
  let az = 0;
  let hasAccel = false;

  if (acc && acc.x != null && acc.y != null && acc.z != null) {
    ax = acc.x / GRAVITY_MS2;
    ay = acc.y / GRAVITY_MS2;
    az = acc.z / GRAVITY_MS2;
    hasAccel = true;
  } else if (
    accGrav &&
    accGrav.x != null &&
    accGrav.y != null &&
    accGrav.z != null
  ) {
    if (!gravityState.initialized) {
      gravityState.x = accGrav.x;
      gravityState.y = accGrav.y;
      gravityState.z = accGrav.z;
      gravityState.initialized = true;
    } else {
      gravityState.x =
        GRAVITY_LP_ALPHA * accGrav.x +
        (1 - GRAVITY_LP_ALPHA) * gravityState.x;
      gravityState.y =
        GRAVITY_LP_ALPHA * accGrav.y +
        (1 - GRAVITY_LP_ALPHA) * gravityState.y;
      gravityState.z =
        GRAVITY_LP_ALPHA * accGrav.z +
        (1 - GRAVITY_LP_ALPHA) * gravityState.z;
    }

    ax = (accGrav.x - gravityState.x) / GRAVITY_MS2;
    ay = (accGrav.y - gravityState.y) / GRAVITY_MS2;
    az = (accGrav.z - gravityState.z) / GRAVITY_MS2;
    hasAccel = true;
  }

  let gx = 0;
  let gy = 0;
  let gz = 0;
  let hasGyro = false;

  if (rot && rot.alpha != null && rot.beta != null && rot.gamma != null) {
    gx = rot.beta * DEG_TO_RAD;
    gy = rot.gamma * DEG_TO_RAD;
    gz = rot.alpha * DEG_TO_RAD;
    hasGyro = true;
  }

  return {
    sample: [ax, ay, az, gx, gy, gz],
    hasAccel,
    hasGyro,
  };
}

export function validateWindow(window) {
  if (!Array.isArray(window) || window.length !== WINDOW_SIZE) {
    return { valid: false, error: `Expected ${WINDOW_SIZE} timesteps.` };
  }

  for (let i = 0; i < window.length; i += 1) {
    const row = window[i];
    if (!Array.isArray(row) || row.length !== 6) {
      return { valid: false, error: `Timestep ${i} is not length 6.` };
    }
    if (row.some((v) => typeof v !== "number" || !Number.isFinite(v))) {
      return { valid: false, error: `Timestep ${i} contains invalid values.` };
    }
  }

  return { valid: true, error: null };
}

export async function requestMotionPermission() {
  if (typeof DeviceMotionEvent.requestPermission !== "function") {
    return { granted: true, error: null };
  }

  try {
    const state = await DeviceMotionEvent.requestPermission();
    if (state === "granted") {
      return { granted: true, error: null };
    }
    return { granted: false, error: "Motion sensor permission denied." };
  } catch (err) {
    return {
      granted: false,
      error: err?.message || "Failed to request motion permission.",
    };
  }
}
