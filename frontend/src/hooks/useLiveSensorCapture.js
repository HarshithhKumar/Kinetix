import { useCallback, useEffect, useRef, useState } from "react";
import {
  SAMPLE_INTERVAL_MS,
  WINDOW_SIZE,
  createGravityState,
  detectSensorSupport,
  motionEventToSample,
  requestMotionPermission,
  validateWindow,
} from "../services/liveSensors";

const CHART_UPDATE_EVERY = 8;

/**
 * Live sensor capture hook.
 * Sensor events update refs; React state updates are throttled.
 */
export function useLiveSensorCapture() {
  const support = detectSensorSupport();

  const [status, setStatus] = useState(() =>
    support.supported ? "permission_required" : "not_supported",
  );
  const [permissionGranted, setPermissionGranted] = useState(false);
  const [accelAvailable, setAccelAvailable] = useState(false);
  const [gyroAvailable, setGyroAvailable] = useState(false);
  const [sampleCount, setSampleCount] = useState(0);
  const [chartPreview, setChartPreview] = useState([]);
  const [capturedWindow, setCapturedWindow] = useState([]);
  const [error, setError] = useState(support.reason || "");

  const samplesRef = useRef([]);
  const gravityRef = useRef(createGravityState());
  const latestMotionRef = useRef(null);
  const sampleTimerRef = useRef(null);
  const motionHandlerRef = useRef(null);
  const statusRef = useRef(status);

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  const cleanupCapture = useCallback(() => {
    if (sampleTimerRef.current) {
      clearInterval(sampleTimerRef.current);
      sampleTimerRef.current = null;
    }
    if (motionHandlerRef.current) {
      window.removeEventListener("devicemotion", motionHandlerRef.current);
      motionHandlerRef.current = null;
    }
  }, []);

  useEffect(() => cleanupCapture, [cleanupCapture]);

  const attachMotionListener = useCallback(() => {
    if (motionHandlerRef.current) return;

    const handler = (event) => {
      latestMotionRef.current = event;

      const { hasAccel, hasGyro } = motionEventToSample(
        event,
        gravityRef.current,
      );

      if (hasAccel) setAccelAvailable(true);
      if (hasGyro) setGyroAvailable(true);
    };

    motionHandlerRef.current = handler;
    window.addEventListener("devicemotion", handler, true);
  }, []);

  const enableMotion = useCallback(async () => {
    setError("");

    if (!support.supported) {
      setStatus("not_supported");
      setError(
        support.reason ||
          "Live sensors are not supported on this device/browser.",
      );
      return false;
    }

    if (support.needsPermission) {
      const result = await requestMotionPermission();
      if (!result.granted) {
        setPermissionGranted(false);
        setStatus("permission_required");
        setError(result.error || "Motion sensor permission denied.");
        return false;
      }
    }

    setPermissionGranted(true);
    attachMotionListener();
    setStatus("ready");
    return true;
  }, [attachMotionListener, support]);

  const resetCapture = useCallback(() => {
    cleanupCapture();
    samplesRef.current = [];
    gravityRef.current = createGravityState();
    latestMotionRef.current = null;
    setSampleCount(0);
    setChartPreview([]);
    setCapturedWindow([]);
    setError("");

    if (!support.supported) {
      setStatus("not_supported");
      return;
    }

    setStatus(permissionGranted ? "ready" : "permission_required");
  }, [cleanupCapture, permissionGranted, support.supported]);

  const startCapture = useCallback(() => {
    if (!permissionGranted) {
      setError("Enable motion sensors before capturing.");
      setStatus("permission_required");
      return;
    }

    cleanupCapture();
    samplesRef.current = [];
    gravityRef.current = createGravityState();
    setSampleCount(0);
    setChartPreview([]);
    setCapturedWindow([]);
    setError("");
    setStatus("collecting");
    attachMotionListener();

    sampleTimerRef.current = setInterval(() => {
      const event = latestMotionRef.current;
      if (!event) return;

      const { sample, hasAccel, hasGyro } = motionEventToSample(
        event,
        gravityRef.current,
      );

      if (!hasAccel || !hasGyro) {
        return;
      }

      const count = samplesRef.current.length;
      if (count >= WINDOW_SIZE) return;

      samplesRef.current.push(sample);

      const newCount = samplesRef.current.length;
      if (newCount % CHART_UPDATE_EVERY === 0 || newCount === WINDOW_SIZE) {
        setChartPreview(samplesRef.current.slice());
        setSampleCount(newCount);
      } else {
        setSampleCount(newCount);
      }

      if (newCount >= WINDOW_SIZE) {
        cleanupCapture();
        const windowData = samplesRef.current.slice();
        setCapturedWindow(windowData);
        setChartPreview(windowData);
        setStatus("window_ready");
      }
    }, SAMPLE_INTERVAL_MS);

    setTimeout(() => {
      if (
        statusRef.current === "collecting" &&
        samplesRef.current.length === 0
      ) {
        cleanupCapture();
        setStatus("error");
        setError(
          "No motion sensor data received. Use a phone with accelerometer and gyroscope.",
        );
      }
    }, 4000);
  }, [attachMotionListener, cleanupCapture, permissionGranted]);

  const getValidatedWindow = useCallback(() => {
    const result = validateWindow(capturedWindow);
    return result;
  }, [capturedWindow]);

  return {
    support,
    status,
    setStatus,
    permissionGranted,
    accelAvailable,
    gyroAvailable,
    sampleCount,
    chartPreview,
    capturedWindow,
    error,
    setError,
    enableMotion,
    startCapture,
    resetCapture,
    getValidatedWindow,
    cleanupCapture,
  };
}
