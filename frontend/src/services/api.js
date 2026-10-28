import axios from "axios";

const DEFAULT_API_URL = "http://localhost:8000";

function resolveApiBaseUrl() {
  const configured = import.meta.env.VITE_API_URL?.trim();
  if (configured) {
    return configured.replace(/\/$/, "");
  }
  return DEFAULT_API_URL;
}

const API_BASE_URL = resolveApiBaseUrl();
const REQUEST_TIMEOUT_MS = 10000;

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: REQUEST_TIMEOUT_MS,
  headers: {
    "Content-Type": "application/json",
  },
});

export const checkHealth = () => api.get("/health");

export const getSampleSensorData = () => api.get("/sample");

export const predictActivity = (sensorData, options = {}) =>
  api.post("/predict", {
    sensor_data: sensorData,
    ...(options.source ? { input_source: options.source } : {}),
  });

export const getMetrics = () =>
  axios.get("/assets/metrics.json", { timeout: REQUEST_TIMEOUT_MS });

export const getTrainingHistory = () =>
  axios.get("/assets/training_history.json", { timeout: REQUEST_TIMEOUT_MS });

export { API_BASE_URL, DEFAULT_API_URL, REQUEST_TIMEOUT_MS };
