import { useCallback, useEffect, useState } from "react";
import { BookOpen, Menu } from "lucide-react";
import AnalyticsPage from "./pages/AnalyticsPage";
import Header from "./components/Header";
import HistoryPage from "./pages/HistoryPage";
import OverviewPage from "./pages/OverviewPage";
import PredictPage from "./pages/PredictPage";
import SensorsPage from "./pages/SensorsPage";
import StatusPage from "./pages/StatusPage";
import { useLiveSensorCapture } from "./hooks/useLiveSensorCapture";
import {
  API_BASE_URL,
  checkHealth,
  getMetrics,
  getSampleSensorData,
  getTrainingHistory,
  predictActivity,
} from "./services/api";
import { validateWindow } from "./services/liveSensors";
import { APP_ROUTES, getRouteTitle, resolveRoute } from "./lib/routes";
import "./App.css";

const SESSION_HISTORY_KEY = "kinetix.predictions.v1";
const MAX_SESSION_HISTORY = 100;

function getErrorMessage(error, fallback) {
  if (error?.response?.data?.detail) {
    return String(error.response.data.detail);
  }
  if (error?.code === "ECONNABORTED") {
    return "The request timed out. Check the API connection and try again.";
  }
  if (error?.message) {
    return error.message;
  }
  return fallback;
}

function readSessionHistory() {
  try {
    const stored = window.sessionStorage.getItem(SESSION_HISTORY_KEY);
    if (!stored) {
      return { entries: [], error: "" };
    }

    const parsed = JSON.parse(stored);
    if (!Array.isArray(parsed)) {
      return {
        entries: [],
        error: "Saved session history was invalid and could not be displayed.",
      };
    }
    const entries = parsed.filter(
      (entry) =>
        entry &&
        typeof entry.id === "string" &&
        typeof entry.activity === "string" &&
        typeof entry.source === "string" &&
        typeof entry.createdAt === "string" &&
        typeof entry.confidence === "number" &&
        Number.isFinite(entry.confidence) &&
        entry.confidence >= 0 &&
        entry.confidence <= 1,
    );
    return {
      entries: entries.slice(0, MAX_SESSION_HISTORY),
      error:
        entries.length === parsed.length
          ? ""
          : "Some saved session history entries were invalid and were skipped.",
    };
  } catch {
    return {
      entries: [],
      error: "Session storage is unavailable in this browser.",
    };
  }
}

function formatProbabilities(probabilityMap) {
  if (!probabilityMap || typeof probabilityMap !== "object") {
    throw new Error("The API returned no class probabilities.");
  }

  return Object.entries(probabilityMap).map(([name, value]) => {
    const numericValue = Number(value);
    if (
      !Number.isFinite(numericValue) ||
      numericValue < 0 ||
      numericValue > 1
    ) {
      throw new Error("The API returned an invalid class probability.");
    }
    return { name, value: numericValue * 100 };
  });
}

function validatePredictionResponse(data) {
  if (
    !data ||
    typeof data.activity !== "string" ||
    !data.activity ||
    typeof data.confidence !== "number" ||
    !Number.isFinite(data.confidence) ||
    data.confidence < 0 ||
    data.confidence > 1
  ) {
    throw new Error("The API returned an invalid prediction response.");
  }
  return {
    activity: data.activity,
    confidence: data.confidence,
    probabilities: formatProbabilities(data.probabilities),
  };
}

function App() {
  const [route, setRoute] = useState(() =>
    resolveRoute(window.location.pathname, window.location.hash),
  );
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [inputMode, setInputMode] = useState("demo");

  const [demo, setDemo] = useState({
    status: "idle",
    activity: "",
    confidence: 0,
    groundTruth: "",
    probabilities: [],
    sensorData: [],
    featureNames: [],
    error: "",
  });
  const live = useLiveSensorCapture();
  const [liveResult, setLiveResult] = useState({
    activity: "",
    confidence: 0,
    probabilities: [],
  });

  const [health, setHealth] = useState({
    status: "checking",
    details: null,
    error: "",
    durationMs: null,
    lastChecked: null,
  });
  const [metrics, setMetrics] = useState(null);
  const [metricsError, setMetricsError] = useState("");
  const [trainingHistory, setTrainingHistory] = useState(null);
  const [trainingHistoryError, setTrainingHistoryError] = useState("");

  const [initialHistory] = useState(readSessionHistory);
  const [history, setHistory] = useState(initialHistory.entries);
  const [historyStorageError, setHistoryStorageError] = useState(
    initialHistory.error,
  );

  const [sensorSample, setSensorSample] = useState({
    status: "idle",
    data: null,
    error: "",
  });

  const navigate = useCallback((path) => {
    if (!APP_ROUTES.some((item) => item.path === path)) {
      return;
    }
    if (window.location.pathname !== path) {
      window.history.pushState(null, "", path);
    }
    setRoute(path);
    setSidebarOpen(false);
    if (live.status === "collecting") {
      live.cleanupCapture();
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [live]);

  useEffect(() => {
    const syncRoute = () => {
      const resolved = resolveRoute(
        window.location.pathname,
        window.location.hash,
      );
      if (window.location.pathname !== resolved || window.location.hash) {
        window.history.replaceState(null, "", resolved);
      }
      setRoute(resolved);
      setSidebarOpen(false);
    };

    if (window.location.pathname !== route || window.location.hash) {
      window.history.replaceState(null, "", route);
    }
    window.addEventListener("popstate", syncRoute);
    window.addEventListener("hashchange", syncRoute);
    return () => {
      window.removeEventListener("popstate", syncRoute);
      window.removeEventListener("hashchange", syncRoute);
    };
  }, [route]);

  const refreshHealth = useCallback(async () => {
    setHealth((current) => ({ ...current, status: "checking", error: "" }));
    const startedAt = performance.now();
    try {
      const response = await checkHealth();
      if (response.data?.status !== "ok") {
        throw new Error("The health endpoint returned an unexpected response.");
      }
      setHealth({
        status: "online",
        details: response.data,
        error: "",
        durationMs: Math.round(performance.now() - startedAt),
        lastChecked: new Date().toISOString(),
      });
    } catch (error) {
      setHealth({
        status: "offline",
        details: null,
        error: getErrorMessage(error, "Unable to reach the FastAPI service."),
        durationMs: null,
        lastChecked: new Date().toISOString(),
      });
    }
  }, []);

  useEffect(() => {
    refreshHealth();

    getMetrics()
      .then((response) => setMetrics(response.data))
      .catch((error) =>
        setMetricsError(
          getErrorMessage(error, "Evaluation metrics could not be loaded."),
        ),
      );

    getTrainingHistory()
      .then((response) => setTrainingHistory(response.data))
      .catch((error) =>
        setTrainingHistoryError(
          getErrorMessage(error, "Training history could not be loaded."),
        ),
      );
  }, [refreshHealth]);

  const recordPrediction = (prediction, source, groundTruth = "") => {
    const entry = {
      id: globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`,
      activity: prediction.activity,
      confidence: prediction.confidence,
      probabilities: prediction.probabilities,
      source,
      groundTruth,
      createdAt: new Date().toISOString(),
    };
    const updated = [entry, ...history].slice(0, MAX_SESSION_HISTORY);
    setHistory(updated);
    try {
      window.sessionStorage.setItem(SESSION_HISTORY_KEY, JSON.stringify(updated));
      setHistoryStorageError("");
    } catch {
      setHistoryStorageError(
        "Session storage is unavailable; this history will not survive a page reload.",
      );
    }
  };

  const runDemoPrediction = async () => {
    setDemo((current) => ({
      ...current,
      status: "loading",
      activity: "",
      confidence: 0,
      groundTruth: "",
      probabilities: [],
      sensorData: [],
      featureNames: [],
      error: "",
    }));

    try {
      const sampleResponse = await getSampleSensorData();
      const sample = sampleResponse.data;
      const validation = validateWindow(sample?.sensor_data);
      if (!validation.valid) {
        throw new Error(
          validation.error || "The sample endpoint returned an invalid window.",
        );
      }

      setDemo((current) => ({
        ...current,
        sensorData: sample.sensor_data,
        featureNames: sample.feature_names || [],
        groundTruth: sample.ground_truth || "",
      }));

      const response = await predictActivity(sample.sensor_data, {
        source: "dataset",
      });
      const result = validatePredictionResponse(response.data);
      setDemo((current) => ({
        ...current,
        ...result,
        groundTruth: sample.ground_truth || "",
        status: "complete",
        sensorData: sample.sensor_data,
        featureNames: sample.feature_names || [],
        error: "",
      }));
      recordPrediction(result, "dataset", sample.ground_truth || "");
    } catch (error) {
      setDemo((current) => ({
        ...current,
        status: "error",
        error: getErrorMessage(error, "Dataset prediction failed."),
      }));
    }
  };

  const runLivePrediction = async () => {
    setLiveResult({ activity: "", confidence: 0, probabilities: [] });
    live.setError("");

    const validation = live.getValidatedWindow();
    if (!validation.valid) {
      live.setStatus("error");
      live.setError(validation.error || "Invalid sensor window.");
      return;
    }

    if (health.status !== "online") {
      live.setStatus("error");
      live.setError("Backend unavailable. Check the API connection and retry.");
      return;
    }

    live.setStatus("analyzing");
    try {
      const response = await predictActivity(live.capturedWindow, {
        source: "live",
      });
      const result = validatePredictionResponse(response.data);
      setLiveResult(result);
      live.setStatus("complete");
      recordPrediction(result, "live");
    } catch (error) {
      live.setStatus("error");
      live.setError(getErrorMessage(error, "Live prediction failed."));
    }
  };

  const loadSensorSample = useCallback(async () => {
    setSensorSample({ status: "loading", data: null, error: "" });
    try {
      const response = await getSampleSensorData();
      const validation = validateWindow(response.data?.sensor_data);
      if (!validation.valid) {
        throw new Error(
          validation.error || "The sample endpoint returned an invalid window.",
        );
      }
      setSensorSample({
        status: "loaded",
        data: response.data,
        error: "",
      });
    } catch (error) {
      setSensorSample({
        status: "error",
        data: null,
        error: getErrorMessage(error, "Unable to load a UCI HAR sensor sample."),
      });
    }
  }, []);

  useEffect(() => {
    if (route === "/sensors" && sensorSample.status === "idle") {
      loadSensorSample();
    }
  }, [route, sensorSample.status, loadSensorSample]);

  const handleModeChange = (mode) => {
    setInputMode(mode);
    live.cleanupCapture();
  };

  const clearHistory = () => {
    setHistory([]);
    try {
      window.sessionStorage.removeItem(SESSION_HISTORY_KEY);
      setHistoryStorageError("");
    } catch {
      setHistoryStorageError(
        "Session storage could not be cleared; the displayed history is empty for now.",
      );
    }
  };

  const latestPrediction = history[0] || null;
  const apiStatusLabel = {
    checking: "Checking API",
    online: "API Connected",
    offline: "API Offline",
  }[health.status];

  return (
    <div className="app">
      <Header
        apiStatus={health.status}
        currentPath={route}
        isOpen={sidebarOpen}
        onNavigate={navigate}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="main-shell">
        <div className="topbar">
          <button
            type="button"
            className="menu-toggle"
            aria-label={
              sidebarOpen ? "Close navigation menu" : "Open navigation menu"
            }
            aria-expanded={sidebarOpen}
            onClick={() => setSidebarOpen((open) => !open)}
          >
            <Menu size={17} />
          </button>
          <div className="topbar-context" aria-label="Current page">
            <span>Kinetix</span>
            <span className="topbar-divider">/</span>
            <strong>{getRouteTitle(route)}</strong>
          </div>
          <div className="topbar-actions">
            <div className={`header-status ${health.status}`}>
              <span className="status-dot" />
              {apiStatusLabel}
            </div>
            <a
              className="api-docs-link"
              href={`${API_BASE_URL}/docs`}
              target="_blank"
              rel="noreferrer"
              aria-label="Open FastAPI documentation"
            >
              <BookOpen size={13} />
              API Docs
            </a>
          </div>
        </div>

        <main className="layout" key={route}>
          {route === "/overview" && (
            <OverviewPage
              metrics={metrics}
              apiStatus={health.status}
              prediction={demo}
              onPredict={runDemoPrediction}
              onNavigate={navigate}
            />
          )}
          {route === "/predict" && (
            <PredictPage
              inputMode={inputMode}
              onModeChange={handleModeChange}
              demo={demo}
              live={live}
              liveResult={liveResult}
              modelOnline={health.status === "online"}
              onDemoPredict={runDemoPrediction}
              onLivePredict={runLivePrediction}
            />
          )}
          {route === "/analytics" && (
            <>
              <AnalyticsPage
                metrics={metrics}
                trainingHistory={trainingHistory}
                latestPrediction={latestPrediction}
              />
              {metricsError && (
                <p className="inline-warning" role="status">
                  Evaluation metrics unavailable: {metricsError}
                </p>
              )}
              {trainingHistoryError && !trainingHistory && (
                <p className="inline-warning" role="status">
                  Training history unavailable: {trainingHistoryError}
                </p>
              )}
            </>
          )}
          {route === "/history" && (
            <HistoryPage
              entries={history}
              storageError={historyStorageError}
              onClear={clearHistory}
            />
          )}
          {route === "/sensors" && (
            <SensorsPage
              sample={sensorSample.data}
              status={sensorSample.status}
              error={sensorSample.error}
              onRetry={loadSensorSample}
            />
          )}
          {route === "/status" && (
            <StatusPage health={health} onRefresh={refreshHealth} />
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
