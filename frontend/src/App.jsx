import { useEffect, useState } from "react";
import EvaluationPanel from "./components/EvaluationPanel";
import Header from "./components/Header";
import InputModeSelector from "./components/InputModeSelector";
import LiveSensorPanel from "./components/LiveSensorPanel";
import ModelInfo from "./components/ModelInfo";
import PredictionPanel from "./components/PredictionPanel";
import ProbabilityChart from "./components/ProbabilityChart";
import SensorChart from "./components/SensorChart";
import TrainingPanel from "./components/TrainingPanel";
import { useLiveSensorCapture } from "./hooks/useLiveSensorCapture";
import {
  checkHealth,
  getMetrics,
  getSampleSensorData,
  getTrainingHistory,
  predictActivity,
} from "./services/api";
import { FEATURE_NAMES } from "./services/liveSensors";
import "./App.css";

function App() {
  const [inputMode, setInputMode] = useState("demo");

  const [demoStatus, setDemoStatus] = useState("idle");
  const [demoActivity, setDemoActivity] = useState("");
  const [demoConfidence, setDemoConfidence] = useState(0);
  const [demoGroundTruth, setDemoGroundTruth] = useState("");
  const [demoProbabilities, setDemoProbabilities] = useState([]);
  const [demoSensorData, setDemoSensorData] = useState([]);
  const [demoFeatureNames, setDemoFeatureNames] = useState([]);
  const [demoError, setDemoError] = useState("");

  const live = useLiveSensorCapture();
  const [liveActivity, setLiveActivity] = useState("");
  const [liveConfidence, setLiveConfidence] = useState(0);
  const [liveProbabilities, setLiveProbabilities] = useState([]);

  const [modelOnline, setModelOnline] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState(null);

  useEffect(() => {
    const init = async () => {
      try {
        await checkHealth();
        setModelOnline(true);
      } catch {
        setModelOnline(false);
      }

      try {
        const metricsResponse = await getMetrics();
        setMetrics(metricsResponse.data);
      } catch {
        setMetrics(null);
      }

      try {
        const historyResponse = await getTrainingHistory();
        setTrainingHistory(historyResponse.data);
      } catch {
        setTrainingHistory(null);
      }
    };

    init();
  }, []);

  const runDemoPrediction = async () => {
    setDemoStatus("loading");
    setDemoError("");
    setDemoActivity("");
    setDemoConfidence(0);
    setDemoGroundTruth("");
    setDemoProbabilities([]);
    setDemoSensorData([]);
    setDemoFeatureNames([]);

    try {
      const sampleResponse = await getSampleSensorData();
      const {
        sensor_data: window,
        ground_truth: truth,
        feature_names: names,
      } = sampleResponse.data;

      if (!Array.isArray(window) || window.length !== 128) {
        throw new Error("Invalid sample from API.");
      }

      setDemoSensorData(window);
      setDemoGroundTruth(truth || "");
      setDemoFeatureNames(names || []);

      const response = await predictActivity(window, { source: "dataset" });
      const {
        activity: predictedActivity,
        confidence: predictedConfidence,
        probabilities: probabilityMap,
      } = response.data;

      setDemoActivity(predictedActivity);
      setDemoConfidence(predictedConfidence);
      setDemoProbabilities(
        Object.entries(probabilityMap || {}).map(([name, value]) => ({
          name,
          value: Number(value) * 100,
        })),
      );
      setDemoStatus("complete");
    } catch (err) {
      console.error(err);
      setDemoStatus("error");
      setDemoError("Unable to connect to Kinetix API");
    }
  };

  const runLivePrediction = async () => {
    live.setError("");
    setLiveActivity("");
    setLiveConfidence(0);
    setLiveProbabilities([]);

    const validation = live.getValidatedWindow();
    if (!validation.valid) {
      live.setStatus("error");
      live.setError(validation.error || "Invalid sensor window.");
      return;
    }

    if (!modelOnline) {
      live.setStatus("error");
      live.setError("Backend unavailable. Check API connection.");
      return;
    }

    live.setStatus("analyzing");

    try {
      const response = await predictActivity(live.capturedWindow, {
        source: "live",
      });
      const {
        activity: predictedActivity,
        confidence: predictedConfidence,
        probabilities: probabilityMap,
      } = response.data;

      setLiveActivity(predictedActivity);
      setLiveConfidence(predictedConfidence);
      setLiveProbabilities(
        Object.entries(probabilityMap || {}).map(([name, value]) => ({
          name,
          value: Number(value) * 100,
        })),
      );
      live.setStatus("complete");
    } catch (err) {
      console.error(err);
      live.setStatus("error");
      live.setError(
        err?.response?.data?.detail ||
          err?.message ||
          "Prediction failed. Check API connection.",
      );
    }
  };

  const handleModeChange = (mode) => {
    setInputMode(mode);
    if (mode === "live") {
      live.cleanupCapture();
    }
  };

  const liveChartData =
    live.status === "complete" || live.status === "window_ready"
      ? live.capturedWindow
      : live.chartPreview;

  return (
    <div className="app">
      <Header modelOnline={modelOnline} metrics={metrics} />

      <main className="layout">
        <div className="section-bar section-bar-mode">
          <span className="section-label">Inference</span>
          <InputModeSelector mode={inputMode} onChange={handleModeChange} />
        </div>

        {inputMode === "demo" ? (
          <>
            <div className="pipeline-row">
              <div className="pipeline-item">
                <span>Source</span>
                <strong>UCI HAR test</strong>
              </div>
              <div className="pipeline-item">
                <span>Window</span>
                <strong>128 × 6</strong>
              </div>
              <div className="pipeline-item">
                <span>Model</span>
                <strong>BiLSTM 128 → LSTM 64</strong>
              </div>
              <div className="pipeline-item">
                <span>Output</span>
                <strong>6-class softmax</strong>
              </div>
            </div>

            <div className="prediction-row">
              <PredictionPanel
                status={demoStatus}
                activity={demoActivity}
                confidence={demoConfidence}
                groundTruth={demoGroundTruth}
                error={demoError}
                onPredict={runDemoPrediction}
              />
              <ProbabilityChart
                probabilities={demoProbabilities}
                status={demoStatus}
              />
            </div>

            <SensorChart
              sensorData={demoSensorData}
              status={demoStatus}
              featureNames={demoFeatureNames}
              mode="demo"
            />
          </>
        ) : (
          <>
            <div className="pipeline-row">
              <div className="pipeline-item">
                <span>Source</span>
                <strong>DeviceMotion</strong>
              </div>
              <div className="pipeline-item">
                <span>Window</span>
                <strong>128 × 6 @ ~50 Hz</strong>
              </div>
              <div className="pipeline-item">
                <span>Preprocess</span>
                <strong>Experimental</strong>
              </div>
              <div className="pipeline-item">
                <span>API</span>
                <strong>POST /predict</strong>
              </div>
            </div>

            <div className="prediction-row">
              <LiveSensorPanel
                live={live}
                activity={liveActivity}
                confidence={liveConfidence}
                onAnalyze={runLivePrediction}
                modelOnline={modelOnline}
              />
              <ProbabilityChart
                probabilities={liveProbabilities}
                status={live.status}
                tag="Live"
                loadingMessage="Analyzing live sensor window…"
                emptyMessage="Capture 128 samples, then run prediction."
              />
            </div>

            <SensorChart
              sensorData={liveChartData}
              status={live.status}
              featureNames={FEATURE_NAMES}
              mode="live"
              title={
                live.status === "complete" || live.status === "window_ready"
                  ? "Captured Window"
                  : "Live Sensor Trace"
              }
            />
          </>
        )}

        <div className="section-bar">
          <span className="section-label">Model &amp; Evaluation</span>
        </div>

        <div className="bottom-row">
          <ModelInfo metrics={metrics} />
          <EvaluationPanel metrics={metrics} />
        </div>

        <div className="section-bar">
          <span className="section-label">Training</span>
        </div>

        <TrainingPanel history={trainingHistory} metrics={metrics} />
      </main>
    </div>
  );
}

export default App;
