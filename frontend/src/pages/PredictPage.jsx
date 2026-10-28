import InputModeSelector from "../components/InputModeSelector";
import LiveSensorPanel from "../components/LiveSensorPanel";
import PageHeading from "../components/PageHeading";
import PredictionPanel from "../components/PredictionPanel";
import ProbabilityChart from "../components/ProbabilityChart";
import SensorChart from "../components/SensorChart";
import { FEATURE_NAMES } from "../services/liveSensors";

export default function PredictPage({
  inputMode,
  onModeChange,
  demo,
  live,
  liveResult,
  modelOnline,
  onDemoPredict,
  onLivePredict,
}) {
  const liveChartData =
    live.status === "complete" || live.status === "window_ready"
      ? live.capturedWindow
      : live.chartPreview;

  return (
    <>
      <PageHeading
        title="Live Prediction"
        description="Run inference through the existing Kinetix API using an official UCI HAR test window or an experimental phone sensor capture."
      />

      <div className="section-bar section-bar-mode">
        <div className="section-heading">
          <h2>Input method</h2>
          <p>Predictions are returned by the configured FastAPI model.</p>
        </div>
        <InputModeSelector mode={inputMode} onChange={onModeChange} />
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
              <span>Endpoint</span>
              <strong>GET /sample → POST /predict</strong>
            </div>
          </div>
          <div className="prediction-row">
            <PredictionPanel
              status={demo.status}
              activity={demo.activity}
              confidence={demo.confidence}
              groundTruth={demo.groundTruth}
              error={demo.error}
              onPredict={onDemoPredict}
            />
            <ProbabilityChart
              probabilities={demo.probabilities}
              status={demo.status}
            />
          </div>
          <SensorChart
            sensorData={demo.sensorData}
            status={demo.status}
            featureNames={demo.featureNames}
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
              <strong>Experimental approximation</strong>
            </div>
            <div className="pipeline-item">
              <span>Endpoint</span>
              <strong>POST /predict</strong>
            </div>
          </div>
          <div className="prediction-row">
            <LiveSensorPanel
              live={live}
              activity={liveResult.activity}
              confidence={liveResult.confidence}
              onAnalyze={onLivePredict}
              modelOnline={modelOnline}
            />
            <ProbabilityChart
              probabilities={liveResult.probabilities}
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
    </>
  );
}
