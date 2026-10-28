import { Database, RotateCw } from "lucide-react";
import PageHeading from "../components/PageHeading";
import SensorChart from "../components/SensorChart";

export default function SensorsPage({ sample, status, error, onRetry }) {
  const canDisplay = Array.isArray(sample?.sensor_data);

  return (
    <>
      <PageHeading
        title="Sensor Explorer"
        description="Inspect the raw accelerometer and gyroscope window returned by the existing UCI HAR sample endpoint."
        actions={
          <button
            type="button"
            className="btn-secondary"
            onClick={onRetry}
            disabled={status === "loading"}
          >
            <RotateCw size={13} className={status === "loading" ? "spin" : ""} />
            {status === "loading" ? "Loading sample" : "Load sample"}
          </button>
        }
      />

      <section className="pipeline-row sensor-facts">
        <div className="pipeline-item">
          <span>Data source</span>
          <strong>GET /sample</strong>
        </div>
        <div className="pipeline-item">
          <span>Window dimensions</span>
          <strong>{sample?.shape?.join(" × ") || "128 × 6 expected"}</strong>
        </div>
        <div className="pipeline-item">
          <span>Ground truth</span>
          <strong>{sample?.ground_truth || "Available with sample"}</strong>
        </div>
        <div className="pipeline-item">
          <span>Feature count</span>
          <strong>{sample?.feature_names?.length || 6}</strong>
        </div>
      </section>

      {status === "loading" && (
        <div className="panel empty-state">
          Loading the real UCI test window from the FastAPI sample endpoint…
        </div>
      )}
      {status === "error" && (
        <div className="panel error-state" role="alert">
          <Database size={18} />
          <div>
            <strong>Sample data unavailable</strong>
            <p>{error}</p>
            <p>
              The UCI HAR dataset must be present for <code>GET /sample</code>.
              The endpoint does not accept uploads.
            </p>
          </div>
        </div>
      )}
      {status === "idle" && (
        <div className="panel empty-state">
          Load a sample to inspect a real sensor window.
        </div>
      )}
      {canDisplay && status === "loaded" && (
        <SensorChart
          sensorData={sample.sensor_data}
          status="complete"
          featureNames={sample.feature_names}
          mode="demo"
          title="UCI HAR test window"
          subtitle="Raw inertial sample · 128 timesteps · acceleration in g and gyroscope in rad/s"
        />
      )}
    </>
  );
}
