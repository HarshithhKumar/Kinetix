import { Activity, BarChart3, Database, Gauge } from "lucide-react";
import MetricCard from "../components/MetricCard";
import PageHeading from "../components/PageHeading";
import PredictionPanel from "../components/PredictionPanel";
import ProbabilityChart from "../components/ProbabilityChart";

export default function OverviewPage({
  metrics,
  apiStatus,
  prediction,
  onPredict,
  onNavigate,
}) {
  const classCount = metrics?.per_class
    ? Object.keys(metrics.per_class).length
    : null;

  return (
    <>
      <PageHeading
        title="Activity Classification"
        description="Classify real smartphone motion windows with the Kinetix human activity recognition model."
        actions={
          <button
            type="button"
            className="btn-secondary"
            onClick={() => onNavigate("/predict")}
          >
            Open prediction workspace
          </button>
        }
      />

      <section className="overview-stats" aria-label="Model overview">
        <MetricCard
          label="Test accuracy"
          value={
            metrics?.accuracy_percentage != null
              ? `${metrics.accuracy_percentage.toFixed(2)}%`
              : "Unavailable"
          }
          note="UCI HAR held-out test set"
          icon={Gauge}
          tone="blue"
        />
        <MetricCard
          label="Macro F1"
          value={
            metrics?.macro_f1 != null
              ? `${(metrics.macro_f1 * 100).toFixed(2)}%`
              : "Unavailable"
          }
          note="Saved offline evaluation"
          icon={BarChart3}
          tone="purple"
        />
        <MetricCard
          label="Test windows"
          value={
            metrics?.test_samples != null
              ? metrics.test_samples.toLocaleString()
              : "Unavailable"
          }
          note={classCount ? `${classCount} activity classes` : "UCI HAR evaluation"}
          icon={Database}
          tone="green"
        />
        <MetricCard
          label="Backend status"
          value={
            apiStatus === "online"
              ? "Online"
              : apiStatus === "checking"
                ? "Checking"
                : "Offline"
          }
          note="FastAPI health endpoint"
          icon={Activity}
          tone="amber"
        />
      </section>

      <section className="dashboard-section">
        <div className="section-bar">
          <div className="section-heading">
            <h2>Quick classify</h2>
            <p>
              Loads a real UCI test window and submits it to the configured
              FastAPI model.
            </p>
          </div>
          <button
            type="button"
            className="btn-ghost"
            onClick={() => onNavigate("/analytics")}
          >
            View model analytics
          </button>
        </div>
        <div className="prediction-row">
          <PredictionPanel
            status={prediction.status}
            activity={prediction.activity}
            confidence={prediction.confidence}
            groundTruth={prediction.groundTruth}
            error={prediction.error}
            onPredict={onPredict}
          />
          <ProbabilityChart
            probabilities={prediction.probabilities}
            status={prediction.status}
          />
        </div>
        {prediction.sensorData.length > 0 && (
          <div className="inline-result-note">
            Sample sensor window loaded. Inspect it in{" "}
            <button type="button" onClick={() => onNavigate("/sensors")}>
              Sensor Explorer
            </button>
            .
          </div>
        )}
      </section>
    </>
  );
}
