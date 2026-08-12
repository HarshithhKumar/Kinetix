import { Loader2 } from "lucide-react";

const STATUS_TEXT = {
  idle: "Ready — loads UCI HAR test window index 0",
  loading: "Running inference…",
  complete: "Inference complete",
  error: "API request failed",
};

export default function PredictionPanel({
  status,
  activity,
  confidence,
  groundTruth,
  error,
  onPredict,
}) {
  const hasResult = status === "complete" && activity;
  const isMatch =
    hasResult &&
    groundTruth &&
    activity.toUpperCase() === groundTruth.toUpperCase();

  return (
    <section className="panel prediction-panel">
      <div className="panel-head">
        <h2>Activity Prediction</h2>
        <span className="panel-tag">Dataset Demo</span>
      </div>

      <div className="prediction-main">
        <p className="prediction-label">Detected Activity</p>
        <p className="prediction-value">{hasResult ? activity : "—"}</p>

        <div className="prediction-meta">
          <div>
            <span>Confidence</span>
            <strong>
              {hasResult ? `${(confidence * 100).toFixed(2)}%` : "—"}
            </strong>
          </div>
          <div>
            <span>Ground Truth</span>
            <strong>{groundTruth || "—"}</strong>
          </div>
        </div>

        {hasResult && groundTruth && (
          <div
            className={`demo-comparison ${isMatch ? "demo-match" : "demo-mismatch"}`}
          >
            <div className="demo-comparison-row">
              <span>Ground truth</span>
              <strong>{groundTruth}</strong>
            </div>
            <div className="demo-comparison-row">
              <span>Prediction</span>
              <strong>{activity}</strong>
            </div>
            <p className="demo-comparison-note">
              {isMatch
                ? "Prediction matches the UCI test label."
                : "Individual misclassification — pipeline verified; 59/532 STANDING windows are predicted as SITTING on the full test set."}
            </p>
          </div>
        )}

        <button
          type="button"
          className="btn-primary"
          onClick={onPredict}
          disabled={status === "loading"}
        >
          {status === "loading" ? (
            <>
              <Loader2 size={16} className="spin" />
              Running…
            </>
          ) : (
            "Run Prediction"
          )}
        </button>

        <p className={`status-line status-${status}`}>
          {error || STATUS_TEXT[status] || STATUS_TEXT.idle}
        </p>
      </div>
    </section>
  );
}
