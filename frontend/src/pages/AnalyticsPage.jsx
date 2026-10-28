import { BarChart3, Gauge, Target } from "lucide-react";
import EvaluationPanel from "../components/EvaluationPanel";
import MetricCard from "../components/MetricCard";
import PageHeading from "../components/PageHeading";
import TrainingPanel from "../components/TrainingPanel";

export default function AnalyticsPage({ metrics, trainingHistory, latestPrediction }) {
  return (
    <>
      <PageHeading
        title="Model Analytics"
        description="Offline UCI HAR test-set evaluation and saved training curves. These metrics are distinct from confidence on any individual live prediction."
      />
      <section className="overview-stats" aria-label="Offline evaluation summary">
        <MetricCard
          label="Accuracy"
          value={
            metrics?.accuracy_percentage != null
              ? `${metrics.accuracy_percentage.toFixed(2)}%`
              : "Unavailable"
          }
          note={
            metrics?.test_samples
              ? `${metrics.test_samples.toLocaleString()} UCI test windows`
              : "Offline evaluation"
          }
          icon={Gauge}
          tone="blue"
        />
        <MetricCard
          label="Macro precision"
          value={
            metrics?.macro_precision != null
              ? `${(metrics.macro_precision * 100).toFixed(2)}%`
              : "Unavailable"
          }
          note="Unweighted class average"
          icon={Target}
          tone="purple"
        />
        <MetricCard
          label="Macro recall"
          value={
            metrics?.macro_recall != null
              ? `${(metrics.macro_recall * 100).toFixed(2)}%`
              : "Unavailable"
          }
          note="Unweighted class average"
          icon={BarChart3}
          tone="green"
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
          tone="amber"
        />
      </section>

      {latestPrediction && (
        <section className="panel">
          <div className="panel-head">
            <div>
              <h2>Most recent live inference</h2>
              <p className="panel-sub">
                A single model response, not an evaluation statistic.
              </p>
            </div>
            <span className="panel-tag">Live result</span>
          </div>
          <div className="latest-result">
            <strong>{latestPrediction.activity}</strong>
            <span>
              Confidence: {(latestPrediction.confidence * 100).toFixed(2)}%
            </span>
            <small>
              {latestPrediction.source} ·{" "}
              {new Date(latestPrediction.createdAt).toLocaleString()}
            </small>
          </div>
        </section>
      )}

      <EvaluationPanel metrics={metrics} />
      <TrainingPanel history={trainingHistory} metrics={metrics} />
    </>
  );
}
