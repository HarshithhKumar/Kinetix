import { Activity } from "lucide-react";

export default function Header({ modelOnline, metrics }) {
  return (
    <header className="header">
      <div className="header-brand">
        <Activity size={20} strokeWidth={2.2} />
        <div>
          <h1>Kinetix</h1>
          <p>Human Activity Recognition</p>
        </div>
      </div>

      <div className="header-right">
        {metrics && (
          <div className="header-metrics">
            <span>Accuracy {metrics.accuracy_percentage.toFixed(2)}%</span>
            <span>P {(metrics.macro_precision * 100).toFixed(1)}%</span>
            <span>R {(metrics.macro_recall * 100).toFixed(1)}%</span>
            <span>F1 {(metrics.macro_f1 * 100).toFixed(1)}%</span>
          </div>
        )}

        <div className={`header-status ${modelOnline ? "online" : "offline"}`}>
          <span className="status-dot" />
          {modelOnline ? "Model Online" : "API Offline"}
        </div>
      </div>
    </header>
  );
}
