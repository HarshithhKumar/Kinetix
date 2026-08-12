import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

function toEpochRows(history) {
  if (!history?.accuracy?.length) return [];

  return history.accuracy.map((trainAcc, index) => ({
    epoch: index + 1,
    trainAcc: trainAcc * 100,
    valAcc: (history.val_accuracy?.[index] ?? 0) * 100,
    trainLoss: history.loss?.[index] ?? 0,
    valLoss: history.val_loss?.[index] ?? 0,
  }));
}

export default function TrainingPanel({ history, metrics }) {
  const rows = useMemo(() => toEpochRows(history), [history]);

  if (!rows.length) {
    return (
      <section className="panel training-panel">
        <div className="panel-head">
          <h2>Training History</h2>
        </div>
        <div className="empty-state">
          Training history not loaded. Expected at{" "}
          <code>/assets/training_history.json</code>.
        </div>
      </section>
    );
  }

  const bestEpoch = metrics?.training?.best_epoch;
  const epochsTrained = metrics?.training?.epochs_trained ?? rows.length;

  return (
    <section className="panel training-panel">
      <div className="panel-head">
        <div>
          <h2>Training History</h2>
          <p className="panel-sub">
            {epochsTrained} epochs
            {bestEpoch != null ? ` · best validation epoch ${bestEpoch}` : ""}
          </p>
        </div>
      </div>

      <div className="training-charts">
        <div className="training-chart-block">
          <p className="chart-label">Accuracy</p>
          <div className="chart-area chart-area-compact">
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={rows} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3144" />
                <XAxis
                  dataKey="epoch"
                  tick={{ fill: "#8b93a7", fontSize: 10 }}
                  label={{
                    value: "Epoch",
                    position: "insideBottom",
                    offset: -2,
                    fill: "#6b7280",
                    fontSize: 11,
                  }}
                />
                <YAxis
                  domain={[0, 100]}
                  tick={{ fill: "#8b93a7", fontSize: 10 }}
                  tickFormatter={(v) => `${v}%`}
                  width={42}
                />
                <Tooltip
                  formatter={(value, name) => [
                    `${Number(value).toFixed(2)}${name.includes("Loss") ? "" : "%"}`,
                    name,
                  ]}
                  contentStyle={{
                    background: "#151a24",
                    border: "1px solid #2a3144",
                    borderRadius: "4px",
                    fontSize: "12px",
                  }}
                />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <Line
                  type="monotone"
                  dataKey="trainAcc"
                  name="Train accuracy"
                  stroke="#4f6ef7"
                  strokeWidth={1.5}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="valAcc"
                  name="Val accuracy"
                  stroke="#5ea87a"
                  strokeWidth={1.5}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="training-chart-block">
          <p className="chart-label">Loss</p>
          <div className="chart-area chart-area-compact">
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={rows} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3144" />
                <XAxis
                  dataKey="epoch"
                  tick={{ fill: "#8b93a7", fontSize: 10 }}
                  label={{
                    value: "Epoch",
                    position: "insideBottom",
                    offset: -2,
                    fill: "#6b7280",
                    fontSize: 11,
                  }}
                />
                <YAxis tick={{ fill: "#8b93a7", fontSize: 10 }} width={42} />
                <Tooltip
                  formatter={(value, name) => [Number(value).toFixed(4), name]}
                  contentStyle={{
                    background: "#151a24",
                    border: "1px solid #2a3144",
                    borderRadius: "4px",
                    fontSize: "12px",
                  }}
                />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <Line
                  type="monotone"
                  dataKey="trainLoss"
                  name="Train loss"
                  stroke="#c49a3c"
                  strokeWidth={1.5}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="valLoss"
                  name="Val loss"
                  stroke="#c45c7a"
                  strokeWidth={1.5}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </section>
  );
}
