import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const F1_COLORS = {
  WALKING: "#4f6ef7",
  WALKING_UPSTAIRS: "#6b84f9",
  WALKING_DOWNSTAIRS: "#3d57d8",
  SITTING: "#5ea87a",
  STANDING: "#c49a3c",
  LAYING: "#c45c7a",
};

export default function EvaluationPanel({ metrics }) {
  if (!metrics) {
    return (
      <section className="panel eval-panel">
        <div className="panel-head">
          <h2>Evaluation</h2>
        </div>
        <div className="empty-state">
          Metrics not loaded from <code>/assets/metrics.json</code>.
        </div>
      </section>
    );
  }

  const summary = [
    { label: "Accuracy", value: `${metrics.accuracy_percentage.toFixed(2)}%` },
    {
      label: "Macro Precision",
      value: `${(metrics.macro_precision * 100).toFixed(2)}%`,
    },
    {
      label: "Macro Recall",
      value: `${(metrics.macro_recall * 100).toFixed(2)}%`,
    },
    { label: "Macro F1", value: `${(metrics.macro_f1 * 100).toFixed(2)}%` },
  ];

  const perClass = metrics.per_class ? Object.entries(metrics.per_class) : [];

  const f1ChartData = perClass.map(([name, stats]) => ({
    name: name.replace("WALKING_", "W_").replace("_", " "),
    fullName: name,
    f1: stats.f1_score * 100,
  }));

  return (
    <section className="panel eval-panel">
      <div className="panel-head">
        <h2>Evaluation</h2>
        <p className="panel-sub">
          Official UCI HAR test set · {metrics.test_samples} windows · raw inertial
          → StandardScaler → BiLSTM
        </p>
      </div>

      <div className="metric-summary">
        {summary.map(({ label, value }) => (
          <div key={label} className="metric-card">
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>

      {f1ChartData.length > 0 && (
        <div className="f1-chart-wrap">
          <p className="chart-label">Per-class F1</p>
          <div className="chart-area chart-area-compact">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart
                data={f1ChartData}
                margin={{ top: 8, right: 8, left: 0, bottom: 24 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  vertical={false}
                  stroke="#2a3144"
                />
                <XAxis
                  dataKey="name"
                  tick={{ fill: "#8b93a7", fontSize: 10 }}
                  interval={0}
                  angle={-20}
                  textAnchor="end"
                  height={48}
                />
                <YAxis
                  domain={[0, 100]}
                  tick={{ fill: "#8b93a7", fontSize: 10 }}
                  tickFormatter={(v) => `${v}%`}
                  width={42}
                />
                <Tooltip
                  formatter={(value, _name, props) => [
                    `${Number(value).toFixed(2)}%`,
                    props.payload.fullName,
                  ]}
                  contentStyle={{
                    background: "#151a24",
                    border: "1px solid #2a3144",
                    borderRadius: "4px",
                    fontSize: "12px",
                  }}
                />
                <Bar dataKey="f1" radius={[3, 3, 0, 0]} barSize={28}>
                  {f1ChartData.map((entry) => (
                    <Cell
                      key={entry.fullName}
                      fill={F1_COLORS[entry.fullName] || "#4f6ef7"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {perClass.length > 0 && (
        <div className="table-wrap">
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Activity</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>Support</th>
              </tr>
            </thead>
            <tbody>
              {perClass.map(([name, stats]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{(stats.precision * 100).toFixed(1)}%</td>
                  <td>{(stats.recall * 100).toFixed(1)}%</td>
                  <td>{(stats.f1_score * 100).toFixed(1)}%</td>
                  <td>{stats.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="confusion-wrap">
        <p className="confusion-label">Confusion Matrix</p>
        <img
          src="/assets/confusion_matrix.png"
          alt="Confusion matrix for Kinetix BiLSTM on UCI HAR test set"
          className="confusion-image"
        />
      </div>
    </section>
  );
}
