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

const COLORS = {
  WALKING: "#4f6ef7",
  WALKING_UPSTAIRS: "#6b84f9",
  WALKING_DOWNSTAIRS: "#3d57d8",
  SITTING: "#5ea87a",
  STANDING: "#c49a3c",
  LAYING: "#c45c7a",
};

export default function ProbabilityChart({
  probabilities,
  status,
  title = "Probability Distribution",
  tag,
  loadingMessage = "Computing class probabilities…",
  emptyMessage = "Run a prediction to view softmax outputs.",
}) {
  const sorted = [...probabilities].sort((a, b) => b.value - a.value);
  const isLoading = status === "loading" || status === "analyzing";

  return (
    <section className="panel chart-panel">
      <div className="panel-head">
        <div className="panel-head-inline">
          <h2>{title}</h2>
          {tag && <span className="panel-tag">{tag}</span>}
        </div>
      </div>

      {isLoading && (
        <div className="empty-state">{loadingMessage}</div>
      )}

      {!isLoading && !sorted.length && (
        <div className="empty-state">{emptyMessage}</div>
      )}

      {sorted.length > 0 && (
        <div className="chart-area">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart
              data={sorted}
              layout="vertical"
              margin={{ top: 8, right: 16, left: 8, bottom: 8 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                horizontal
                vertical={false}
                stroke="#2a3144"
              />
              <XAxis
                type="number"
                domain={[0, 100]}
                tick={{ fill: "#8b93a7", fontSize: 11 }}
                tickFormatter={(v) => `${v}%`}
              />
              <YAxis
                type="category"
                dataKey="name"
                width={130}
                tick={{ fill: "#c5cad6", fontSize: 11 }}
              />
              <Tooltip
                formatter={(value) => [`${Number(value).toFixed(2)}%`, "Prob"]}
                contentStyle={{
                  background: "#151a24",
                  border: "1px solid #2a3144",
                  borderRadius: "4px",
                  fontSize: "12px",
                }}
              />
              <Bar dataKey="value" radius={[0, 3, 3, 0]} barSize={18}>
                {sorted.map((entry) => (
                  <Cell
                    key={entry.name}
                    fill={COLORS[entry.name] || "#4f6ef7"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}
