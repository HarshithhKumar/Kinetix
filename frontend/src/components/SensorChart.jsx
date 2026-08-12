import { useMemo, useState } from "react";
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

const GROUPS = {
  accelerometer: {
    label: "Accelerometer",
    series: [
      { key: "accX", name: "acc_x", color: "#4f6ef7" },
      { key: "accY", name: "acc_y", color: "#5ea87a" },
      { key: "accZ", name: "acc_z", color: "#c49a3c" },
    ],
  },
  gyroscope: {
    label: "Gyroscope",
    series: [
      { key: "gyroX", name: "gyro_x", color: "#6b84f9" },
      { key: "gyroY", name: "gyro_y", color: "#c45c7a" },
      { key: "gyroZ", name: "gyro_z", color: "#4db6ac" },
    ],
  },
};

function toChartRows(sensorData) {
  if (!sensorData?.length) return [];

  return sensorData.map((row, index) => ({
    t: index + 1,
    accX: row[0],
    accY: row[1],
    accZ: row[2],
    gyroX: row[3],
    gyroY: row[4],
    gyroZ: row[5],
  }));
}

export default function SensorChart({
  sensorData,
  status,
  featureNames,
  mode = "demo",
  title = "Sensor Window",
  subtitle,
  emptyMessage,
}) {
  const [activeGroup, setActiveGroup] = useState("accelerometer");
  const chartData = useMemo(() => toChartRows(sensorData), [sensorData]);
  const group = GROUPS[activeGroup];

  const defaultSubtitle =
    mode === "live"
      ? "Live capture · UCI-like body acc (g) + gyro (rad/s)"
      : "128 timesteps · raw UCI HAR body acceleration & gyroscope";

  const defaultEmpty =
    mode === "live"
      ? "Start capture to view live sensor trace."
      : "Sensor trace appears after running a prediction.";

  const isLoading =
    status === "loading" || (mode === "live" && status === "collecting");

  return (
    <section className="panel sensor-panel">
      <div className="panel-head sensor-head">
        <div>
          <h2>{title}</h2>
          <p className="panel-sub">{subtitle || defaultSubtitle}</p>
        </div>

        <div className="tab-group">
          {Object.entries(GROUPS).map(([key, { label }]) => (
            <button
              key={key}
              type="button"
              className={`tab ${activeGroup === key ? "active" : ""}`}
              onClick={() => setActiveGroup(key)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {featureNames?.length === 6 && (
        <p className="feature-row">{featureNames.join(" · ")}</p>
      )}

      {isLoading && !chartData.length && (
        <div className="empty-state">
          {mode === "live" ? "Collecting sensor samples…" : "Loading sample window…"}
        </div>
      )}

      {!isLoading && !chartData.length && (
        <div className="empty-state">{emptyMessage || defaultEmpty}</div>
      )}

      {chartData.length > 0 && (
        <>
          {mode === "live" && status === "collecting" && (
            <p className="live-chart-progress">
              Live preview · {chartData.length} samples
            </p>
          )}
          <div className="chart-area chart-area-tall">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart
                data={chartData}
                margin={{ top: 8, right: 12, left: 0, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3144" />
                <XAxis
                  dataKey="t"
                  tick={{ fill: "#8b93a7", fontSize: 10 }}
                  label={{
                    value: "Timestep",
                    position: "insideBottom",
                    offset: -2,
                    fill: "#6b7280",
                    fontSize: 11,
                  }}
                />
                <YAxis tick={{ fill: "#8b93a7", fontSize: 10 }} width={48} />
                <Tooltip
                  contentStyle={{
                    background: "#151a24",
                    border: "1px solid #2a3144",
                    borderRadius: "4px",
                    fontSize: "12px",
                  }}
                />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                {group.series.map(({ key, name, color }) => (
                  <Line
                    key={key}
                    type="monotone"
                    dataKey={key}
                    name={name}
                    stroke={color}
                    strokeWidth={1.5}
                    dot={false}
                    activeDot={{ r: 3 }}
                    isAnimationActive={mode !== "live" || status !== "collecting"}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </section>
  );
}
