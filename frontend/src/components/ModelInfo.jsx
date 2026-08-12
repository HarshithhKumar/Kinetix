export default function ModelInfo({ metrics }) {
  const training = metrics?.training;
  const architecture = training?.architecture || "BiLSTM(128) → LSTM(64)";
  const accuracy = metrics?.accuracy_percentage;
  const macroF1 = metrics?.macro_f1;

  const rows = [
    { label: "Architecture", value: architecture },
    { label: "Dataset", value: "UCI HAR Dataset" },
    { label: "Input", value: "128 × 6" },
    { label: "Features", value: "body acc + gyro" },
    { label: "Classes", value: "6 activities" },
    { label: "Parameters", value: training?.parameters?.toLocaleString() ?? "—" },
    { label: "Preprocessing", value: "StandardScaler (train-fit)" },
    {
      label: "Test Accuracy",
      value: accuracy != null ? `${accuracy.toFixed(2)}%` : "—",
    },
    {
      label: "Macro F1",
      value: macroF1 != null ? `${(macroF1 * 100).toFixed(2)}%` : "—",
    },
  ];

  return (
    <section className="panel info-panel">
      <div className="panel-head">
        <h2>Model</h2>
        <p className="panel-sub">BiLSTM · TensorFlow / Keras</p>
      </div>

      <dl className="info-grid">
        {rows.map(({ label, value }) => (
          <div key={label} className="info-row">
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
