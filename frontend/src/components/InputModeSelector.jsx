export default function InputModeSelector({ mode, onChange }) {
  return (
    <div className="input-mode-selector" role="tablist" aria-label="Input mode">
      <button
        type="button"
        role="tab"
        aria-selected={mode === "demo"}
        className={`mode-tab ${mode === "demo" ? "active" : ""}`}
        onClick={() => onChange("demo")}
      >
        Dataset Demo
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={mode === "live"}
        className={`mode-tab ${mode === "live" ? "active" : ""}`}
        onClick={() => onChange("live")}
      >
        Live Phone
      </button>
    </div>
  );
}
