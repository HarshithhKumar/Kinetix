import { Loader2 } from "lucide-react";
import { WINDOW_SIZE } from "../services/liveSensors";

const STATUS_LABELS = {
  not_supported: "Not supported",
  permission_required: "Permission required",
  ready: "Ready",
  collecting: "Collecting",
  window_ready: "Window ready",
  analyzing: "Analyzing",
  complete: "Prediction complete",
  error: "Error",
};

function StatusIndicator({ status }) {
  const active = !["not_supported", "error", "permission_required"].includes(
    status,
  );

  return (
    <div className={`live-status-indicator ${status}`}>
      <span className={`status-dot ${active ? "pulse" : ""}`} />
      <strong>{STATUS_LABELS[status] || status}</strong>
    </div>
  );
}

export default function LiveSensorPanel({
  live,
  activity,
  confidence,
  onAnalyze,
  modelOnline,
}) {
  const {
    support,
    status,
    permissionGranted,
    accelAvailable,
    gyroAvailable,
    sampleCount,
    error,
    enableMotion,
    startCapture,
    resetCapture,
    capturedWindow,
  } = live;

  const canStart =
    permissionGranted &&
    ["ready", "window_ready", "complete", "error"].includes(status);
  const canAnalyze =
    status === "window_ready" && capturedWindow.length === WINDOW_SIZE;
  const isCollecting = status === "collecting";
  const isAnalyzing = status === "analyzing";
  const hasResult = status === "complete" && activity;

  if (!support.supported) {
    return (
      <section className="panel live-panel">
        <div className="panel-head">
          <h2>Live Sensor</h2>
          <span className="panel-tag panel-tag-experimental">Experimental</span>
        </div>
        <div className="live-unsupported">
          <p>Live sensors are not supported on this device/browser.</p>
          <p className="live-note">
            DeviceMotionEvent is required. Use a mobile browser (Safari iOS,
            Chrome Android) over HTTPS or localhost.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section className="panel live-panel">
      <div className="panel-head live-panel-head">
        <div>
          <h2>Live Sensor</h2>
          <p className="panel-sub">
            Experimental · browser motion → UCI-like 128×6 window
          </p>
        </div>
        <span className="panel-tag panel-tag-experimental">Experimental</span>
      </div>

      <div className="live-status-block">
        <div className="live-status-row">
          <span className="live-label">LIVE SENSOR</span>
          <StatusIndicator status={status} />
        </div>

        <div className="live-capability-grid">
          <div className="live-capability">
            <span>Accelerometer</span>
            <strong className={accelAvailable ? "ok" : "pending"}>
              {accelAvailable ? "✓" : "—"}
            </strong>
          </div>
          <div className="live-capability">
            <span>Gyroscope</span>
            <strong className={gyroAvailable ? "ok" : "pending"}>
              {gyroAvailable ? "✓" : "—"}
            </strong>
          </div>
          <div className="live-capability">
            <span>Samples</span>
            <strong>
              {sampleCount} / {WINDOW_SIZE}
            </strong>
          </div>
          <div className="live-capability">
            <span>Rate</span>
            <strong>~50 Hz</strong>
          </div>
        </div>
      </div>

      {!permissionGranted && status === "permission_required" && (
        <button type="button" className="btn-primary" onClick={enableMotion}>
          Enable Motion Sensors
        </button>
      )}

      {permissionGranted && (
        <div className="live-actions">
          <button
            type="button"
            className="btn-secondary"
            onClick={startCapture}
            disabled={!canStart || isCollecting || isAnalyzing}
          >
            {isCollecting ? "Capturing…" : "Start Capture"}
          </button>

          {canAnalyze && (
            <button
              type="button"
              className="btn-primary"
              onClick={onAnalyze}
              disabled={!modelOnline || isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 size={16} className="spin" />
                  Analyzing…
                </>
              ) : (
                "Run Prediction"
              )}
            </button>
          )}

          {(sampleCount > 0 || status === "complete") && (
            <button type="button" className="btn-ghost" onClick={resetCapture}>
              Reset
            </button>
          )}
        </div>
      )}

      {status === "window_ready" && (
        <p className="live-ready-note">128 / 128 samples — window ready</p>
      )}

      {hasResult && (
        <div className="live-result-block">
          <p className="live-result-label">LIVE SENSOR PREDICTION</p>
          <p className="prediction-value live-prediction-value">{activity}</p>
          <div className="prediction-meta">
            <div>
              <span>Confidence</span>
              <strong>{(confidence * 100).toFixed(2)}%</strong>
            </div>
          </div>
        </div>
      )}

      {error && <p className="status-line status-error">{error}</p>}

      <p className="live-disclaimer">
        Browser preprocessing approximates UCI body acceleration and gyroscope.
        Live predictions are not validated against labeled data.
      </p>
    </section>
  );
}
