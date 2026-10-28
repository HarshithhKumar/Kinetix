import { Activity, CircleCheck, CircleX, RotateCw, Server } from "lucide-react";
import PageHeading from "../components/PageHeading";

function DiagnosticRow({ label, value, note, state = "neutral" }) {
  const Icon =
    state === "online" ? CircleCheck : state === "offline" ? CircleX : Activity;
  return (
    <div className="diagnostic-row">
      <div className={`diagnostic-icon ${state}`}>
        <Icon size={15} />
      </div>
      <div className="diagnostic-copy">
        <strong>{label}</strong>
        <span>{note}</span>
      </div>
      <span className={`diagnostic-value ${state}`}>{value}</span>
    </div>
  );
}

export default function StatusPage({ health, onRefresh }) {
  const state =
    health.status === "online"
      ? "online"
      : health.status === "offline"
        ? "offline"
        : "neutral";
  const apiValue = {
    checking: "Checking",
    online: "Connected",
    offline: "Unavailable",
  }[health.status];
  const modelValue =
    health.status === "online"
      ? "Loaded at startup"
      : health.status === "offline"
        ? "Unknown"
        : "Checking";

  return (
    <>
      <PageHeading
        title="System Status"
        description="Live diagnostics from the configured FastAPI service. Unavailable details are identified rather than inferred as healthy."
        actions={
          <button
            type="button"
            className="btn-secondary"
            onClick={onRefresh}
            disabled={health.status === "checking"}
          >
            <RotateCw
              size={13}
              className={health.status === "checking" ? "spin" : ""}
            />
            Refresh status
          </button>
        }
      />

      <section className="panel status-panel">
        <div className="panel-head">
          <div>
            <h2>Health diagnostics</h2>
            <p className="panel-sub">
              GET /health ·{" "}
              {health.lastChecked
                ? `Last checked ${new Date(health.lastChecked).toLocaleString()}`
                : "No health check completed yet"}
            </p>
          </div>
          <span className={`header-status ${health.status}`}>
            <span className="status-dot" />
            {apiValue}
          </span>
        </div>

        <div className="diagnostic-list">
          <DiagnosticRow
            label="FastAPI service"
            value={apiValue}
            note={health.details?.service || "Configured health endpoint"}
            state={state}
          />
          <DiagnosticRow
            label="Inference model"
            value={modelValue}
            note={
              health.status === "online"
                ? "The model is loaded during API process startup."
                : "The health endpoint provides no independent model readiness detail."
            }
            state={state}
          />
          <DiagnosticRow
            label="UCI sample dataset"
            value="Not checked"
            note="Dataset availability is checked by GET /sample, not by /health."
          />
          <DiagnosticRow
            label="Prediction history"
            value="Session only"
            note="The backend does not expose a persistent prediction history endpoint."
          />
        </div>

        {health.status === "offline" && (
          <div className="error-state status-error-state" role="alert">
            <Server size={17} />
            <div>
              <strong>Could not reach the FastAPI service</strong>
              <p>{health.error}</p>
              <p>
                Confirm the API is running and that <code>VITE_API_URL</code>{" "}
                points to its base URL.
              </p>
            </div>
          </div>
        )}

        {health.status === "online" && health.details && (
          <div className="health-response">
            <span>Health response</span>
            <code>{JSON.stringify(health.details)}</code>
            {health.durationMs != null && (
              <span>Response time: {health.durationMs} ms</span>
            )}
          </div>
        )}
      </section>
    </>
  );
}
