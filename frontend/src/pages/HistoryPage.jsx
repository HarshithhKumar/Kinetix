import { useMemo, useState } from "react";
import { Search, Trash2 } from "lucide-react";
import PageHeading from "../components/PageHeading";

export default function HistoryPage({
  entries,
  storageError,
  onClear,
}) {
  const [query, setQuery] = useState("");
  const filtered = useMemo(() => {
    const term = query.trim().toLowerCase();
    if (!term) return entries;
    return entries.filter(
      (entry) =>
        entry.activity.toLowerCase().includes(term) ||
        entry.source.toLowerCase().includes(term),
    );
  }, [entries, query]);

  return (
    <>
      <PageHeading
        title="Prediction History"
        description="Successful model responses recorded in this browser tab only. Entries are session-scoped and are not stored by the backend."
        actions={
          entries.length > 0 && (
            <button type="button" className="btn-secondary" onClick={onClear}>
              <Trash2 size={13} />
              Clear session
            </button>
          )
        }
      />

      <section className="panel history-panel">
        <div className="panel-head history-head">
          <div>
            <h2>Session predictions</h2>
            <p className="panel-sub">
              {entries.length}{" "}
              {entries.length === 1 ? "prediction" : "predictions"} in this tab
              session
            </p>
          </div>
          {entries.length > 0 && (
            <label className="history-search">
              <Search size={14} aria-hidden="true" />
              <input
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Filter activity or source"
                aria-label="Filter prediction history"
              />
            </label>
          )}
        </div>

        {storageError && (
          <p className="inline-warning" role="status">
            {storageError} New results will remain visible until this page is
            reloaded.
          </p>
        )}

        {entries.length === 0 ? (
          <div className="empty-state history-empty">
            <div>
              <strong>No predictions in this session yet</strong>
              <p>
                Run a Dataset Demo or Live Phone prediction to add a successful
                model response here.
              </p>
            </div>
          </div>
        ) : filtered.length === 0 ? (
          <div className="empty-state history-empty">
            No session predictions match “{query}”.
          </div>
        ) : (
          <div className="table-wrap">
            <table className="metrics-table history-table">
              <thead>
                <tr>
                  <th>Activity</th>
                  <th>Confidence</th>
                  <th>Input</th>
                  <th>Ground truth</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((entry) => (
                  <tr key={entry.id}>
                    <td>
                      <strong>{entry.activity}</strong>
                    </td>
                    <td>{(entry.confidence * 100).toFixed(2)}%</td>
                    <td>{entry.source === "dataset" ? "Dataset Demo" : "Live Phone"}</td>
                    <td>{entry.groundTruth || "—"}</td>
                    <td>
                      <time dateTime={entry.createdAt}>
                        {new Date(entry.createdAt).toLocaleString()}
                      </time>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}
