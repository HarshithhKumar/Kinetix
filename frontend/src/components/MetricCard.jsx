export default function MetricCard({ label, value, note, icon: Icon, tone = "blue" }) {
  return (
    <article className={`overview-stat stat-${tone}`}>
      <span className="stat-label">
        {Icon && <Icon size={12} aria-hidden="true" />}
        {label}
      </span>
      <strong className="stat-value">{value}</strong>
      {note && <span className="stat-note">{note}</span>}
    </article>
  );
}
