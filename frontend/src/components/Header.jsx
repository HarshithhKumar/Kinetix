import {
  Activity,
  BarChart3,
  Brain,
  Clock3,
  LayoutDashboard,
  Waves,
  X,
} from "lucide-react";
import { APP_ROUTES } from "../lib/routes";

const NAV_ICONS = {
  overview: LayoutDashboard,
  predict: Activity,
  analytics: BarChart3,
  history: Clock3,
  sensors: Waves,
  status: Brain,
};

const API_STATUS_LABEL = {
  checking: "Checking connection",
  online: "FastAPI connected",
  offline: "API unavailable",
};

export default function Header({
  apiStatus,
  currentPath,
  isOpen,
  onNavigate,
  onClose,
}) {
  return (
    <>
      <aside
        className={`sidebar ${isOpen ? "is-open" : ""}`}
        aria-label="Kinetix navigation"
      >
        <div className="sidebar-brand">
          <a
            href="/overview"
            className="sidebar-brand-link"
            onClick={(event) => {
              event.preventDefault();
              onNavigate("/overview");
            }}
          >
            <span className="brand-mark" aria-hidden="true">
              <Activity size={18} strokeWidth={2.2} />
            </span>
            <span className="brand-copy">
              <strong>Kinetix</strong>
              <span>Human Activity Recognition</span>
            </span>
          </a>
          <button
            type="button"
            className="sidebar-close"
            aria-label="Close navigation menu"
            onClick={onClose}
          >
            <X size={17} />
          </button>
        </div>

        <nav className="sidebar-nav" aria-label="Main navigation">
          <span className="nav-caption">Workspace</span>
          {APP_ROUTES.map(({ path, label, icon }) => {
            const Icon = NAV_ICONS[icon];
            const isActive = currentPath === path;
            return (
              <a
                key={path}
                className={`nav-link ${isActive ? "active" : ""}`}
                href={path}
                aria-current={isActive ? "page" : undefined}
                onClick={(event) => {
                  event.preventDefault();
                  onNavigate(path);
                }}
              >
                <Icon size={15} strokeWidth={1.8} aria-hidden="true" />
                <span>{label}</span>
              </a>
            );
          })}
        </nav>

        <footer className="sidebar-footer">
          <div className={`sidebar-api ${apiStatus}`}>
            <span className="status-dot" />
            <div className="sidebar-api-copy">
              <small>BACKEND</small>
              <strong>{API_STATUS_LABEL[apiStatus]}</strong>
            </div>
          </div>
          <p className="sidebar-source">
            UCI HAR Dataset
            <br />
            128-sample motion windows
          </p>
        </footer>
      </aside>
      {isOpen && (
        <button
          type="button"
          className="sidebar-backdrop"
          aria-label="Close navigation menu"
          onClick={onClose}
        />
      )}
    </>
  );
}
