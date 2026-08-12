"""Central configuration for the Kinetix backend (pathlib + environment)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(BACKEND_DIR / ".env")


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name, "").strip()
    if value:
        return Path(value).expanduser().resolve()
    return default.resolve()


MODEL_PATH = _path_from_env(
    "KINETIX_MODEL_PATH",
    PROJECT_ROOT / "models" / "kinetix_lstm.keras",
)
SCALER_PATH = _path_from_env(
    "KINETIX_SCALER_PATH",
    PROJECT_ROOT / "models" / "scaler.pkl",
)
DATA_DIR = _path_from_env(
    "KINETIX_DATA_DIR",
    PROJECT_ROOT / "data" / "UCI HAR Dataset",
)


def parse_cors_origins() -> list[str]:
    raw = os.getenv(
        "KINETIX_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def cors_origin_regex() -> str | None:
    """Return regex for LAN dev origins, or None to disable regex matching."""
    raw = os.getenv("KINETIX_CORS_ORIGIN_REGEX", "").strip()
    if raw.lower() in {"", "none", "null", "false", "0"}:
        return None
    if raw:
        return raw
    return (
        r"https?://(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|"
        r"10\.\d{1,3}\.\d{1,3}\.\d{1,3})(:\d+)?"
    )
