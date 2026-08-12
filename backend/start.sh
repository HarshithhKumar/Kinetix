#!/bin/sh
set -eu

# Render and other PaaS platforms inject PORT; default 8000 for local Docker.
PORT="${PORT:-8000}"

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}"
