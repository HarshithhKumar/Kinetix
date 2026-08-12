# Production Docker image for Kinetix FastAPI backend (Render-ready).
# Build from repository root:
#   docker build -t kinetix-backend .
# Run locally:
#   docker run --rm -e PORT=8000 -p 8000:8000 kinetix-backend

FROM python:3.11-slim

WORKDIR /app

# TensorFlow runtime dependency on Debian slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/app ./backend/app

COPY models/kinetix_lstm.keras ./models/kinetix_lstm.keras
COPY models/scaler.pkl ./models/scaler.pkl

ENV PYTHONPATH=/app/backend
ENV KINETIX_MODEL_PATH=/app/models/kinetix_lstm.keras
ENV KINETIX_SCALER_PATH=/app/models/scaler.pkl
ENV KINETIX_DATA_DIR=/app/data
ENV KINETIX_CORS_ORIGIN_REGEX=none
ENV PORT=8000

WORKDIR /app/backend

# Render injects PORT; shell form expands ${PORT:-8000}
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
