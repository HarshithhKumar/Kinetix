# Build from repository root:
#   docker build -f backend/Dockerfile -t kinetix-backend .

FROM python:3.11-slim

WORKDIR /app

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
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
