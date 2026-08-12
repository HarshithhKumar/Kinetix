# Build from repository root:
#   docker build -f backend/Dockerfile .

FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/app ./backend/app
COPY models ./models

ENV PYTHONPATH=/app/backend
ENV KINETIX_MODEL_PATH=/app/models/kinetix_lstm.keras
ENV KINETIX_SCALER_PATH=/app/models/scaler.pkl
ENV KINETIX_DATA_DIR=/app/data/UCI HAR Dataset

EXPOSE 8000

WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
