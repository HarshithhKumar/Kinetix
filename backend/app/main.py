from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import cors_origin_regex, dataset_is_available, parse_cors_origins
from .data_loader import get_raw_sample
from .model import predict_activity
from .preprocessing import preprocess_sensor_data
from .schemas import PredictionRequest, PredictionResponse, SampleResponse

app = FastAPI(title="Kinetix HAR API")

_cors_kwargs: dict = {
    "allow_origins": parse_cors_origins(),
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}
_origin_regex = cors_origin_regex()
if _origin_regex:
    _cors_kwargs["allow_origin_regex"] = _origin_regex

app.add_middleware(CORSMiddleware, **_cors_kwargs)


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "kinetix-har-api"}


@app.get("/sample", response_model=SampleResponse)
def get_sample() -> SampleResponse:
    if not dataset_is_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "UCI HAR dataset not available in this deployment. "
                "POST /predict accepts a 128×6 sensor window directly."
            ),
        )
    try:
        payload = get_raw_sample()
        return SampleResponse(**payload)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "UCI HAR dataset not found. Download the dataset and set "
                "KINETIX_DATA_DIR if needed. See README.md."
            ),
        ) from exc
    except IndexError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        processed = preprocess_sensor_data(request.sensor_data)
        activity, confidence, probabilities = predict_activity(processed)
        return PredictionResponse(
            activity=activity,
            confidence=confidence,
            probabilities=probabilities,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
