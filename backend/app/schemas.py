from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    sensor_data: List[List[float]]
    input_source: Optional[Literal["dataset", "live"]] = Field(
        default=None,
        description="Optional hint: dataset demo or live browser sensors.",
    )


class PredictionResponse(BaseModel):
    activity: str
    confidence: float
    probabilities: Dict[str, float]


class SampleResponse(BaseModel):
    sensor_data: List[List[float]]
    ground_truth: str
    shape: List[int]
    feature_names: List[str]


class ActivityPrediction(PredictionResponse):
    pass
