from __future__ import annotations

from pydantic import BaseModel, Field


class DetectionSchema(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class PredictResponse(BaseModel):
    id: int
    filename: str
    model_version: str
    architecture: str
    detections: list[DetectionSchema]
    detections_count: int
    latency_ms: float
    image_width: int
    image_height: int
    annotated_image_base64: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    architecture: str
    classes: list[str]


class StatsResponse(BaseModel):
    total_predictions: int
    average_latency_ms: float
    average_detections: float
    error_count: int


class HistoryItem(BaseModel):
    id: int
    created_at: str
    filename: str
    model_version: str
    detections_count: int
    latency_ms: float
    confidence_threshold: float
    detections: list[DetectionSchema] = Field(default_factory=list)
    result_image_path: str | None = None
    error: str | None = None
