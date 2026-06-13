from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image
from ultralytics import YOLO

from .config import Settings


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class PredictionResult:
    detections: list[Detection]
    annotated_image_base64: str
    latency_ms: float
    image_width: int
    image_height: int


class DroneDetector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = YOLO(str(settings.weights_path))
        self.class_names = self.model.names

    def predict(self, image_bytes: bytes, confidence: float | None = None) -> PredictionResult:
        started = time.perf_counter()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_width, image_height = image.size

        conf = confidence if confidence is not None else self.settings.confidence_threshold
        results = self.model.predict(
            source=np.array(image),
            conf=conf,
            iou=self.settings.iou_threshold,
            imgsz=self.settings.input_size,
            device=self.settings.device,
            verbose=False,
        )

        detections: list[Detection] = []
        annotated = image

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                class_id = int(box.cls.item())
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=str(self.class_names[class_id]),
                        confidence=float(box.conf.item()),
                        x1=float(box.xyxy[0][0].item()),
                        y1=float(box.xyxy[0][1].item()),
                        x2=float(box.xyxy[0][2].item()),
                        y2=float(box.xyxy[0][3].item()),
                    )
                )

            plotted = results[0].plot()
            annotated = Image.fromarray(plotted[..., ::-1])

        buffer = io.BytesIO()
        annotated.save(buffer, format="JPEG", quality=90)
        annotated_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        latency_ms = (time.perf_counter() - started) * 1000

        return PredictionResult(
            detections=detections,
            annotated_image_base64=annotated_b64,
            latency_ms=latency_ms,
            image_width=image_width,
            image_height=image_height,
        )
