from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"

IMAGE_CASES: list[tuple[str, dict[str, int]]] = [
    ("visdrone_plaza.png", {"pedestrian": 15}),
    ("visdrone_parking.png", {"car": 50, "pedestrian": 1}),
    ("visdrone_street.png", {"pedestrian": 1}),
]

REAL_IMAGES = [
    pytest.param(filename, min_counts, id=Path(filename).stem)
    for filename, min_counts in IMAGE_CASES
]


@pytest.fixture()
def client():
    from service.app.main import app

    with TestClient(app) as test_client:
        yield test_client


def _count_by_class(detections: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in detections:
        name = item["class_name"]
        counts[name] = counts.get(name, 0) + 1
    return counts


@pytest.mark.parametrize("filename,min_counts", REAL_IMAGES)
def test_predict_on_real_drone_image(
    client: TestClient,
    filename: str,
    min_counts: dict[str, int],
) -> None:
    image_path = TEST_DATA_DIR / filename
    assert image_path.exists(), f"Тестовое изображение не найдено: {image_path}"

    with image_path.open("rb") as image_file:
        response = client.post(
            "/predict",
            files={"file": (filename, image_file, "image/png")},
            params={"confidence": 0.25},
        )

    assert response.status_code == 200
    payload = response.json()

    assert payload["filename"] == filename
    assert payload["detections_count"] == len(payload["detections"])
    assert payload["image_width"] > 0
    assert payload["image_height"] > 0
    assert payload["latency_ms"] > 0
    assert len(payload["annotated_image_base64"]) > 100

    actual = _count_by_class(payload["detections"])
    for class_name, minimum in min_counts.items():
        assert actual.get(class_name, 0) >= minimum, (
            f"{filename}: ожидалось >= {minimum} объектов класса '{class_name}', "
            f"получено {actual.get(class_name, 0)}. Все классы: {actual}"
        )


@pytest.mark.parametrize("filename,min_counts", REAL_IMAGES)
def test_detector_direct_on_real_images(filename: str, min_counts: dict[str, int]) -> None:
    from service.app.config import load_settings
    from service.app.detector import DroneDetector

    image_path = TEST_DATA_DIR / filename
    settings = load_settings()
    detector = DroneDetector(settings)
    result = detector.predict(image_path.read_bytes(), confidence=0.25)

    actual: dict[str, int] = {}
    for detection in result.detections:
        actual[detection.class_name] = actual.get(detection.class_name, 0) + 1

    for class_name, minimum in min_counts.items():
        assert actual.get(class_name, 0) >= minimum, (
            f"{filename}: ожидалось >= {minimum} '{class_name}', получено {actual}"
        )

    assert result.latency_ms > 0
    assert len(result.annotated_image_base64) > 100


def test_batch_predict_on_all_real_images(client: TestClient) -> None:
    files = []
    for filename, _ in IMAGE_CASES:
        image_path = TEST_DATA_DIR / filename
        files.append(
            ("files", (filename, image_path.read_bytes(), "image/png")),
        )

    response = client.post("/batch_predict", files=files, params={"confidence": 0.25})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == len(IMAGE_CASES)

    total_detections = sum(item["detections_count"] for item in payload)
    assert total_detections >= 90

    filenames = {item["filename"] for item in payload}
    assert filenames == {filename for filename, _ in IMAGE_CASES}
