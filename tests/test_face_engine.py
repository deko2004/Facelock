from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.face_engine import FaceEngine, FaceEngineError
from tests.conftest import BIDEN, OBAMA_1


def _feature(engine: FaceEngine, path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    assert image is not None
    detections = engine.detect(image)
    assert detections
    return engine.extract_feature(image, detections[0])


def test_models_load(engine: FaceEngine) -> None:
    assert engine.detector_label
    assert engine.recognizer_label
    assert engine.cosine_constant is not None


def test_detect_face_in_sample(engine: FaceEngine, sample_image_path: Path) -> None:
    image = cv2.imread(str(sample_image_path))
    detections = engine.detect(image)
    assert len(detections) == 1
    face = detections[0]
    assert face.w > 0 and face.h > 0
    assert 0.0 <= face.score <= 1.0
    assert face.raw.size >= 15


def test_no_face_image(engine: FaceEngine, no_face_image: Path) -> None:
    image = cv2.imread(str(no_face_image))
    assert engine.detect(image) == []


def test_multiple_face_image(engine: FaceEngine, multi_face_image: Path) -> None:
    image = cv2.imread(str(multi_face_image))
    detections = engine.detect(image)
    assert len(detections) >= 2


def test_feature_extraction_shape(engine: FaceEngine, sample_image_path: Path) -> None:
    image = cv2.imread(str(sample_image_path))
    face = engine.detect(image)[0]
    feature = engine.extract_feature(image, face)
    assert feature.shape == (1, 128)
    assert feature.dtype == np.float32
    assert np.isfinite(feature).all()


def test_self_cosine_similarity(engine: FaceEngine, sample_image_path: Path) -> None:
    feature = _feature(engine, sample_image_path)
    score = engine.cosine_similarity(feature, feature)
    assert score == pytest.approx(1.0, abs=1e-5)


def test_official_self_similarity(engine: FaceEngine, sample_image_path: Path) -> None:
    feature = _feature(engine, sample_image_path)
    score = engine.official_similarity(feature, feature)
    assert score == pytest.approx(1.0, abs=1e-3)


def test_different_people_score_lower(engine: FaceEngine) -> None:
    obama = _feature(engine, OBAMA_1)
    biden = _feature(engine, BIDEN)
    same = engine.cosine_similarity(obama, obama)
    different = engine.cosine_similarity(obama, biden)
    assert different < same
    assert different < 0.363


def test_l2_distance_self(engine: FaceEngine, sample_image_path: Path) -> None:
    feature = _feature(engine, sample_image_path)
    distance = engine.official_l2_distance(feature, feature)
    assert distance == pytest.approx(0.0, abs=1e-3)


def test_missing_models_raise(tmp_path: Path) -> None:
    with pytest.raises(FaceEngineError):
        FaceEngine(
            detector_path=tmp_path / "missing_detector.onnx",
            recognizer_path=tmp_path / "missing_recognizer.onnx",
        )


def test_empty_image_returns_no_faces(engine: FaceEngine) -> None:
    tiny = np.zeros((5, 5, 3), dtype=np.uint8)
    assert engine.detect(tiny) == []
