from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.face_engine import FaceDetection
from core.image_processor import (
    ImageLoadError,
    assess_quality,
    draw_faces,
    load_image,
    save_image,
)


def test_load_valid_image(sample_image_path: Path) -> None:
    image = load_image(sample_image_path)
    assert image.ndim == 3
    assert image.shape[2] == 3


def test_load_missing_image(tmp_path: Path) -> None:
    with pytest.raises(ImageLoadError):
        load_image(tmp_path / "missing.jpg")


def test_load_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_text("not an image")
    with pytest.raises(ImageLoadError):
        load_image(path)


def test_load_corrupt_image(corrupt_image: Path) -> None:
    with pytest.raises(ImageLoadError):
        load_image(corrupt_image)


def test_quality_good_face(engine, sample_image_path: Path) -> None:
    image = load_image(sample_image_path)
    face = engine.detect(image)[0]
    report = assess_quality(image, face)
    assert report.passed
    assert report.face_width > 0
    assert report.blur_variance > 0


def test_quality_rejects_tiny_face(sample_image_path: Path) -> None:
    image = load_image(sample_image_path)
    tiny = FaceDetection(
        index=0,
        x=0,
        y=0,
        w=20,
        h=20,
        score=0.9,
        raw=np.zeros(15, dtype=np.float32),
    )
    report = assess_quality(image, tiny)
    assert not report.passed
    assert any("smaller" in reason for reason in report.reasons)


def test_quality_rejects_dark_blurry_face() -> None:
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    face = FaceDetection(
        index=0,
        x=50,
        y=50,
        w=200,
        h=200,
        score=0.9,
        raw=np.zeros(15, dtype=np.float32),
    )
    report = assess_quality(image, face)
    assert not report.passed
    assert report.reasons


def test_draw_faces_does_not_mutate_input(engine, sample_image_path: Path) -> None:
    image = load_image(sample_image_path)
    detections = engine.detect(image)
    before = image.copy()
    annotated = draw_faces(image, detections, selected_index=0)
    assert np.array_equal(image, before)
    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)


def test_draw_faces_uses_custom_labels(
    engine, sample_image_path: Path, monkeypatch
) -> None:
    image = load_image(sample_image_path)
    detections = engine.detect(image)[:1]
    texts: list[str] = []

    def fake_put_text(img, text, *args, **kwargs):
        texts.append(text)
        return img

    monkeypatch.setattr(cv2, "putText", fake_put_text)
    draw_faces(image, detections, labels=["Alice 0.99"])
    assert "Alice 0.99" in texts


def test_save_image(tmp_path: Path, sample_image_path: Path) -> None:
    image = load_image(sample_image_path)
    output = save_image(tmp_path / "out.jpg", image)
    assert output.exists()
    assert load_image(output).shape == image.shape
