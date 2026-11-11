"""Shared pytest fixtures.  No physical webcam is required."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from config import settings
from core.app_context import AppContext
from core.database import FaceDatabase
from core.face_engine import FaceEngine
from core.security import TemplateCipher

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = (
    ROOT
    / "images"
    / "handsome-caucasian-man-wearing-casual-clothes-glasses-with-happy-cool-smile-face-lucky-person_839833-12772.jpg"
)
OBAMA_1 = ROOT / "images" / "eval_demo" / "obama" / "obama_01.jpg"
OBAMA_2 = ROOT / "images" / "eval_demo" / "obama" / "obama_02.jpg"
BIDEN = ROOT / "images" / "eval_demo" / "biden" / "biden_01.jpg"
EVAL_DEMO = ROOT / "images" / "eval_demo"


def models_available() -> bool:
    return settings.YUNET_MODEL.is_file() and settings.SFACE_MODEL.is_file()


@pytest.fixture(scope="session")
def engine() -> FaceEngine:
    if not models_available():
        pytest.skip("ONNX models are not present in models/")
    return FaceEngine()


@pytest.fixture(scope="session")
def sample_image_path() -> Path:
    if not SAMPLE_IMAGE.is_file():
        pytest.skip("Sample image is missing")
    return SAMPLE_IMAGE


@pytest.fixture()
def context(tmp_path: Path) -> AppContext:
    if not models_available():
        pytest.skip("ONNX models are not present in models/")
    return AppContext.create(
        db_path=tmp_path / "test.db",
        key_path=tmp_path / "test.key",
        settings_path=tmp_path / "settings.json",
        threshold=0.363,
    )


@pytest.fixture()
def cipher(tmp_path: Path) -> TemplateCipher:
    return TemplateCipher(tmp_path / "test.key")


@pytest.fixture()
def database(tmp_path: Path, cipher: TemplateCipher) -> FaceDatabase:
    return FaceDatabase(tmp_path / "test.db", cipher)


@pytest.fixture()
def no_face_image(tmp_path: Path) -> Path:
    path = tmp_path / "no_face.jpg"
    cv2.imwrite(str(path), np.zeros((300, 300, 3), dtype=np.uint8))
    return path


@pytest.fixture()
def multi_face_image(tmp_path: Path, sample_image_path: Path) -> Path:
    image = cv2.imread(str(sample_image_path))
    assert image is not None
    combined = np.hstack([image, image])
    path = tmp_path / "multi_face.jpg"
    cv2.imwrite(str(path), combined)
    return path


@pytest.fixture()
def corrupt_image(tmp_path: Path) -> Path:
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"this is not a JPEG image")
    return path


@pytest.fixture()
def has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
