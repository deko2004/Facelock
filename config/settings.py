"""Central configuration for the Biometric Authentication System.

The values in this module are deliberately conservative.  The match threshold is
a starting point for demonstrations only: it must be calibrated on a validation
set before it can be used for any serious decision.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
IMAGES_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"
DOCS_DIR = BASE_DIR / "docs"

YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

ANTI_SPOOFING_DIR = MODELS_DIR / "anti_spoofing"
PAD_MODEL = Path(
    os.environ.get(
        "FACE_PAD_MODEL",
        str(ANTI_SPOOFING_DIR / "MiniFASNetV2.onnx"),
    )
)
PAD_MODEL_LABEL = "MiniFASNetV2 (ONNX)"
PAD_INPUT_SIZE = (80, 80)
PAD_CROP_SCALE = 2.7
PAD_CLASS_LIVE_INDEX = 1
PAD_LIVE_THRESHOLD = float(os.environ.get("FACE_PAD_LIVE_THRESHOLD", "0.80"))
PAD_SPOOF_THRESHOLD = float(os.environ.get("FACE_PAD_SPOOF_THRESHOLD", "0.60"))
PAD_REQUIRED_LIVE_FRAMES = int(os.environ.get("FACE_PAD_REQUIRED_LIVE_FRAMES", "3"))
PAD_FRAME_INTERVAL = int(os.environ.get("FACE_PAD_FRAME_INTERVAL", "2"))
PAD_FACE_IOU_THRESHOLD = float(os.environ.get("FACE_PAD_FACE_IOU_THRESHOLD", "0.35"))

DATABASE_FILE = DATABASE_DIR / "biometric.db"
TEMPLATE_KEY_FILE = DATABASE_DIR / "templates.key"
APP_SETTINGS_FILE = DATABASE_DIR / "app_settings.json"
EVALUATION_DIR = RESULTS_DIR / "evaluation"

APP_NAME = "BIOMETRIC AUTHENTICATION SYSTEM"
APP_SUBTITLE = "Face Recognition and Verification"
APP_VERSION = "1.0.0"

DEFAULT_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.363"))
MIN_THRESHOLD = 0.0
MAX_THRESHOLD = 1.0

YUNET_SCORE_THRESHOLD = 0.80
YUNET_NMS_THRESHOLD = 0.30
YUNET_TOP_K = 5000

MIN_FACE_SIZE_PX = 80
MIN_BLUR_VARIANCE = 20.0
MIN_BRIGHTNESS = 30.0
MAX_BRIGHTNESS = 240.0
MAX_IMAGE_DIMENSION = 2400

EMBEDDING_DIM = 128
MODEL_LABEL = "SFace 2021dec (128-D)"
DETECTOR_LABEL = "YuNet 2023mar"

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

#: OpenCV's published SFace cosine threshold for its LFW calibration.  This is
#: *not* a universal threshold and is not calibrated for this application.
OPENCV_SFACE_COSINE_THRESHOLD = 0.363


def ensure_directories() -> None:
    """Create the runtime directories if they do not already exist."""
    for directory in (
        DATABASE_DIR,
        IMAGES_DIR,
        RESULTS_DIR,
        EVALUATION_DIR,
        ANTI_SPOOFING_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def clamp_threshold(value: float) -> float:
    """Return *value* limited to the valid similarity range [0, 1]."""
    return max(MIN_THRESHOLD, min(MAX_THRESHOLD, float(value)))


def load_threshold(path: Path | None = None) -> float:
    """Load the persisted threshold, falling back to the configured default."""
    settings_path = Path(path) if path is not None else APP_SETTINGS_FILE
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        return clamp_threshold(float(data.get("threshold", DEFAULT_THRESHOLD)))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return clamp_threshold(DEFAULT_THRESHOLD)


def save_threshold(value: float, path: Path | None = None) -> None:
    """Persist the threshold in a small JSON settings file."""
    settings_path = Path(path) if path is not None else APP_SETTINGS_FILE
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"threshold": clamp_threshold(value)}
    settings_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        settings_path.chmod(0o600)
    except OSError:
        pass
