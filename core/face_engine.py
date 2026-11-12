"""YuNet face detection and SFace feature extraction.

OpenCV 5 renamed the FaceRecognizerSF distance constants that existed in
OpenCV 4.x (``FaceRecognizerSF_DIS_TYPE_COSINE`` became
``FaceRecognizerSF_FR_COSINE``).  This module resolves both names so the same
application works on either major version.  Cosine similarity is computed
explicitly so that its meaning is always "higher means more similar", never a
distance.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import settings


class FaceEngineError(RuntimeError):
    """Raised when a model cannot be loaded or inference fails."""


@dataclass(frozen=True)
class FaceDetection:
    """A single YuNet detection and the raw row needed by SFace alignment."""

    index: int
    x: int
    y: int
    w: int
    h: int
    score: float
    raw: np.ndarray

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    @property
    def label(self) -> str:
        return f"Face {self.index + 1} ({self.w}x{self.h}, score {self.score:.2f})"


def _resolve_cosine_constant() -> int:
    """Return the OpenCV cosine constant across OpenCV 4.x and 5.x."""
    for name in ("FaceRecognizerSF_FR_COSINE", "FaceRecognizerSF_DIS_TYPE_COSINE"):
        value = getattr(cv2, name, None)
        if value is not None:
            return int(value)
    return 0


def _resolve_l2_constant() -> int:
    for name in ("FaceRecognizerSF_FR_NORM_L2", "FaceRecognizerSF_DIS_TYPE_L2"):
        value = getattr(cv2, name, None)
        if value is not None:
            return int(value)
    return 1


class FaceEngine:
    """Load YuNet + SFace once and expose thread-safe inference methods."""

    def __init__(
        self,
        detector_path: Path | str = settings.YUNET_MODEL,
        recognizer_path: Path | str = settings.SFACE_MODEL,
        score_threshold: float = settings.YUNET_SCORE_THRESHOLD,
        nms_threshold: float = settings.YUNET_NMS_THRESHOLD,
        top_k: int = settings.YUNET_TOP_K,
        detection_max_width: int = settings.DETECTION_MAX_WIDTH,
    ) -> None:
        cv2.setNumThreads(max(1, int(settings.OPENCV_NUM_THREADS)))
        self.detector_path = Path(detector_path)
        self.recognizer_path = Path(recognizer_path)
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.detection_max_width = max(0, int(detection_max_width))
        self._lock = threading.RLock()
        self._cosine_type = _resolve_cosine_constant()
        self._l2_type = _resolve_l2_constant()
        self._detector: cv2.FaceDetectorYN | None = None
        self._recognizer: cv2.FaceRecognizerSF | None = None
        self._load_models()

    def _load_models(self) -> None:
        if not self.detector_path.is_file():
            raise FaceEngineError(
                f"YuNet model not found: {self.detector_path}. "
                "Place face_detection_yunet_2023mar.onnx in models/."
            )
        if not self.recognizer_path.is_file():
            raise FaceEngineError(
                f"SFace model not found: {self.recognizer_path}. "
                "Place face_recognition_sface_2021dec.onnx in models/."
            )
        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(self.detector_path),
                "",
                (320, 320),
                self.score_threshold,
                self.nms_threshold,
                self.top_k,
            )
            self._recognizer = cv2.FaceRecognizerSF.create(
                str(self.recognizer_path), ""
            )
        except cv2.error as exc:
            raise FaceEngineError(f"OpenCV could not load the models: {exc}") from exc

        if self._detector is None or self._recognizer is None:
            raise FaceEngineError("OpenCV returned an empty model object.")

    @property
    def detector_label(self) -> str:
        return settings.DETECTOR_LABEL

    @property
    def recognizer_label(self) -> str:
        return settings.MODEL_LABEL

    @property
    def cosine_constant(self) -> int:
        return self._cosine_type

    @property
    def l2_constant(self) -> int:
        return self._l2_type

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise FaceEngineError("Image must be a NumPy array.")
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.ndim != 3 or image.shape[2] != 3:
            raise FaceEngineError("Image must have 1, 3 or 4 channels.")
        return image

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """Detect faces with YuNet, returning detections in model order.

        If ``detection_max_width`` is set, YuNet runs on a down-scaled copy and
        the returned landmarks are mapped back to the original frame so SFace
        alignment still uses full-resolution pixels.
        """
        bgr = self._ensure_bgr(image)
        height, width = bgr.shape[:2]
        if height < 10 or width < 10:
            return []

        scale = 1.0
        detect_image = bgr
        if self.detection_max_width and width > self.detection_max_width:
            scale = self.detection_max_width / float(width)
            detect_image = cv2.resize(
                bgr,
                (
                    max(10, int(round(width * scale))),
                    max(10, int(round(height * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
        detect_height, detect_width = detect_image.shape[:2]

        with self._lock:
            assert self._detector is not None
            try:
                self._detector.setInputSize((int(detect_width), int(detect_height)))
                _, faces = self._detector.detect(detect_image)
            except cv2.error as exc:
                raise FaceEngineError(f"Face detection failed: {exc}") from exc

        if faces is None:
            return []
        array = np.asarray(faces)
        if array.size == 0:
            return []
        if array.ndim == 1:
            array = array.reshape(1, -1)

        detections: list[FaceDetection] = []
        for index, row in enumerate(array):
            raw = np.asarray(row, dtype=np.float32).copy()
            if scale != 1.0:
                raw[:14] /= scale
            values = [int(round(float(v))) for v in raw[:4]]
            x, y, w, h = values
            clipped_x = max(0, min(x, width - 1))
            clipped_y = max(0, min(y, height - 1))
            clipped_w = max(0, min(w, width - clipped_x))
            clipped_h = max(0, min(h, height - clipped_y))
            score = float(raw[-1]) if raw.size >= 15 else 0.0
            detections.append(
                FaceDetection(
                    index=index,
                    x=clipped_x,
                    y=clipped_y,
                    w=clipped_w,
                    h=clipped_h,
                    score=score,
                    raw=raw,
                )
            )
        return detections

    def largest_face(self, detections: list[FaceDetection]) -> FaceDetection | None:
        if not detections:
            return None
        return max(detections, key=lambda face: (face.area, face.score))

    def align_crop(self, image: np.ndarray, detection: FaceDetection) -> np.ndarray:
        """Align and crop a detection to the 112x112 SFace input."""
        bgr = self._ensure_bgr(image)
        with self._lock:
            assert self._recognizer is not None
            try:
                aligned = self._recognizer.alignCrop(bgr, detection.raw)
            except cv2.error as exc:
                raise FaceEngineError(f"Face alignment failed: {exc}") from exc
        if aligned is None or aligned.size == 0:
            raise FaceEngineError("Face alignment returned an empty crop.")
        return aligned

    def extract_feature(
        self, image: np.ndarray, detection: FaceDetection
    ) -> np.ndarray:
        """Return the SFace embedding for a detection as float32 (1, 128)."""
        aligned = self.align_crop(image, detection)
        with self._lock:
            assert self._recognizer is not None
            try:
                feature = self._recognizer.feature(aligned)
            except cv2.error as exc:
                raise FaceEngineError(f"Feature extraction failed: {exc}") from exc
        vector = np.asarray(feature, dtype=np.float32)
        if vector.size == 0:
            raise FaceEngineError("SFace returned an empty feature vector.")
        if not np.isfinite(vector).all():
            raise FaceEngineError("SFace returned non-finite feature values.")
        return vector.reshape(1, -1)

    @staticmethod
    def cosine_similarity(feature_a: np.ndarray, feature_b: np.ndarray) -> float:
        """Compute cosine similarity in [-1, 1]; 1.0 means identical direction.

        This is a similarity score, not a probability that two faces belong to
        the same person.
        """
        a = np.asarray(feature_a, dtype=np.float64).ravel()
        b = np.asarray(feature_b, dtype=np.float64).ravel()
        if a.size == 0 or b.size == 0:
            raise FaceEngineError("Cannot compare empty feature vectors.")
        if a.size != b.size:
            raise FaceEngineError(
                f"Feature dimensions differ ({a.size} vs {b.size})."
            )
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            raise FaceEngineError("Cannot compare zero-length feature vectors.")
        similarity = float(np.dot(a, b) / (norm_a * norm_b))
        return max(-1.0, min(1.0, similarity))

    def official_similarity(
        self, feature_a: np.ndarray, feature_b: np.ndarray
    ) -> float:
        """Use OpenCV's FaceRecognizerSF.match cosine implementation."""
        with self._lock:
            assert self._recognizer is not None
            try:
                value = self._recognizer.match(
                    np.asarray(feature_a, dtype=np.float32),
                    np.asarray(feature_b, dtype=np.float32),
                    self._cosine_type,
                )
            except cv2.error as exc:
                raise FaceEngineError(f"SFace matching failed: {exc}") from exc
        return float(value)

    def official_l2_distance(
        self, feature_a: np.ndarray, feature_b: np.ndarray
    ) -> float:
        """Return the optional OpenCV L2 distance for reference/debugging."""
        with self._lock:
            assert self._recognizer is not None
            try:
                value = self._recognizer.match(
                    np.asarray(feature_a, dtype=np.float32),
                    np.asarray(feature_b, dtype=np.float32),
                    self._l2_type,
                )
            except cv2.error as exc:
                raise FaceEngineError(f"SFace L2 matching failed: {exc}") from exc
        return float(value)
