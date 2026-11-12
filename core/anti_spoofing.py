"""Optional MiniFASNetV2 presentation-attack detection.

The model contract is intentionally kept close to the upstream Silent-Face
implementation:

* BGR face crop expanded by ``2.7`` and resized to ``80x80``.
* ``float32`` pixels in the ``0..255`` range.  The upstream ``ToTensor`` does
  **not** divide by 255, so adding ``/255`` here silently destroys accuracy.
* Three logits where index ``1`` is live/real and indices ``0``/``2`` are
  presentation attacks.

The detector is optional.  Importing this module does not require
``onnxruntime``; a clear error is raised only when a caller tries to load the
model without the dependency installed.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import settings
from core.face_engine import FaceDetection

try:  # optional dependency: keep the rest of the application usable without it
    import onnxruntime as ort
except ImportError:  # pragma: no cover - exercised only on minimal installs
    ort = None


PAD_DISABLED = "DISABLED"
PAD_READY = "READY"
PAD_LOADING = "LOADING"
PAD_ERROR = "ERROR"
PAD_NO_FACE = "NO_FACE"
PAD_ANALYZING = "ANALYZING"
PAD_LIVE = "LIVE"
PAD_SPOOF = "SPOOF"
PAD_UNCERTAIN = "UNCERTAIN"


class AntiSpoofingError(RuntimeError):
    """Raised when the PAD model or its runtime contract cannot be used."""


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _int_dim(value: object) -> int | None:
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _bbox_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection <= 0:
        return 0.0
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class PadPrediction:
    """One raw PAD decision before temporal smoothing."""

    class_scores: tuple[float, ...]
    live_score: float
    spoof_score: float
    label: str
    class_index: int
    confidence: float

    @property
    def is_live(self) -> bool:
        return self.label == "LIVE"


@dataclass(frozen=True)
class PadConfig:
    live_threshold: float = settings.PAD_LIVE_THRESHOLD
    spoof_threshold: float = settings.PAD_SPOOF_THRESHOLD
    required_live_frames: int = settings.PAD_REQUIRED_LIVE_FRAMES
    frame_interval: int = settings.PAD_FRAME_INTERVAL
    face_iou_threshold: float = settings.PAD_FACE_IOU_THRESHOLD

    def __post_init__(self) -> None:
        object.__setattr__(self, "live_threshold", _clamp01(self.live_threshold))
        object.__setattr__(self, "spoof_threshold", _clamp01(self.spoof_threshold))
        object.__setattr__(
            self, "required_live_frames", max(1, int(self.required_live_frames))
        )
        object.__setattr__(self, "frame_interval", max(1, int(self.frame_interval)))
        object.__setattr__(
            self, "face_iou_threshold", _clamp01(self.face_iou_threshold)
        )


@dataclass(frozen=True)
class PadAssessment:
    """Temporally smoothed PAD state for one video frame."""

    status: str
    verified: bool
    spoof_detected: bool
    live_score: float | None
    confidence: float | None
    live_streak: int
    required_live_frames: int
    message: str
    prediction: PadPrediction | None = None


class AntiSpoofingDetector:
    """Load and run MiniFASNetV2 on a single face crop at a time."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        live_threshold: float = settings.PAD_LIVE_THRESHOLD,
        spoof_threshold: float = settings.PAD_SPOOF_THRESHOLD,
        crop_scale: float = settings.PAD_CROP_SCALE,
        input_size: tuple[int, int] = settings.PAD_INPUT_SIZE,
        providers: list[str] | None = None,
        num_threads: int = settings.PAD_NUM_THREADS,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else settings.PAD_MODEL
        self.live_threshold = _clamp01(live_threshold)
        self.spoof_threshold = _clamp01(spoof_threshold)
        self.crop_scale = max(1.0, float(crop_scale))
        self._input_size = (max(1, int(input_size[0])), max(1, int(input_size[1])))
        self._providers = list(providers or ["CPUExecutionProvider"])
        self._num_threads = max(1, int(num_threads))
        self._lock = threading.RLock()
        self._session = None
        self._input_name = ""
        self._output_name = ""
        self._class_count = 3
        self._load()

    @property
    def input_size(self) -> tuple[int, int]:
        """Return ``(width, height)`` expected by the ONNX graph."""
        return self._input_size

    @property
    def class_count(self) -> int:
        return self._class_count

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._session is not None

    def _load(self) -> None:
        if ort is None:
            raise AntiSpoofingError(
                "onnxruntime is not installed. Install the project requirements "
                "with: pip install -r requirements.txt"
            )
        if not self.model_path.is_file():
            raise AntiSpoofingError(
                f"Anti-spoofing model not found: {self.model_path}. "
                "See models/anti_spoofing/README.md for the download details."
            )
        if self.model_path.suffix.lower() != ".onnx":
            raise AntiSpoofingError(
                f"Anti-spoofing model must be an ONNX file: {self.model_path}"
            )
        try:
            options = ort.SessionOptions()
            options.intra_op_num_threads = self._num_threads
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=self._providers,
            )
        except Exception as exc:  # onnxruntime raises several backend-specific errors
            raise AntiSpoofingError(
                f"Could not load the anti-spoofing model: {exc}"
            ) from exc

        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise AntiSpoofingError(
                "MiniFASNetV2 must expose exactly one input and one output tensor."
            )
        input_cfg = inputs[0]
        if len(input_cfg.shape) != 4:
            raise AntiSpoofingError(
                "MiniFASNetV2 input must have shape [batch, 3, height, width]."
            )
        if "float" not in str(input_cfg.type).lower():
            raise AntiSpoofingError(
                f"MiniFASNetV2 input must be float32, got {input_cfg.type!r}."
            )
        channels = _int_dim(input_cfg.shape[1])
        if channels is not None and channels != 3:
            raise AntiSpoofingError(
                f"MiniFASNetV2 expects 3 input channels, got {channels}."
            )
        width = _int_dim(input_cfg.shape[3]) or self._input_size[0]
        height = _int_dim(input_cfg.shape[2]) or self._input_size[1]
        self._input_size = (width, height)

        output_cfg = outputs[0]
        if len(output_cfg.shape) != 2:
            raise AntiSpoofingError(
                "MiniFASNetV2 output must have shape [batch, classes]."
            )
        classes = _int_dim(output_cfg.shape[1]) or 3
        if classes != 3:
            raise AntiSpoofingError(
                f"MiniFASNetV2 must return 3 classes, got {classes}."
            )
        if settings.PAD_CLASS_LIVE_INDEX >= classes:
            raise AntiSpoofingError(
                "Configured live class index is outside the model output."
            )

        self._session = session
        self._input_name = input_cfg.name
        self._output_name = output_cfg.name
        self._class_count = classes

    def close(self) -> None:
        with self._lock:
            self._session = None

    def crop_face(
        self, image: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> np.ndarray:
        """Return the exact BGR crop used by the upstream model pipeline."""
        image = self._ensure_bgr(image)
        x, y, w, h = (int(value) for value in bbox)
        src_h, src_w = image.shape[:2]
        if w <= 0 or h <= 0:
            raise AntiSpoofingError("Face bounding box has no area.")

        scale = min((src_h - 1) / h, (src_w - 1) / w, self.crop_scale)
        new_w = w * scale
        new_h = h * scale
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        x1 = max(0, int(center_x - new_w / 2.0))
        y1 = max(0, int(center_y - new_h / 2.0))
        x2 = min(src_w - 1, int(center_x + new_w / 2.0))
        y2 = min(src_h - 1, int(center_y + new_h / 2.0))
        if x2 <= x1 or y2 <= y1:
            raise AntiSpoofingError("Face crop is outside the image.")
        cropped = image[y1 : y2 + 1, x1 : x2 + 1]
        return cv2.resize(cropped, self._input_size, interpolation=cv2.INTER_LINEAR)

    def predict(self, image: np.ndarray, detection: FaceDetection) -> PadPrediction:
        """Crop a detected face and run PAD inference."""
        return self.predict_crop(self.crop_face(image, detection.bbox))

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """Build the ``[1, 3, H, W]`` float32 tensor in the ``0..255`` range."""
        face = self._ensure_bgr(face_bgr)
        if (face.shape[1], face.shape[0]) != self._input_size:
            face = cv2.resize(face, self._input_size, interpolation=cv2.INTER_LINEAR)
        # Do not divide by 255.  Upstream Silent-Face ToTensor intentionally
        # feeds raw 0..255 pixels to MiniFASNet.
        tensor = face.astype(np.float32)
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        if not np.isfinite(tensor).all():
            raise AntiSpoofingError("Face crop contains non-finite pixel values.")
        return np.ascontiguousarray(tensor)

    def predict_crop(self, face_bgr: np.ndarray) -> PadPrediction:
        tensor = self.preprocess(face_bgr)
        with self._lock:
            if self._session is None:
                raise AntiSpoofingError("Anti-spoofing model is not loaded.")
            try:
                outputs = self._session.run(
                    [self._output_name], {self._input_name: tensor}
                )
            except Exception as exc:
                raise AntiSpoofingError(
                    f"Anti-spoofing inference failed: {exc}"
                ) from exc

        logits = np.asarray(outputs[0], dtype=np.float32)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise AntiSpoofingError(
                f"Unexpected MiniFASNetV2 output shape: {logits.shape}."
            )
        if logits.shape[1] != self._class_count:
            raise AntiSpoofingError(
                f"Expected {self._class_count} classes, got {logits.shape[1]}."
            )
        if not np.isfinite(logits).all():
            raise AntiSpoofingError("MiniFASNetV2 returned non-finite logits.")

        probabilities = self._softmax(logits)[0]
        live_index = settings.PAD_CLASS_LIVE_INDEX
        live_score = float(probabilities[live_index])
        spoof_score = float(
            max(
                probabilities[index]
                for index in range(self._class_count)
                if index != live_index
            )
        )
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])
        return PadPrediction(
            class_scores=tuple(float(value) for value in probabilities),
            live_score=live_score,
            spoof_score=spoof_score,
            label="LIVE" if class_index == live_index else "SPOOF",
            class_index=class_index,
            confidence=confidence,
        )

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise AntiSpoofingError("Face image is empty.")
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.ndim != 3 or image.shape[2] != 3:
            raise AntiSpoofingError("Face image must be an 8-bit BGR image.")
        return image

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        return exponentiated / np.sum(exponentiated, axis=1, keepdims=True)


class PadTracker:
    """Turn per-frame PAD predictions into a stable authentication decision."""

    def __init__(self, config: PadConfig | None = None) -> None:
        self.config = config or PadConfig()
        self._live_streak = 0
        self._spoof_streak = 0
        self._verified = False
        self._spoof_detected = False
        self._last_face_key: tuple[int, int, int, int] | None = None

    @property
    def live_streak(self) -> int:
        return self._live_streak

    @property
    def verified(self) -> bool:
        return self._verified

    def reset(self, _reason: str | None = None) -> None:
        self._live_streak = 0
        self._spoof_streak = 0
        self._verified = False
        self._spoof_detected = False
        self._last_face_key = None

    def face_changed(self, face_key: tuple[int, int, int, int]) -> bool:
        if self._last_face_key is None:
            return False
        return (
            _bbox_iou(self._last_face_key, face_key) < self.config.face_iou_threshold
        )

    def update(
        self,
        prediction: PadPrediction,
        face_key: tuple[int, int, int, int],
    ) -> PadAssessment:
        if self.face_changed(face_key):
            self.reset("face changed")
        self._last_face_key = tuple(int(value) for value in face_key)

        if prediction.live_score >= self.config.live_threshold:
            self._live_streak += 1
            self._spoof_streak = 0
            self._spoof_detected = False
            if self._live_streak >= self.config.required_live_frames:
                self._verified = True
                status = PAD_LIVE
                message = (
                    f"Live face confirmed ({self._live_streak}/"
                    f"{self.config.required_live_frames} frames)."
                )
            else:
                self._verified = False
                status = PAD_ANALYZING
                message = (
                    f"Checking liveness ({self._live_streak}/"
                    f"{self.config.required_live_frames} live frames)."
                )
        elif prediction.spoof_score >= self.config.spoof_threshold:
            self._live_streak = 0
            self._spoof_streak += 1
            self._verified = False
            self._spoof_detected = True
            status = PAD_SPOOF
            message = "Presentation attack detected. Authentication blocked."
        else:
            self._live_streak = 0
            self._spoof_streak = 0
            self._verified = False
            self._spoof_detected = False
            status = PAD_UNCERTAIN
            message = (
                "Liveness confidence is too low. Hold still and improve lighting."
            )

        return PadAssessment(
            status=status,
            verified=self._verified,
            spoof_detected=self._spoof_detected,
            live_score=prediction.live_score,
            confidence=prediction.confidence,
            live_streak=self._live_streak,
            required_live_frames=self.config.required_live_frames,
            message=message,
            prediction=prediction,
        )

    def update_no_face(self) -> PadAssessment:
        self.reset("no face")
        return self._assessment(
            PAD_NO_FACE,
            "No face detected. Authentication is blocked.",
        )

    def update_loading(self) -> PadAssessment:
        self.reset("loading")
        return self._assessment(
            PAD_LOADING,
            "Anti-spoofing model is loading.",
        )

    def update_error(self, message: str) -> PadAssessment:
        self.reset("error")
        return self._assessment(
            PAD_ERROR,
            message or "Anti-spoofing inference failed.",
        )

    def _assessment(self, status: str, message: str) -> PadAssessment:
        return PadAssessment(
            status=status,
            verified=False,
            spoof_detected=False,
            live_score=None,
            confidence=None,
            live_streak=0,
            required_live_frames=self.config.required_live_frames,
            message=message,
        )


@dataclass(frozen=True)
class PadSnapshot:
    enabled: bool
    detector: AntiSpoofingDetector | None
    config: PadConfig
    error: str | None
    loading: bool


class PadRuntime:
    """Thread-safe hand-off between the GUI toggle and the camera worker."""

    def __init__(self, config: PadConfig | None = None) -> None:
        self._lock = threading.RLock()
        self._config = config or PadConfig()
        self._detector: AntiSpoofingDetector | None = None
        self._enabled = False
        self._error: str | None = None
        self._loading = False

    @property
    def config(self) -> PadConfig:
        with self._lock:
            return self._config

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def enable(self) -> None:
        with self._lock:
            self._enabled = True

    def disable(self) -> None:
        with self._lock:
            self._enabled = False

    def set_loading(self, loading: bool) -> None:
        with self._lock:
            self._loading = bool(loading)
            if loading:
                self._error = None

    def set_detector(self, detector: AntiSpoofingDetector) -> None:
        with self._lock:
            self._detector = detector
            self._error = None
            self._loading = False

    def set_error(self, message: str) -> None:
        with self._lock:
            self._error = str(message) if message else "Unknown anti-spoofing error."
            self._loading = False

    def snapshot(self) -> PadSnapshot:
        with self._lock:
            return PadSnapshot(
                enabled=self._enabled,
                detector=self._detector,
                config=self._config,
                error=self._error,
                loading=self._loading,
            )

    def close(self) -> None:
        with self._lock:
            detector = self._detector
            self._detector = None
            self._enabled = False
            self._error = None
            self._loading = False
        if detector is not None:
            detector.close()


class AntiSpoofingService:
    """Lazily load the optional PAD model and expose its runtime state."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        config: PadConfig | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else settings.PAD_MODEL
        self.runtime = PadRuntime(config)
        self._load_lock = threading.Lock()
        self._closed = False

    @property
    def enabled(self) -> bool:
        return self.runtime.enabled

    def enable(self) -> None:
        if self._closed:
            raise AntiSpoofingError("Anti-spoofing service is closed.")
        self.runtime.enable()

    def disable(self) -> None:
        self.runtime.disable()

    def ensure_loaded(self) -> AntiSpoofingDetector:
        with self._load_lock:
            if self._closed:
                raise AntiSpoofingError("Anti-spoofing service is closed.")
            snapshot = self.runtime.snapshot()
            if snapshot.detector is not None:
                return snapshot.detector
            self.runtime.set_loading(True)
            try:
                detector = AntiSpoofingDetector(
                    self.model_path,
                    live_threshold=snapshot.config.live_threshold,
                    spoof_threshold=snapshot.config.spoof_threshold,
                )
            except Exception as exc:
                message = str(exc)
                self.runtime.set_error(message)
                raise AntiSpoofingError(message) from exc
            if self._closed:
                detector.close()
                raise AntiSpoofingError("Anti-spoofing service is closed.")
            self.runtime.set_detector(detector)
            return detector

    def close(self) -> None:
        with self._load_lock:
            self._closed = True
            self.runtime.close()
