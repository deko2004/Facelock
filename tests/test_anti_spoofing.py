from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from config import settings
from core.anti_spoofing import (
    PAD_ANALYZING,
    PAD_ERROR,
    PAD_LIVE,
    PAD_NO_FACE,
    PAD_SPOOF,
    PAD_UNCERTAIN,
    AntiSpoofingDetector,
    AntiSpoofingError,
    AntiSpoofingService,
    PadConfig,
    PadPrediction,
    PadRuntime,
    PadTracker,
)
from core.face_engine import FaceDetection, FaceEngine
from core.matcher import EnrolledIdentity
from gui.camera_worker import LiveCameraWorker

ASSETS = Path(__file__).resolve().parent / "assets" / "anti_spoofing"


def _pad_available() -> bool:
    return (
        settings.PAD_MODEL.is_file()
        and importlib.util.find_spec("onnxruntime") is not None
    )


def _require_pad() -> None:
    if not _pad_available():
        pytest.skip("MiniFASNetV2 model or onnxruntime is not available")


@pytest.fixture(scope="module")
def detector() -> AntiSpoofingDetector:
    _require_pad()
    instance = AntiSpoofingDetector()
    try:
        yield instance
    finally:
        instance.close()


def _face(image_path: Path, engine: FaceEngine) -> tuple[np.ndarray, FaceDetection]:
    image = cv2.imread(str(image_path))
    assert image is not None, f"Could not read {image_path}"
    detections = engine.detect(image)
    assert detections, f"No face detected in {image_path.name}"
    face = engine.largest_face(detections)
    assert face is not None
    return image, face


def _live_prediction(score: float = 0.95) -> PadPrediction:
    other = (1.0 - score) / 2.0
    return PadPrediction(
        class_scores=(other, score, other),
        live_score=score,
        spoof_score=other,
        label="LIVE",
        class_index=1,
        confidence=score,
    )


def _spoof_prediction(score: float = 0.95) -> PadPrediction:
    other = (1.0 - score) / 2.0
    return PadPrediction(
        class_scores=(other, other, score),
        live_score=other,
        spoof_score=score,
        label="SPOOF",
        class_index=2,
        confidence=score,
    )


def _screen_replay(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    small = cv2.resize(
        image, (160, max(1, int(height * 160 / width))), interpolation=cv2.INTER_AREA
    )
    upscaled = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(
        upscaled, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(20, 20, 20)
    )


def test_model_contract(detector: AntiSpoofingDetector) -> None:
    assert detector.loaded
    assert detector.input_size == (80, 80)
    assert detector.class_count == 3


def test_preprocessing_keeps_raw_0_to_255_range(
    detector: AntiSpoofingDetector,
) -> None:
    face = np.zeros((80, 80, 3), dtype=np.uint8)
    face[:, :, 0] = 255
    face[:, :, 1] = 128
    face[:, :, 2] = 64

    tensor = detector.preprocess(face)

    assert tensor.shape == (1, 3, 80, 80)
    assert tensor.dtype == np.float32
    assert tensor.max() > 1.0
    assert tensor.min() >= 0.0
    assert tensor[0, 0, 0, 0] == pytest.approx(255.0)
    assert tensor[0, 1, 0, 0] == pytest.approx(128.0)
    assert tensor[0, 2, 0, 0] == pytest.approx(64.0)


def test_genuine_sample_is_live(
    detector: AntiSpoofingDetector, engine: FaceEngine
) -> None:
    image, face = _face(ASSETS / "image_T1.jpg", engine)
    prediction = detector.predict(image, face)

    assert prediction.label == "LIVE"
    assert prediction.live_score >= settings.PAD_LIVE_THRESHOLD
    assert prediction.spoof_score < settings.PAD_SPOOF_THRESHOLD


@pytest.mark.parametrize("name", ["image_F1.jpg", "image_F2.jpg"])
def test_official_attack_samples_are_rejected(
    detector: AntiSpoofingDetector, engine: FaceEngine, name: str
) -> None:
    image, face = _face(ASSETS / name, engine)
    prediction = detector.predict(image, face)

    assert prediction.label == "SPOOF"
    assert prediction.spoof_score >= settings.PAD_SPOOF_THRESHOLD
    assert prediction.live_score < settings.PAD_LIVE_THRESHOLD


def test_screen_replay_is_rejected(
    detector: AntiSpoofingDetector, engine: FaceEngine
) -> None:
    image, _ = _face(ASSETS / "image_T1.jpg", engine)
    replay = _screen_replay(image)
    detections = engine.detect(replay)
    assert detections
    face = engine.largest_face(detections)
    assert face is not None

    prediction = detector.predict(replay, face)

    assert prediction.spoof_score >= settings.PAD_SPOOF_THRESHOLD
    assert prediction.label == "SPOOF"


def test_tracker_requires_consecutive_live_frames() -> None:
    tracker = PadTracker(
        PadConfig(live_threshold=0.80, spoof_threshold=0.60, required_live_frames=3)
    )
    key = (10, 10, 50, 50)

    first = tracker.update(_live_prediction(), key)
    second = tracker.update(_live_prediction(), key)
    third = tracker.update(_live_prediction(), key)

    assert first.status == PAD_ANALYZING
    assert not first.verified
    assert second.status == PAD_ANALYZING
    assert not second.verified
    assert third.status == PAD_LIVE
    assert third.verified

    spoof = tracker.update(_spoof_prediction(), key)
    assert spoof.status == PAD_SPOOF
    assert spoof.spoof_detected
    assert not spoof.verified

    live_again = tracker.update(_live_prediction(), key)
    assert live_again.status == PAD_ANALYZING
    assert not live_again.verified


def test_tracker_resets_when_primary_face_changes() -> None:
    tracker = PadTracker(
        PadConfig(live_threshold=0.80, spoof_threshold=0.60, required_live_frames=2)
    )
    first_key = (10, 10, 50, 50)
    tracker.update(_live_prediction(), first_key)
    assert tracker.update(_live_prediction(), first_key).verified

    changed = tracker.update(_live_prediction(), (250, 250, 50, 50))
    assert changed.status == PAD_ANALYZING
    assert not changed.verified


def test_tracker_reports_uncertain_confidence() -> None:
    tracker = PadTracker(
        PadConfig(live_threshold=0.80, spoof_threshold=0.60, required_live_frames=2)
    )
    assessment = tracker.update(
        PadPrediction(
            class_scores=(0.15, 0.70, 0.15),
            live_score=0.70,
            spoof_score=0.15,
            label="LIVE",
            class_index=1,
            confidence=0.70,
        ),
        (10, 10, 50, 50),
    )
    assert assessment.status == PAD_UNCERTAIN
    assert not assessment.verified


def test_tracker_reports_no_face_and_errors() -> None:
    tracker = PadTracker()
    assert tracker.update_no_face().status == PAD_NO_FACE
    assert tracker.update_error("boom").status == PAD_ERROR


def test_service_reports_missing_model(tmp_path: Path) -> None:
    service = AntiSpoofingService(model_path=tmp_path / "missing.onnx")
    service.enable()
    with pytest.raises(AntiSpoofingError):
        service.ensure_loaded()
    snapshot = service.runtime.snapshot()
    assert snapshot.error
    assert snapshot.detector is None
    service.close()


def test_service_loads_and_closes() -> None:
    _require_pad()
    service = AntiSpoofingService()
    service.enable()
    detector = service.ensure_loaded()
    assert detector.loaded
    service.close()
    assert not service.enabled
    assert service.runtime.snapshot().detector is None


class FakeCamera:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.released = False

    def open(self, index: int) -> None:
        pass

    def read(self) -> np.ndarray:
        return self.frame.copy()

    def release(self) -> None:
        self.released = True


class CountingEngine:
    cosine_similarity = staticmethod(FaceEngine.cosine_similarity)

    def __init__(self, feature: np.ndarray, detections: list[FaceDetection] | None = None):
        self.feature = feature
        self.extract_calls = 0
        self.detections = detections or [
            FaceDetection(
                index=0,
                x=10,
                y=10,
                w=30,
                h=30,
                score=0.99,
                raw=np.zeros(15, dtype=np.float32),
            )
        ]

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        return list(self.detections)

    def largest_face(self, detections: list[FaceDetection]) -> FaceDetection | None:
        return detections[0] if detections else None

    def extract_feature(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        self.extract_calls += 1
        return self.feature


class FakePadDetector:
    def __init__(
        self,
        prediction: PadPrediction | None = None,
        error: Exception | None = None,
    ) -> None:
        self.prediction = prediction
        self.error = error
        self.calls = 0

    def predict(self, image: np.ndarray, detection: FaceDetection) -> PadPrediction:
        self.calls += 1
        if self.error is not None:
            raise self.error
        assert self.prediction is not None
        return self.prediction

    def close(self) -> None:
        pass


def _collect_until(worker: LiveCameraWorker, status: str, timeout: float = 2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = worker.result_queue.get(timeout=timeout)
        if result.pad_status == status:
            return result
    raise AssertionError(f"Did not receive PAD status {status}")


def test_worker_blocks_sface_when_spoofed() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    feature = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    engine = CountingEngine(feature)
    runtime = PadRuntime(PadConfig(required_live_frames=1))
    runtime.set_detector(FakePadDetector(_spoof_prediction()))
    runtime.enable()
    worker = LiveCameraWorker(
        engine=engine,
        camera_index=0,
        threshold=0.5,
        camera_manager=FakeCamera(frame),
        identities=[EnrolledIdentity("1", "Alice", feature)],
        pad_runtime=runtime,
    )

    worker.start()
    try:
        result = _collect_until(worker, PAD_SPOOF)
    finally:
        worker.stop()
        worker.join(timeout=2.0)
        runtime.close()

    assert result.pad_enabled
    assert result.pad_spoof
    assert not result.pad_verified
    assert result.similarity is None
    assert engine.extract_calls == 0


def test_worker_runs_sface_after_live_streak() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    feature = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    engine = CountingEngine(feature)
    runtime = PadRuntime(PadConfig(required_live_frames=2))
    runtime.set_detector(FakePadDetector(_live_prediction()))
    runtime.enable()
    worker = LiveCameraWorker(
        engine=engine,
        camera_index=0,
        threshold=0.5,
        camera_manager=FakeCamera(frame),
        identities=[EnrolledIdentity("1", "Alice", feature)],
        pad_runtime=runtime,
    )

    worker.start()
    try:
        result = _collect_until(worker, PAD_LIVE)
    finally:
        worker.stop()
        worker.join(timeout=2.0)
        runtime.close()

    assert result.pad_verified
    assert result.labels[0].startswith("Alice")
    assert engine.extract_calls > 0


def test_worker_blocks_sface_on_pad_error() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    feature = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    engine = CountingEngine(feature)
    runtime = PadRuntime(PadConfig(required_live_frames=1))
    runtime.set_detector(
        FakePadDetector(error=AntiSpoofingError("simulated PAD failure"))
    )
    runtime.enable()
    worker = LiveCameraWorker(
        engine=engine,
        camera_index=0,
        threshold=0.5,
        camera_manager=FakeCamera(frame),
        identities=[EnrolledIdentity("1", "Alice", feature)],
        pad_runtime=runtime,
    )

    worker.start()
    try:
        result = _collect_until(worker, PAD_ERROR)
    finally:
        worker.stop()
        worker.join(timeout=2.0)
        runtime.close()

    assert "simulated PAD failure" in result.pad_message
    assert not result.pad_verified
    assert engine.extract_calls == 0
