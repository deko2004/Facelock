from __future__ import annotations

import queue
import time

import numpy as np

from core.face_engine import FaceDetection, FaceEngine
from core.matcher import EnrolledIdentity
from gui.camera_worker import LiveCameraWorker


class FakeCamera:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.opened = False
        self.released = False

    def open(self, index: int) -> None:
        self.opened = True

    def read(self) -> np.ndarray:
        return self.frame.copy()

    def release(self) -> None:
        self.released = True


class FakeEngine:
    cosine_similarity = staticmethod(FaceEngine.cosine_similarity)

    def __init__(self, feature: np.ndarray) -> None:
        self.feature = feature
        self.detection = FaceDetection(
            index=0,
            x=10,
            y=10,
            w=30,
            h=30,
            score=0.99,
            raw=np.zeros(15, dtype=np.float32),
        )

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        return [self.detection]

    def largest_face(self, detections: list[FaceDetection]) -> FaceDetection | None:
        return detections[0] if detections else None

    def extract_feature(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        return self.feature


class CountingEngine(FakeEngine):
    def __init__(self, feature: np.ndarray) -> None:
        super().__init__(feature)
        self.detect_calls = 0
        self.extract_calls = 0

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        self.detect_calls += 1
        return super().detect(frame)

    def extract_feature(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        self.extract_calls += 1
        return super().extract_feature(frame, detection)


def _collect_results(
    worker: LiveCameraWorker, count: int, timeout: float = 2.0
) -> list:
    results = []
    deadline = time.time() + timeout
    while len(results) < count and time.time() < deadline:
        try:
            result = worker.result_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if not result.finished and not result.error:
            results.append(result)
    return results


def test_recognition_interval_is_not_shorter_than_detection() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    worker = LiveCameraWorker(
        engine=FakeEngine(np.array([[1.0, 0.0, 0.0]], dtype=np.float32)),
        camera_index=0,
        threshold=0.5,
        camera_manager=FakeCamera(frame),
        detection_interval=4,
        recognition_interval=1,
    )

    assert worker.detection_interval == 4
    assert worker.recognition_interval == 4


def test_worker_reuses_features_between_recognition_frames() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    feature = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    engine = CountingEngine(feature)
    worker = LiveCameraWorker(
        engine=engine,
        camera_index=0,
        threshold=0.5,
        camera_manager=FakeCamera(frame),
        identities=[EnrolledIdentity("1", "Alice", feature)],
        detection_interval=1,
        recognition_interval=4,
    )

    worker.start()
    try:
        results = _collect_results(worker, 10)
    finally:
        worker.stop()
        worker.join(timeout=2.0)

    assert len(results) == 10
    assert engine.detect_calls >= 10
    assert 1 <= engine.extract_calls < engine.detect_calls


def test_worker_reuses_detections_between_detection_frames() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    feature = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    engine = CountingEngine(feature)
    worker = LiveCameraWorker(
        engine=engine,
        camera_index=0,
        threshold=0.5,
        camera_manager=FakeCamera(frame),
        identities=[EnrolledIdentity("1", "Alice", feature)],
        detection_interval=3,
        recognition_interval=1,
    )

    worker.start()
    try:
        results = _collect_results(worker, 9)
    finally:
        worker.stop()
        worker.join(timeout=2.0)

    assert len(results) == 9
    assert 1 <= engine.detect_calls <= 4


def test_worker_releases_camera_and_capture_thread() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    camera = FakeCamera(frame)
    worker = LiveCameraWorker(
        engine=FakeEngine(np.array([[1.0, 0.0, 0.0]], dtype=np.float32)),
        camera_index=0,
        camera_manager=camera,
    )

    worker.start()
    try:
        result = worker.result_queue.get(timeout=2.0)
        assert not result.finished
    finally:
        worker.stop()
        worker.join(timeout=2.0)

    assert camera.released
    assert worker._capture_thread is not None
    assert not worker._capture_thread.is_alive()


def test_worker_labels_matching_enrolled_face() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    feature = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    identity = EnrolledIdentity("1", "Alice", feature)
    worker = LiveCameraWorker(
        engine=FakeEngine(feature),
        camera_index=0,
        threshold=0.5,
        camera_manager=FakeCamera(frame),
        identities=[identity],
    )

    worker.start()
    try:
        result = worker.result_queue.get(timeout=2.0)
    finally:
        worker.stop()
        worker.join(timeout=2.0)

    assert result.error is None
    assert result.labels == ["Alice  1.00"]
    assert result.recognized_count == 1
    assert result.identifications[0].display_name == "Alice"


def test_worker_labels_unknown_face_when_below_threshold() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    feature = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    probe = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    identity = EnrolledIdentity("1", "Alice", feature)
    worker = LiveCameraWorker(
        engine=FakeEngine(probe),
        camera_index=0,
        threshold=0.5,
        camera_manager=FakeCamera(frame),
        identities=[identity],
    )

    worker.start()
    try:
        result = worker.result_queue.get(timeout=2.0)
    finally:
        worker.stop()
        worker.join(timeout=2.0)

    assert result.error is None
    assert result.labels == ["Unknown  0.00"]
    assert result.recognized_count == 0
