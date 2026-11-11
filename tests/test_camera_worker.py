from __future__ import annotations

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
