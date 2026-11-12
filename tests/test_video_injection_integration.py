from __future__ import annotations

import queue
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.active_liveness import CHALLENGE_TURN_LEFT, LIVENESS_PASSED
from core.face_engine import FaceDetection, FaceEngine
from core.frame_source import SOURCE_RECORDED, RecordedVideoSource
from core.protection import (
    FINAL_ACCEPTED,
    FINAL_NOT_RUN,
    FINAL_REJECTED,
    LAYER_ACTIVE_LIVENESS,
    LAYER_NONE,
    LAYER_SOURCE_INTEGRITY,
    ProtectionConfig,
)
from core.source_integrity import (
    SOURCE_NOT_RUN,
    SOURCE_REJECTED,
    CameraSourceIntegrityService,
)
from gui.camera_worker import LiveCameraWorker


def _write_video(path: Path, frames: int = 8) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        20.0,
        (64, 48),
    )
    if not writer.isOpened():
        pytest.skip("MJPG video writer is unavailable in this OpenCV build")
    try:
        for _ in range(frames):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    finally:
        writer.release()


def _detection(
    index: int = 0,
    nose_x: float = 50.0,
    bbox: tuple[int, int, int, int] = (10, 10, 30, 30),
) -> FaceDetection:
    raw = np.zeros(15, dtype=np.float32)
    x, y, w, h = bbox
    raw[:4] = (x, y, w, h)
    raw[4:6] = (40.0, 40.0)
    raw[6:8] = (60.0, 40.0)
    raw[8:10] = (nose_x, 55.0)
    raw[10:12] = (42.0, 70.0)
    raw[12:14] = (58.0, 70.0)
    return FaceDetection(index, x, y, w, h, 0.99, raw)


class SequenceEngine:
    cosine_similarity = staticmethod(FaceEngine.cosine_similarity)

    def __init__(self, detections: list[list[FaceDetection]], feature: np.ndarray):
        self.detections = detections
        self.feature = feature
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        batch = self.detections[min(self.calls, len(self.detections) - 1)]
        self.calls += 1
        return batch

    def largest_face(self, detections: list[FaceDetection]) -> FaceDetection | None:
        return detections[0] if detections else None

    def extract_feature(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        return self.feature


def _collect_until(worker: LiveCameraWorker, predicate, timeout: float = 3.0):
    deadline = time.time() + timeout
    latest = None
    while time.time() < deadline:
        try:
            result = worker.result_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if result.error:
            raise AssertionError(result.error)
        if result.finished:
            continue
        latest = result
        if predicate(result):
            return result
    return latest


def _worker(
    tmp_path: Path,
    engine: SequenceEngine,
    *,
    protection: ProtectionConfig,
    source_integrity=None,
    session=None,
    template: np.ndarray | None = None,
    liveness=None,
) -> LiveCameraWorker:
    path = tmp_path / "replay.avi"
    _write_video(path)
    source = RecordedVideoSource(path, loop=True, pace=False)
    return LiveCameraWorker(
        engine=engine,
        camera_index=0,
        template=template,
        threshold=0.5,
        frame_source=source,
        source_integrity=source_integrity,
        source_session=session,
        liveness=liveness,
        protection=protection,
        detection_interval=1,
        recognition_interval=1,
    )


def _stop(worker: LiveCameraWorker) -> None:
    worker.stop()
    worker.join(timeout=2.0)


def test_recorded_video_is_accepted_when_protections_are_off(tmp_path: Path) -> None:
    feature = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    engine = SequenceEngine([[_detection()]], feature)
    service = CameraSourceIntegrityService()
    worker = _worker(
        tmp_path,
        engine,
        protection=ProtectionConfig(),
        source_integrity=service,
        session=None,
        template=feature,
    )
    worker.start()
    try:
        result = _collect_until(worker, lambda item: item.final_decision == FINAL_ACCEPTED)
    finally:
        _stop(worker)
    assert result is not None
    assert result.final_decision == FINAL_ACCEPTED
    assert result.layer_detected == LAYER_NONE
    assert result.identity_executed
    assert result.source_kind == SOURCE_RECORDED
    assert result.source_integrity_status == SOURCE_NOT_RUN
    assert not result.passive_pad_ran


def test_source_integrity_blocks_recorded_video_before_identity(tmp_path: Path) -> None:
    feature = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    engine = SequenceEngine([[_detection()]], feature)
    service = CameraSourceIntegrityService()
    info = RecordedVideoSource(tmp_path / "replay.avi").info
    session = service.start_session(info)
    worker = _worker(
        tmp_path,
        engine,
        protection=ProtectionConfig(
            source_integrity_enabled=True,
            label="Full Protection",
        ),
        source_integrity=service,
        session=session,
        template=feature,
    )
    worker.start()
    try:
        result = _collect_until(worker, lambda item: item.final_decision == FINAL_REJECTED)
    finally:
        _stop(worker)
    assert result is not None
    assert result.final_decision == FINAL_REJECTED
    assert result.layer_detected == LAYER_SOURCE_INTEGRITY
    assert result.source_integrity_status == SOURCE_REJECTED
    assert not result.identity_executed
    assert not result.passive_pad_ran
    assert not result.liveness_ran


def test_active_liveness_blocks_multiple_faces_before_identity(tmp_path: Path) -> None:
    feature = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    engine = SequenceEngine(
        [[_detection(0), _detection(1, bbox=(90, 10, 30, 30))]],
        feature,
    )
    service = CameraSourceIntegrityService()
    from core.active_liveness import ActiveLivenessService

    liveness = ActiveLivenessService()
    liveness.issue_challenge(CHALLENGE_TURN_LEFT)
    worker = _worker(
        tmp_path,
        engine,
        protection=ProtectionConfig(
            active_liveness_enabled=True,
            label="Liveness Simulation",
            simulation=True,
        ),
        source_integrity=service,
        template=feature,
        liveness=liveness,
    )
    worker.start()
    try:
        result = _collect_until(worker, lambda item: item.final_decision == FINAL_REJECTED)
    finally:
        _stop(worker)
    assert result is not None
    assert result.layer_detected == LAYER_ACTIVE_LIVENESS
    assert result.liveness_status == "FAILED"
    assert result.liveness_ran
    assert not result.identity_executed


def test_pending_liveness_keeps_identity_blocked(tmp_path: Path) -> None:
    feature = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    engine = SequenceEngine([[_detection()]], feature)
    from core.active_liveness import ActiveLivenessService

    liveness = ActiveLivenessService(timeout=60.0)
    liveness.issue_challenge(CHALLENGE_TURN_LEFT)
    worker = _worker(
        tmp_path,
        engine,
        protection=ProtectionConfig(
            active_liveness_enabled=True,
            label="Liveness Simulation",
            simulation=True,
        ),
        template=feature,
        liveness=liveness,
    )
    worker.start()
    try:
        result = _collect_until(worker, lambda item: item.liveness_ran)
    finally:
        _stop(worker)
    assert result is not None
    assert result.final_decision == FINAL_NOT_RUN
    assert result.layer_detected == LAYER_ACTIVE_LIVENESS
    assert result.liveness_status == "PENDING"
    assert not result.identity_executed


def test_passed_liveness_allows_identity_pipeline(tmp_path: Path) -> None:
    feature = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    neutral = _detection()
    turned = _detection(nose_x=58.0)
    engine = SequenceEngine([[neutral], [turned]], feature)
    from core.active_liveness import ActiveLivenessService

    liveness = ActiveLivenessService(timeout=60.0, required_frames=3)
    liveness.issue_challenge(CHALLENGE_TURN_LEFT)
    worker = _worker(
        tmp_path,
        engine,
        protection=ProtectionConfig(
            active_liveness_enabled=True,
            label="Liveness Simulation",
            simulation=True,
        ),
        template=feature,
        liveness=liveness,
    )
    worker.start()
    try:
        result = _collect_until(
            worker,
            lambda item: item.final_decision == FINAL_ACCEPTED,
            timeout=5.0,
        )
    finally:
        _stop(worker)
    assert result is not None
    assert result.final_decision == FINAL_ACCEPTED
    assert result.liveness_status == LIVENESS_PASSED
    assert result.identity_executed
