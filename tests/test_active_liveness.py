from __future__ import annotations

import random

import numpy as np
import pytest

from core.active_liveness import (
    CHALLENGE_LOOK_UP,
    CHALLENGE_TURN_LEFT,
    LIVENESS_EXPIRED,
    LIVENESS_FACE_CHANGED,
    LIVENESS_FAILED,
    LIVENESS_NOT_RUN,
    LIVENESS_PASSED,
    LIVENESS_PENDING,
    LIVENESS_TRACKING_LOST,
    ActiveLivenessError,
    ActiveLivenessService,
    estimate_head_pose,
)
from core.face_engine import FaceDetection


def _detection(
    nose_x: float = 50.0,
    nose_y: float = 55.0,
    bbox: tuple[int, int, int, int] = (30, 30, 40, 50),
    score: float = 0.99,
) -> FaceDetection:
    raw = np.zeros(15, dtype=np.float32)
    x, y, w, h = bbox
    raw[:4] = (x, y, w, h)
    raw[4:6] = (40.0, 40.0)
    raw[6:8] = (60.0, 40.0)
    raw[8:10] = (nose_x, nose_y)
    raw[10:12] = (42.0, 70.0)
    raw[12:14] = (58.0, 70.0)
    return FaceDetection(0, x, y, w, h, score, raw)


def _service(**kwargs) -> ActiveLivenessService:
    return ActiveLivenessService(rng=random.Random(7), **kwargs)


def test_no_challenge_is_not_run() -> None:
    service = _service()
    assessment = service.observe([_detection()], (100, 100, 3), now=1.0)
    assert assessment.status == LIVENESS_NOT_RUN
    assert not assessment.passed
    assert not assessment.failed


def test_challenge_uses_baseline_after_issuance_and_requires_temporal_movement() -> None:
    service = _service(required_frames=3, yaw_threshold=0.18)
    issued = service.issue_challenge(CHALLENGE_TURN_LEFT, now=0.0)
    assert issued.status == LIVENESS_PENDING
    assert issued.challenge == CHALLENGE_TURN_LEFT

    baseline = service.observe([_detection()], (100, 100, 3), now=0.1)
    assert baseline.status == LIVENESS_PENDING
    assert baseline.progress == 0
    assert baseline.baseline_yaw == pytest.approx(0.0)

    moved = _detection(nose_x=58.0)
    first = service.observe([moved], (100, 100, 3), now=0.2)
    second = service.observe([moved], (100, 100, 3), now=0.3)
    third = service.observe([moved], (100, 100, 3), now=0.4)
    assert first.progress == 1
    assert second.progress == 2
    assert third.status == LIVENESS_PASSED
    assert third.passed


def test_wrong_movement_does_not_accumulate_evidence() -> None:
    service = _service(required_frames=3)
    service.issue_challenge(CHALLENGE_TURN_LEFT, now=0.0)
    service.observe([_detection()], (100, 100, 3), now=0.1)
    wrong = service.observe([_detection(nose_x=42.0)], (100, 100, 3), now=0.2)
    assert wrong.progress == 0
    assert wrong.status == LIVENESS_PENDING


def test_vertical_challenge_uses_pitch_delta() -> None:
    service = _service(required_frames=2, pitch_threshold=0.12)
    service.issue_challenge(CHALLENGE_LOOK_UP, now=0.0)
    service.observe([_detection()], (100, 100, 3), now=0.1)
    up = _detection(nose_y=50.0)
    service.observe([up], (100, 100, 3), now=0.2)
    passed = service.observe([up], (100, 100, 3), now=0.3)
    assert passed.status == LIVENESS_PASSED


def test_timeout_expires_challenge() -> None:
    service = _service(timeout=2.0)
    service.issue_challenge(CHALLENGE_TURN_LEFT, now=0.0)
    assessment = service.observe([_detection()], (100, 100, 3), now=3.0)
    assert assessment.status == LIVENESS_EXPIRED
    assert assessment.expired
    assert assessment.failed


def test_missing_face_and_multiple_faces_fail_closed() -> None:
    service = _service()
    service.issue_challenge(CHALLENGE_TURN_LEFT, now=0.0)
    missing = service.observe([], (100, 100, 3), now=0.1)
    assert missing.status == LIVENESS_TRACKING_LOST
    assert missing.tracking_lost

    service.issue_challenge(CHALLENGE_TURN_LEFT, now=1.0)
    multiple = service.observe(
        [_detection(), _detection(bbox=(90, 30, 40, 50))],
        (100, 100, 3),
        now=1.1,
    )
    assert multiple.status == LIVENESS_FAILED


def test_low_confidence_and_incomplete_landmarks_fail() -> None:
    low = _detection(score=0.2)
    pose = estimate_head_pose(low, min_confidence=0.8)
    assert not pose.valid

    incomplete = FaceDetection(
        0, 30, 30, 40, 50, 0.99, np.zeros(10, dtype=np.float32)
    )
    assert not estimate_head_pose(incomplete).valid


def test_face_change_is_detected() -> None:
    service = _service()
    service.issue_challenge(CHALLENGE_TURN_LEFT, now=0.0)
    service.observe([_detection()], (100, 100, 3), now=0.1)
    changed = service.observe(
        [_detection(bbox=(200, 200, 40, 50))], (100, 100, 3), now=0.2
    )
    assert changed.status == LIVENESS_FACE_CHANGED
    assert changed.face_changed


def test_new_challenge_resets_a_failed_session() -> None:
    service = _service()
    service.issue_challenge(CHALLENGE_TURN_LEFT, now=0.0)
    service.observe([], (100, 100, 3), now=0.1)
    assert service.snapshot(now=0.2).failed

    issued = service.issue_challenge(CHALLENGE_TURN_LEFT, now=1.0)
    assert issued.status == LIVENESS_PENDING
    assert issued.progress == 0
    assert issued.challenge == CHALLENGE_TURN_LEFT


def test_unknown_challenge_is_rejected() -> None:
    service = _service()
    with pytest.raises(ActiveLivenessError):
        service.issue_challenge("NOT_A_CHALLENGE", now=0.0)
