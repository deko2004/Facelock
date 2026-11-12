"""Randomized active-liveness challenge-response using YuNet landmarks.

The service asks for a physical head movement, records the challenge time, and
verifies a movement relative to a baseline captured *after* the challenge was
issued.  It does not inspect arbitrary pixel differences and it does not claim
to detect blinks.  Head pose is estimated from the five YuNet landmarks:

* right eye, left eye, nose, right mouth corner, left mouth corner.

The normalized yaw is the nose offset from the eye midpoint divided by the
inter-eye distance.  The normalized pitch is the nose vertical offset divided by
the same distance.  This is a landmark heuristic, not metric 3-D head pose.
"""

from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from config import settings
from core.face_engine import FaceDetection

LIVENESS_NOT_RUN = "NOT_RUN"
LIVENESS_PENDING = "PENDING"
LIVENESS_PASSED = "PASSED"
LIVENESS_FAILED = "FAILED"
LIVENESS_EXPIRED = "EXPIRED"
LIVENESS_TRACKING_LOST = "TRACKING_LOST"
LIVENESS_FACE_CHANGED = "FACE_CHANGED"
LIVENESS_ERROR = "ERROR"

LIVENESS_STATUS_LABELS = {
    LIVENESS_NOT_RUN: "Not Run",
    LIVENESS_PENDING: "Pending",
    LIVENESS_PASSED: "Passed",
    LIVENESS_FAILED: "Failed",
    LIVENESS_EXPIRED: "Expired",
    LIVENESS_TRACKING_LOST: "Tracking Lost",
    LIVENESS_FACE_CHANGED: "Face Changed",
    LIVENESS_ERROR: "Error",
}

CHALLENGE_TURN_LEFT = "TURN_HEAD_LEFT"
CHALLENGE_TURN_RIGHT = "TURN_HEAD_RIGHT"
CHALLENGE_LOOK_UP = "LOOK_UP"
CHALLENGE_LOOK_DOWN = "LOOK_DOWN"

CHALLENGES = (
    CHALLENGE_TURN_LEFT,
    CHALLENGE_TURN_RIGHT,
    CHALLENGE_LOOK_UP,
    CHALLENGE_LOOK_DOWN,
)

CHALLENGE_LABELS = {
    CHALLENGE_TURN_LEFT: "Turn your head to your left",
    CHALLENGE_TURN_RIGHT: "Turn your head to your right",
    CHALLENGE_LOOK_UP: "Look up",
    CHALLENGE_LOOK_DOWN: "Look down",
}

CHALLENGE_SHORT_LABELS = {
    CHALLENGE_TURN_LEFT: "Head left",
    CHALLENGE_TURN_RIGHT: "Head right",
    CHALLENGE_LOOK_UP: "Look up",
    CHALLENGE_LOOK_DOWN: "Look down",
}


class ActiveLivenessError(RuntimeError):
    """Raised when a challenge cannot be issued or evaluated."""


@dataclass(frozen=True)
class HeadPose:
    yaw: float
    pitch: float
    confidence: float
    valid: bool
    reason: str = ""


@dataclass(frozen=True)
class LivenessAssessment:
    status: str
    passed: bool
    failed: bool
    challenge: str | None
    challenge_label: str
    issued_at: float | None
    deadline: float | None
    remaining_seconds: float | None
    progress: int
    required_frames: int
    yaw: float | None = None
    pitch: float | None = None
    baseline_yaw: float | None = None
    baseline_pitch: float | None = None
    face_confidence: float | None = None
    message: str = ""
    expired: bool = False
    tracking_lost: bool = False
    face_changed: bool = False
    details: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def status_label(self) -> str:
        return LIVENESS_STATUS_LABELS.get(self.status, self.status)


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


def estimate_head_pose(
    detection: FaceDetection,
    min_confidence: float = settings.LIVENESS_MIN_CONFIDENCE,
) -> HeadPose:
    """Estimate normalized yaw/pitch from the five YuNet landmarks."""
    raw = np.asarray(detection.raw, dtype=np.float32).ravel()
    if raw.size < 15:
        return HeadPose(0.0, 0.0, 0.0, False, "Landmark vector is incomplete.")
    landmarks = raw[4:14].astype(np.float64)
    if not np.isfinite(landmarks).all():
        return HeadPose(0.0, 0.0, 0.0, False, "Landmarks contain non-finite values.")
    confidence = float(detection.score)
    if not math.isfinite(confidence) or confidence < float(min_confidence):
        return HeadPose(
            0.0,
            0.0,
            confidence if math.isfinite(confidence) else 0.0,
            False,
            f"Face confidence {confidence:.3f} is below {min_confidence:.3f}.",
        )

    right_eye = landmarks[0:2]
    left_eye = landmarks[2:4]
    nose = landmarks[4:6]
    right_mouth = landmarks[6:8]
    left_mouth = landmarks[8:10]

    eye_mid = (right_eye + left_eye) / 2.0
    eye_distance = float(np.linalg.norm(left_eye - right_eye))
    if eye_distance < 2.0:
        return HeadPose(0.0, 0.0, confidence, False, "Inter-eye distance is too small.")
    mouth_mid = (right_mouth + left_mouth) / 2.0
    if float(np.linalg.norm(mouth_mid - eye_mid)) < 2.0:
        return HeadPose(0.0, 0.0, confidence, False, "Facial landmarks are degenerate.")

    yaw = float((nose[0] - eye_mid[0]) / eye_distance)
    pitch = float((nose[1] - eye_mid[1]) / eye_distance)
    if not math.isfinite(yaw) or not math.isfinite(pitch):
        return HeadPose(0.0, 0.0, confidence, False, "Head pose is non-finite.")
    return HeadPose(yaw, pitch, confidence, True, "")


class ActiveLivenessService:
    """Thread-safe randomized challenge-response state machine."""

    def __init__(
        self,
        timeout: float = settings.LIVENESS_TIMEOUT,
        required_frames: int = settings.LIVENESS_REQUIRED_FRAMES,
        yaw_threshold: float = settings.LIVENESS_YAW_THRESHOLD,
        pitch_threshold: float = settings.LIVENESS_PITCH_THRESHOLD,
        min_confidence: float = settings.LIVENESS_MIN_CONFIDENCE,
        face_iou_threshold: float = settings.LIVENESS_FACE_IOU_THRESHOLD,
        audit: Callable[[str, str | None, str], None] | None = None,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.timeout = max(1.0, float(timeout))
        self.required_frames = max(1, int(required_frames))
        self.yaw_threshold = max(0.01, float(yaw_threshold))
        self.pitch_threshold = max(0.01, float(pitch_threshold))
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.face_iou_threshold = max(0.0, min(1.0, float(face_iou_threshold)))
        self._audit_callback = audit
        self._rng = rng or random.SystemRandom()
        self._clock = clock
        self._lock = threading.RLock()
        self._reset_state()

    def _reset_state(self) -> None:
        self._status = LIVENESS_NOT_RUN
        self._challenge: str | None = None
        self._issued_at: float | None = None
        self._deadline: float | None = None
        self._baseline_yaw: float | None = None
        self._baseline_pitch: float | None = None
        self._face_key: tuple[int, int, int, int] | None = None
        self._progress = 0
        self._message = "Active liveness has not been started."
        self._last_yaw: float | None = None
        self._last_pitch: float | None = None
        self._last_confidence: float | None = None
        self._expired = False
        self._tracking_lost = False
        self._face_changed = False
        self._last_audited_status = ""

    def reset(self, reason: str = "reset") -> None:
        with self._lock:
            self._reset_state()
            self._message = reason
            self._status = LIVENESS_NOT_RUN

    def start_session(self, now: float | None = None) -> LivenessAssessment:
        return self.issue_challenge(now=now)

    def issue_challenge(
        self,
        challenge: str | None = None,
        now: float | None = None,
    ) -> LivenessAssessment:
        timestamp = self._clock() if now is None else float(now)
        with self._lock:
            selected = challenge or self._rng.choice(CHALLENGES)
            if selected not in CHALLENGES:
                raise ActiveLivenessError(f"Unknown liveness challenge: {selected}")
            self._status = LIVENESS_PENDING
            self._challenge = selected
            self._issued_at = timestamp
            self._deadline = timestamp + self.timeout
            self._baseline_yaw = None
            self._baseline_pitch = None
            self._face_key = None
            self._progress = 0
            self._message = (
                f"Challenge issued: {CHALLENGE_LABELS[selected]}. "
                "Keep your face visible and follow the instruction."
            )
            self._last_yaw = None
            self._last_pitch = None
            self._last_confidence = None
            self._expired = False
            self._tracking_lost = False
            self._face_changed = False
            self._last_audited_status = LIVENESS_PENDING
            self._audit(
                "liveness_challenge_issued",
                None,
                f"challenge={selected}; issued_at={timestamp:.3f}; "
                f"timeout={self.timeout:.1f}s",
            )
            return self._assessment(timestamp)

    def observe(
        self,
        detections: Sequence[FaceDetection],
        frame_shape: tuple[int, ...] | None = None,
        now: float | None = None,
    ) -> LivenessAssessment:
        """Evaluate one processed frame against the active challenge."""
        timestamp = self._clock() if now is None else float(now)
        with self._lock:
            if self._challenge is None or self._status == LIVENESS_NOT_RUN:
                return self._assessment(timestamp)
            if self._status in {
                LIVENESS_PASSED,
                LIVENESS_FAILED,
                LIVENESS_EXPIRED,
                LIVENESS_TRACKING_LOST,
                LIVENESS_FACE_CHANGED,
                LIVENESS_ERROR,
            }:
                return self._assessment(timestamp)

            if self._deadline is not None and timestamp > self._deadline:
                return self._fail(
                    LIVENESS_EXPIRED,
                    "The liveness challenge expired before the movement was verified.",
                    expired=True,
                    now=timestamp,
                )

            if not detections:
                return self._fail(
                    LIVENESS_TRACKING_LOST,
                    "Face tracking was lost; the challenge must be restarted.",
                    tracking_lost=True,
                    now=timestamp,
                )
            if len(detections) != 1:
                return self._fail(
                    LIVENESS_FAILED,
                    "Multiple faces detected; the challenge cannot be verified safely.",
                    now=timestamp,
                )

            detection = detections[0]
            pose = estimate_head_pose(detection, self.min_confidence)
            self._last_yaw = pose.yaw
            self._last_pitch = pose.pitch
            self._last_confidence = pose.confidence
            if not pose.valid:
                return self._fail(
                    LIVENESS_FAILED,
                    f"Head pose is not reliable: {pose.reason}",
                    now=timestamp,
                )

            face_key = tuple(int(value) for value in detection.bbox)
            if self._face_key is None:
                self._face_key = face_key
            elif (
                _bbox_iou(self._face_key, face_key) < self.face_iou_threshold
            ):
                self._face_key = face_key
                return self._fail(
                    LIVENESS_FACE_CHANGED,
                    "The tracked face changed; issue a new liveness challenge.",
                    face_changed=True,
                    now=timestamp,
                )

            if self._baseline_yaw is None or self._baseline_pitch is None:
                self._baseline_yaw = pose.yaw
                self._baseline_pitch = pose.pitch
                self._progress = 0
                self._message = (
                    f"Baseline recorded. Now {CHALLENGE_LABELS[self._challenge].lower()}."
                )
                return self._assessment(timestamp)

            moved = self._movement_satisfied(pose)
            if moved:
                self._progress += 1
                if self._progress >= self.required_frames:
                    self._status = LIVENESS_PASSED
                    self._message = (
                        f"Liveness challenge passed ({self._progress}/"
                        f"{self.required_frames} consistent frames)."
                    )
                    self._audit(
                        "liveness_challenge_passed",
                        None,
                        f"challenge={self._challenge}; frames={self._progress}; "
                        f"yaw={pose.yaw:.3f}; pitch={pose.pitch:.3f}",
                    )
                else:
                    self._message = (
                        f"Movement detected ({self._progress}/"
                        f"{self.required_frames}). Hold the movement briefly."
                    )
            else:
                self._progress = 0
                self._message = (
                    f"Waiting for movement: {CHALLENGE_LABELS[self._challenge]}."
                )
            return self._assessment(timestamp)

    def _movement_satisfied(self, pose: HeadPose) -> bool:
        assert self._challenge is not None
        assert self._baseline_yaw is not None
        assert self._baseline_pitch is not None
        delta_yaw = pose.yaw - self._baseline_yaw
        delta_pitch = pose.pitch - self._baseline_pitch
        if self._challenge == CHALLENGE_TURN_LEFT:
            return delta_yaw >= self.yaw_threshold
        if self._challenge == CHALLENGE_TURN_RIGHT:
            return delta_yaw <= -self.yaw_threshold
        if self._challenge == CHALLENGE_LOOK_UP:
            return delta_pitch <= -self.pitch_threshold
        if self._challenge == CHALLENGE_LOOK_DOWN:
            return delta_pitch >= self.pitch_threshold
        return False

    def _fail(
        self,
        status: str,
        message: str,
        now: float,
        expired: bool = False,
        tracking_lost: bool = False,
        face_changed: bool = False,
    ) -> LivenessAssessment:
        self._status = status
        self._message = message
        self._progress = 0
        self._expired = expired
        self._tracking_lost = tracking_lost
        self._face_changed = face_changed
        if self._last_audited_status != status:
            self._last_audited_status = status
            self._audit(
                "liveness_challenge_failed",
                None,
                f"challenge={self._challenge}; status={status}; reason={message}",
            )
        return self._assessment(now)

    def _assessment(self, now: float) -> LivenessAssessment:
        remaining = None
        if self._deadline is not None:
            remaining = max(0.0, self._deadline - now)
        challenge = self._challenge
        return LivenessAssessment(
            status=self._status,
            passed=self._status == LIVENESS_PASSED,
            failed=self._status
            in {
                LIVENESS_FAILED,
                LIVENESS_EXPIRED,
                LIVENESS_TRACKING_LOST,
                LIVENESS_FACE_CHANGED,
                LIVENESS_ERROR,
            },
            challenge=challenge,
            challenge_label=CHALLENGE_LABELS.get(challenge or "", "Not issued"),
            issued_at=self._issued_at,
            deadline=self._deadline,
            remaining_seconds=remaining,
            progress=self._progress,
            required_frames=self.required_frames,
            yaw=self._last_yaw,
            pitch=self._last_pitch,
            baseline_yaw=self._baseline_yaw,
            baseline_pitch=self._baseline_pitch,
            face_confidence=self._last_confidence,
            message=self._message,
            expired=self._expired,
            tracking_lost=self._tracking_lost,
            face_changed=self._face_changed,
            details=(
                ("challenge", challenge or "none"),
                ("issued_at", f"{self._issued_at:.3f}" if self._issued_at else "none"),
                ("timeout_seconds", f"{self.timeout:.1f}"),
            ),
        )

    def snapshot(self, now: float | None = None) -> LivenessAssessment:
        timestamp = self._clock() if now is None else float(now)
        with self._lock:
            return self._assessment(timestamp)

    def close(self) -> None:
        with self._lock:
            self._reset_state()

    def _audit(
        self, event_type: str, user_id: str | None, detail: str
    ) -> None:
        if self._audit_callback is None:
            return
        try:
            self._audit_callback(event_type, user_id, detail)
        except Exception:
            pass
