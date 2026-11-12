"""Background webcam worker used by the live, verification and capture pages."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import settings
from core.active_liveness import (
    LIVENESS_FAILED,
    LIVENESS_NOT_RUN,
    LIVENESS_PASSED,
    LIVENESS_PENDING,
    ActiveLivenessService,
    LivenessAssessment,
)
from core.anti_spoofing import (
    PAD_ANALYZING,
    PAD_DISABLED,
    PAD_ERROR,
    PAD_LIVE,
    PAD_LOADING,
    PAD_NO_FACE,
    PAD_NOT_RUN,
    PAD_SPOOF,
    PAD_UNCERTAIN,
    AntiSpoofingError,
    PadAssessment,
    PadPrediction,
    PadRuntime,
    PadSnapshot,
    PadTracker,
)
from core.camera_manager import CameraError, CameraManager
from core.diagnostics import PerformanceMonitor, PerformanceSnapshot
from core.face_engine import FaceDetection, FaceEngine, FaceEngineError
from core.frame_buffer import LatestFrameBuffer
from core.frame_source import (
    SOURCE_UNKNOWN,
    FrameSource,
    FrameSourceEnded,
    FrameSourceError,
    PhysicalCameraSource,
)
from core.image_processor import draw_faces, draw_pad_status
from core.matcher import EnrolledIdentity, Identification, identify
from core.protection import (
    FINAL_ACCEPTED,
    FINAL_NOT_RUN,
    FINAL_REJECTED,
    LAYER_ACTIVE_LIVENESS,
    LAYER_IDENTITY,
    LAYER_NONE,
    LAYER_PASSIVE_PAD,
    LAYER_SOURCE_INTEGRITY,
    ProtectionConfig,
)
from core.source_integrity import (
    SOURCE_NOT_RUN,
    SOURCE_REJECTED,
    SOURCE_TRUSTED,
    CameraSourceIntegrityService,
    SourceAssessment,
    SourceSession,
)


def _bbox_iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
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


@dataclass
class _FeatureCacheEntry:
    bbox: tuple[int, int, int, int]
    feature: np.ndarray
    frame_number: int


@dataclass
class LiveFrameResult:
    raw_frame: np.ndarray | None = None
    annotated_frame: np.ndarray | None = None
    detections: list[FaceDetection] = field(default_factory=list)
    identifications: list[Identification] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    similarity: float | None = None
    verified: bool | None = None
    user_id: str | None = None
    fps: float = 0.0
    error: str | None = None
    finished: bool = False
    pad_enabled: bool = False
    pad_status: str = PAD_DISABLED
    pad_live_score: float | None = None
    pad_confidence: float | None = None
    pad_message: str = ""
    pad_verified: bool = False
    pad_spoof: bool = False
    pad_live_streak: int = 0
    pad_required_frames: int = 0
    source_kind: str = SOURCE_UNKNOWN
    source_label: str = "Unknown"
    source_identity: str = ""
    source_integrity_status: str = SOURCE_NOT_RUN
    source_integrity_reason: str = ""
    source_integrity_ran: bool = False
    source_changed: bool = False
    liveness_status: str = LIVENESS_NOT_RUN
    liveness_challenge: str | None = None
    liveness_challenge_label: str = ""
    liveness_message: str = ""
    liveness_progress: int = 0
    liveness_required_frames: int = 0
    liveness_remaining_seconds: float | None = None
    liveness_ran: bool = False
    passive_pad_ran: bool = False
    identity_executed: bool = False
    protection_mode: str = "Original / Vulnerable"
    final_decision: str = FINAL_NOT_RUN
    decision_reason: str = ""
    layer_detected: str = LAYER_NONE
    processing_ms: float = 0.0
    diagnostics: PerformanceSnapshot | None = None

    @property
    def recognized_count(self) -> int:
        return sum(1 for identification in self.identifications if identification.matched)

    @property
    def source_status_label(self) -> str:
        return {
            "NOT_RUN": "Not Run",
            "TRUSTED": "Trusted",
            "REJECTED": "Rejected",
            "UNCERTAIN": "Uncertain",
        }.get(self.source_integrity_status, self.source_integrity_status)

    @property
    def liveness_status_label(self) -> str:
        return {
            "NOT_RUN": "Not Run",
            "PENDING": "Pending",
            "PASSED": "Passed",
            "FAILED": "Failed",
            "EXPIRED": "Expired",
            "TRACKING_LOST": "Tracking Lost",
            "FACE_CHANGED": "Face Changed",
            "ERROR": "Error",
        }.get(self.liveness_status, self.liveness_status)


class LiveCameraWorker(threading.Thread):
    """Read frames, run YuNet/SFace and publish the latest result.

    When a PAD runtime is supplied and enabled, the largest face is checked for
    liveness before SFace identity work is allowed.  The Tkinter main thread
    polls ``result_queue``; no Tk widget is touched from this thread.
    """

    def __init__(
        self,
        engine: FaceEngine,
        camera_index: int,
        template: np.ndarray | None = None,
        user_id: str | None = None,
        threshold: float = 0.363,
        camera_manager: CameraManager | None = None,
        identities: list[EnrolledIdentity] | None = None,
        pad_runtime: PadRuntime | None = None,
        detection_interval: int | None = None,
        recognition_interval: int | None = None,
        frame_source: FrameSource | None = None,
        source_integrity: CameraSourceIntegrityService | None = None,
        source_session: SourceSession | None = None,
        liveness: ActiveLivenessService | None = None,
        protection: ProtectionConfig | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.engine = engine
        self.camera_index = int(camera_index)
        self.template = template
        self.user_id = user_id
        self.threshold = float(threshold)
        self.identities = list(identities or [])
        self.camera = camera_manager or (
            None if frame_source is not None else CameraManager()
        )
        self.source = frame_source or PhysicalCameraSource(
            self.camera or CameraManager(), self.camera_index
        )
        self.source_integrity = source_integrity
        self.source_session = source_session
        self.liveness = liveness
        self.protection = protection or ProtectionConfig()
        self.pad_runtime = pad_runtime
        self.detection_interval = max(
            1,
            int(
                detection_interval
                if detection_interval is not None
                else settings.DETECTION_INTERVAL
            ),
        )
        requested_recognition = max(
            1,
            int(
                recognition_interval
                if recognition_interval is not None
                else settings.RECOGNITION_INTERVAL
            ),
        )
        # SFace should not run more often than detections refresh; otherwise it
        # could align a current frame with landmarks from an older frame.
        self.recognition_interval = max(
            requested_recognition, self.detection_interval
        )
        self.feature_cache_iou = settings.FEATURE_CACHE_IOU
        self.result_queue: queue.Queue[LiveFrameResult] = queue.Queue(
            maxsize=settings.RESULT_QUEUE_SIZE
        )
        self._frames = LatestFrameBuffer()
        self._capture_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._fps = 0.0
        self._process_fps = 0.0
        self._capture_last_time = 0.0
        self._process_last_time = 0.0
        self._frame_counter = 0
        self._detection_valid = False
        self._cached_detections: list[FaceDetection] = []
        self._feature_cache: list[_FeatureCacheEntry] = []
        self._monitor = PerformanceMonitor()
        self._pad_tracker = PadTracker(
            pad_runtime.config if pad_runtime is not None else None
        )
        self._pad_frame_counter = 0
        self._last_pad_prediction: PadPrediction | None = None
        self._pad_was_enabled = False
        self._pad_detector_id: int | None = None
        self._last_source_assessment: SourceAssessment | None = None
        self._last_liveness_assessment: LivenessAssessment | None = None
        self._last_pad_assessment: PadAssessment | None = None
        self._identity_executed = False
        self._passive_pad_ran = False

    @property
    def stopping(self) -> bool:
        return self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()
        self._frames.close()
        self.source.release()

    def _publish(self, result: LiveFrameResult) -> None:
        try:
            self.result_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.result_queue.put_nowait(result)
        except queue.Full:
            pass

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self.source.read()
            except FrameSourceEnded as exc:
                if not self._stop_event.is_set():
                    self._publish(
                        LiveFrameResult(error=str(exc), final_decision=FINAL_REJECTED)
                    )
                self._stop_event.set()
                self._frames.close()
                return
            except (CameraError, FrameSourceError) as exc:
                if not self._stop_event.is_set():
                    self._publish(LiveFrameResult(error=str(exc)))
                self._stop_event.set()
                self._frames.close()
                return
            except Exception as exc:
                if not self._stop_event.is_set():
                    self._publish(
                        LiveFrameResult(error=f"Camera capture error: {exc}")
                    )
                self._stop_event.set()
                self._frames.close()
                return

            now = time.perf_counter()
            if self._capture_last_time:
                delta = max(1e-6, now - self._capture_last_time)
                instant_fps = 1.0 / delta
                self._fps = (
                    instant_fps
                    if self._fps == 0.0
                    else 0.85 * self._fps + 0.15 * instant_fps
                )
            self._capture_last_time = now
            self._frames.put(frame)

    def _detections_for(self, frame: np.ndarray) -> list[FaceDetection]:
        if (
            not self._detection_valid
            or self._frame_counter % self.detection_interval == 0
        ):
            started = time.perf_counter()
            self._cached_detections = self.engine.detect(frame)
            self._monitor.record_detect(
                (time.perf_counter() - started) * 1000.0
            )
            self._detection_valid = True
            if not self._cached_detections:
                self._feature_cache.clear()
        self._frame_counter += 1
        return self._cached_detections

    def _feature_for(
        self, frame: np.ndarray, detection: FaceDetection
    ) -> np.ndarray:
        best: _FeatureCacheEntry | None = None
        best_iou = 0.0
        for entry in self._feature_cache:
            iou = _bbox_iou(entry.bbox, detection.bbox)
            if iou > best_iou:
                best = entry
                best_iou = iou
        if (
            best is not None
            and best_iou >= self.feature_cache_iou
            and self._frame_counter - best.frame_number < self.recognition_interval
        ):
            return best.feature

        started = time.perf_counter()
        feature = self.engine.extract_feature(frame, detection)
        self._monitor.record_sface((time.perf_counter() - started) * 1000.0)
        self._feature_cache = [
            entry
            for entry in self._feature_cache
            if _bbox_iou(entry.bbox, detection.bbox) < self.feature_cache_iou
            and self._frame_counter - entry.frame_number
            < self.recognition_interval
        ]
        self._feature_cache.append(
            _FeatureCacheEntry(detection.bbox, feature, self._frame_counter)
        )
        return feature

    def _process_loop(self) -> None:
        last_sequence = 0
        while not self._stop_event.is_set():
            item = self._frames.get(last_sequence, timeout=0.1)
            if item is None:
                continue
            frame, last_sequence = item
            now = time.perf_counter()
            if self._process_last_time:
                delta = max(1e-6, now - self._process_last_time)
                instant_fps = 1.0 / delta
                self._process_fps = (
                    instant_fps
                    if self._process_fps == 0.0
                    else 0.85 * self._process_fps + 0.15 * instant_fps
                )
            self._process_last_time = now
            started = now

            detections = self._detections_for(frame)
            snapshot = (
                self.pad_runtime.snapshot()
                if self.pad_runtime is not None
                else None
            )
            pad_active = snapshot is not None and snapshot.enabled
            detector_id = (
                id(snapshot.detector)
                if snapshot is not None and snapshot.detector is not None
                else None
            )
            if (
                pad_active != self._pad_was_enabled
                or detector_id != self._pad_detector_id
            ):
                self._pad_tracker.reset("PAD runtime changed")
                self._last_pad_prediction = None
                self._feature_cache.clear()
                self._pad_was_enabled = pad_active
                self._pad_detector_id = detector_id

            source_assessment = self._evaluate_source()
            if (
                self.protection.source_integrity_enabled
                and not source_assessment.trusted
            ):
                liveness_assessment = self._not_run_liveness(
                    "Active liveness was not reached because source integrity failed."
                )
                result = self._blocked_source_result(
                    frame, detections, source_assessment, snapshot
                )
            else:
                liveness_assessment = self._evaluate_liveness(
                    detections, frame.shape
                )
                if (
                    self.protection.active_liveness_enabled
                    and not liveness_assessment.passed
                ):
                    result = self._blocked_liveness_result(
                        frame, detections, liveness_assessment, snapshot
                    )
                elif snapshot is not None and snapshot.enabled:
                    result = self._analyze_protected(frame, detections, snapshot)
                else:
                    result = self._analyze_unprotected(frame, detections)

            self._apply_protection_context(
                result, source_assessment, liveness_assessment
            )
            self._finalize_result(result)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._monitor.record_process(elapsed_ms)
            result.processing_ms = elapsed_ms
            result.diagnostics = self._monitor.snapshot(
                self._fps, self._process_fps, self._frames.dropped
            )
            self._publish(result)

    def run(self) -> None:
        capture_thread: threading.Thread | None = None
        try:
            self.source.open()
            capture_thread = threading.Thread(
                target=self._capture_loop,
                name="camera-capture",
                daemon=True,
            )
            self._capture_thread = capture_thread
            capture_thread.start()
            self._process_loop()
        except (CameraError, FrameSourceError) as exc:
            if not self._stop_event.is_set():
                self._publish(LiveFrameResult(error=str(exc)))
        except FaceEngineError as exc:
            if not self._stop_event.is_set():
                self._publish(LiveFrameResult(error=f"Face engine error: {exc}"))
        except Exception as exc:  # defensive: never leave a silent dead camera
            if not self._stop_event.is_set():
                self._publish(LiveFrameResult(error=f"Camera worker error: {exc}"))
        finally:
            self._stop_event.set()
            self._frames.close()
            if capture_thread is not None and capture_thread.is_alive():
                capture_thread.join(timeout=1.0)
            self.source.release()
            self._publish(LiveFrameResult(finished=True))

    def _evaluate_source(self) -> SourceAssessment:
        info = self.source.info
        if self.source_integrity is None:
            if self.protection.source_integrity_enabled:
                return SourceAssessment(
                    SOURCE_REJECTED,
                    False,
                    info.kind,
                    info.kind_label,
                    "Source integrity is enabled but the service is unavailable.",
                    info.identifier,
                )
            return SourceAssessment(
                SOURCE_NOT_RUN,
                False,
                info.kind,
                info.kind_label,
                "Source integrity protection is off. This source is not trusted.",
                info.identifier,
                policy_applied=False,
            )
        return self.source_integrity.evaluate(
            info,
            self.source_session,
            enabled=self.protection.source_integrity_enabled,
        )

    def _not_run_liveness(self, message: str) -> LivenessAssessment:
        return LivenessAssessment(
            LIVENESS_NOT_RUN,
            False,
            False,
            None,
            "Not issued",
            None,
            None,
            None,
            0,
            settings.LIVENESS_REQUIRED_FRAMES,
            message=message,
        )

    def _evaluate_liveness(
        self,
        detections: list[FaceDetection],
        frame_shape: tuple[int, ...],
    ) -> LivenessAssessment:
        if not self.protection.active_liveness_enabled:
            return LivenessAssessment(
                LIVENESS_NOT_RUN,
                False,
                False,
                None,
                "Not issued",
                None,
                None,
                None,
                0,
                settings.LIVENESS_REQUIRED_FRAMES,
                message="Active liveness protection is off.",
            )
        if self.liveness is None:
            return LivenessAssessment(
                LIVENESS_FAILED,
                False,
                True,
                None,
                "Not issued",
                None,
                None,
                None,
                0,
                settings.LIVENESS_REQUIRED_FRAMES,
                message="Active liveness is enabled but the service is unavailable.",
            )
        return self.liveness.observe(detections, frame_shape)

    def _blocked_source_result(
        self,
        frame: np.ndarray,
        detections: list[FaceDetection],
        assessment: SourceAssessment,
        snapshot: PadSnapshot | None,
    ) -> LiveFrameResult:
        message = f"SOURCE REJECTED: {assessment.reason}"
        return self._gate_result(
            frame,
            detections,
            message,
            PAD_ERROR,
            snapshot,
            final_decision=FINAL_REJECTED,
            decision_reason=assessment.reason,
            layer=LAYER_SOURCE_INTEGRITY,
        )

    def _blocked_liveness_result(
        self,
        frame: np.ndarray,
        detections: list[FaceDetection],
        assessment: LivenessAssessment,
        snapshot: PadSnapshot | None,
    ) -> LiveFrameResult:
        pending = assessment.status in {LIVENESS_PENDING, LIVENESS_NOT_RUN}
        final = FINAL_NOT_RUN if pending else FINAL_REJECTED
        status = PAD_ANALYZING if pending else PAD_ERROR
        message = f"ACTIVE LIVENESS: {assessment.message}"
        return self._gate_result(
            frame,
            detections,
            message,
            status,
            snapshot,
            final_decision=final,
            decision_reason=assessment.message,
            layer=LAYER_ACTIVE_LIVENESS,
        )

    def _gate_result(
        self,
        frame: np.ndarray,
        detections: list[FaceDetection],
        message: str,
        draw_status: str,
        snapshot: PadSnapshot | None,
        final_decision: str,
        decision_reason: str,
        layer: str,
    ) -> LiveFrameResult:
        self._feature_cache.clear()
        primary = self.engine.largest_face(detections)
        identifications = [
            Identification(None, None, None, False) for _ in detections
        ]
        labels = ["" for _ in detections]
        if primary is not None:
            labels[primary.index] = message.split(":", 1)[0]
        annotated = draw_faces(
            frame,
            detections,
            selected_index=primary.index if primary is not None else None,
            labels=labels,
        )
        annotated = draw_pad_status(annotated, message, draw_status)
        pad_enabled = snapshot is not None and snapshot.enabled
        return LiveFrameResult(
            raw_frame=frame,
            annotated_frame=annotated,
            detections=detections,
            identifications=identifications,
            labels=labels,
            similarity=None,
            verified=(
                False
                if final_decision == FINAL_REJECTED and self.template is not None
                else None
            ),
            user_id=self.user_id,
            fps=self._fps,
            pad_enabled=pad_enabled,
            pad_status=PAD_NOT_RUN,
            pad_message="Passive PAD was not reached." if pad_enabled else "Protection is off.",
            pad_verified=False,
            pad_spoof=False,
            final_decision=final_decision,
            decision_reason=decision_reason,
            layer_detected=layer,
        )

    def _apply_protection_context(
        self,
        result: LiveFrameResult,
        source: SourceAssessment,
        liveness: LivenessAssessment,
    ) -> None:
        result.source_kind = source.kind
        result.source_label = source.kind_label
        result.source_identity = source.identity
        result.source_integrity_status = source.status
        result.source_integrity_reason = source.reason
        result.source_integrity_ran = (
            self.protection.source_integrity_enabled
            and self.source_integrity is not None
        )
        result.source_changed = source.source_changed
        result.liveness_status = liveness.status
        result.liveness_challenge = liveness.challenge
        result.liveness_challenge_label = liveness.challenge_label
        result.liveness_message = liveness.message
        result.liveness_progress = liveness.progress
        result.liveness_required_frames = liveness.required_frames
        result.liveness_remaining_seconds = liveness.remaining_seconds
        result.liveness_ran = (
            self.protection.active_liveness_enabled
            and self.liveness is not None
            and not (
                self.protection.source_integrity_enabled
                and not source.trusted
            )
        )
        result.protection_mode = self.protection.label

    def _finalize_result(self, result: LiveFrameResult) -> None:
        if result.error is not None:
            result.final_decision = FINAL_REJECTED
            result.decision_reason = result.error
            result.layer_detected = result.layer_detected or LAYER_NONE
            return
        if result.final_decision != FINAL_NOT_RUN:
            return
        if result.source_integrity_ran and result.source_integrity_status != SOURCE_TRUSTED:
            result.final_decision = FINAL_REJECTED
            result.decision_reason = result.source_integrity_reason
            result.layer_detected = LAYER_SOURCE_INTEGRITY
            return
        if result.liveness_ran and result.liveness_status != LIVENESS_PASSED:
            if result.liveness_status in {LIVENESS_PENDING, LIVENESS_NOT_RUN}:
                result.final_decision = FINAL_NOT_RUN
            else:
                result.final_decision = FINAL_REJECTED
            result.decision_reason = result.liveness_message
            result.layer_detected = LAYER_ACTIVE_LIVENESS
            return
        if result.pad_enabled and result.pad_status != PAD_LIVE:
            if result.pad_status in {PAD_ANALYZING, PAD_LOADING, PAD_NO_FACE}:
                result.final_decision = FINAL_NOT_RUN
            else:
                result.final_decision = FINAL_REJECTED
            result.decision_reason = result.pad_message
            result.layer_detected = LAYER_PASSIVE_PAD
            return
        if result.identity_executed:
            identified = result.verified is True or (
                self.template is None and result.recognized_count > 0
            )
            if identified:
                result.final_decision = FINAL_ACCEPTED
                result.decision_reason = "Face identity matched the enrolled template."
                result.layer_detected = LAYER_NONE
            else:
                result.final_decision = FINAL_REJECTED
                result.decision_reason = "Face identity did not match the enrolled template."
                result.layer_detected = LAYER_IDENTITY
            return
        if result.detections and (
            self.template is not None or self.identities
        ):
            result.final_decision = FINAL_REJECTED
            result.decision_reason = "Identity verification could not be completed."
            result.layer_detected = LAYER_IDENTITY

    def _analyze_unprotected(
        self, frame: np.ndarray, detections: list[FaceDetection]
    ) -> LiveFrameResult:
        """Original behaviour: no PAD overhead and no PAD decision."""
        face: FaceDetection | None = self.engine.largest_face(detections)
        features: dict[int, np.ndarray] = {}

        if self.identities:
            for detection in detections:
                try:
                    features[detection.index] = self._feature_for(frame, detection)
                except FaceEngineError:
                    continue
        elif face is not None and self.template is not None:
            try:
                features[face.index] = self._feature_for(frame, face)
            except FaceEngineError:
                pass

        identifications: list[Identification] = []
        labels: list[str] = []
        for detection in detections:
            probe = features.get(detection.index)
            if self.identities and probe is not None:
                identification = identify(
                    self.engine, probe, self.identities, self.threshold
                )
            else:
                identification = Identification(None, None, None, False)
            identifications.append(identification)
            if identification.matched:
                labels.append(
                    f"{identification.display_name}  "
                    f"{identification.similarity:.2f}"
                )
            elif identification.similarity is not None:
                labels.append(f"Unknown  {identification.similarity:.2f}")
            else:
                labels.append("")

        similarity: float | None = None
        verified: bool | None = None
        if face is not None and self.template is not None:
            probe = features.get(face.index)
            if probe is None:
                try:
                    probe = self._feature_for(frame, face)
                except FaceEngineError:
                    probe = None
            if probe is not None:
                try:
                    similarity = self.engine.cosine_similarity(probe, self.template)
                    verified = similarity >= self.threshold
                except FaceEngineError:
                    similarity = None
                    verified = None

        recognized = sum(
            1 for identification in identifications if identification.matched
        )
        if not detections:
            status = "No face detected"
        elif similarity is None:
            if self.identities:
                status = (
                    f"{len(detections)} face(s) detected, "
                    f"{recognized} recognized"
                )
            else:
                status = f"{len(detections)} face(s) detected"
        else:
            status = (
                f"{'VERIFIED' if verified else 'NOT VERIFIED'}  "
                f"cosine {similarity:.3f} / {self.threshold:.3f}"
            )

        annotated = draw_faces(
            frame,
            detections,
            selected_index=face.index if face is not None else None,
            status_text=status,
            labels=labels,
        )
        if similarity is not None:
            colour = (40, 200, 120) if verified else (70, 90, 255)
            cv2.putText(
                annotated,
                f"{similarity:.3f}",
                (12, annotated.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                colour,
                2,
                cv2.LINE_AA,
            )

        return LiveFrameResult(
            raw_frame=frame,
            annotated_frame=annotated,
            detections=detections,
            identifications=identifications,
            labels=labels,
            similarity=similarity,
            verified=verified,
            user_id=self.user_id,
            fps=self._fps,
            pad_enabled=False,
            pad_status=PAD_DISABLED,
            pad_message="Protection is off.",
            passive_pad_ran=False,
            identity_executed=bool(features),
        )

    def _analyze_protected(
        self,
        frame: np.ndarray,
        detections: list[FaceDetection],
        snapshot: PadSnapshot,
    ) -> LiveFrameResult:
        primary = self.engine.largest_face(detections)
        config = snapshot.config

        if snapshot.detector is None:
            if snapshot.error:
                assessment = self._pad_tracker.update_error(snapshot.error)
            else:
                assessment = self._pad_tracker.update_loading()
            return self._blocked_pad_result(frame, detections, primary, assessment)

        if primary is None:
            assessment = self._pad_tracker.update_no_face()
            return self._blocked_pad_result(frame, detections, None, assessment)

        if self._pad_tracker.face_changed(primary.bbox):
            self._pad_tracker.reset("face changed")
            self._last_pad_prediction = None
            self._feature_cache.clear()

        should_infer = (
            self._last_pad_prediction is None
            or (self._pad_frame_counter % config.frame_interval == 0)
        )
        self._pad_frame_counter += 1

        if should_infer:
            started = time.perf_counter()
            try:
                prediction = snapshot.detector.predict(frame, primary)
            except AntiSpoofingError as exc:
                self._last_pad_prediction = None
                assessment = self._pad_tracker.update_error(str(exc))
                return self._blocked_pad_result(
                    frame, detections, primary, assessment
                )
            self._last_pad_prediction = prediction
            self._monitor.record_pad((time.perf_counter() - started) * 1000.0)
        else:
            prediction = self._last_pad_prediction
            if prediction is None:
                assessment = self._pad_tracker.update_error(
                    "PAD inference did not produce a result."
                )
                return self._blocked_pad_result(
                    frame, detections, primary, assessment
                )

        assessment = self._pad_tracker.update(prediction, primary.bbox)
        if not assessment.verified:
            return self._blocked_pad_result(frame, detections, primary, assessment)
        return self._live_pad_result(frame, detections, primary, assessment)

    def _blocked_pad_result(
        self,
        frame: np.ndarray,
        detections: list[FaceDetection],
        primary: FaceDetection | None,
        assessment: PadAssessment,
    ) -> LiveFrameResult:
        self._feature_cache.clear()
        identifications = [
            Identification(None, None, None, False) for _ in detections
        ]
        labels = ["" for _ in detections]
        if primary is not None:
            labels[primary.index] = self._pad_label(assessment.status)
            if len(detections) > 1:
                for detection in detections:
                    if detection.index != primary.index:
                        labels[detection.index] = "PAD: primary only"

        annotated = draw_faces(
            frame,
            detections,
            selected_index=primary.index if primary is not None else None,
            labels=labels,
        )
        annotated = draw_pad_status(annotated, assessment.message, assessment.status)
        return LiveFrameResult(
            raw_frame=frame,
            annotated_frame=annotated,
            detections=detections,
            identifications=identifications,
            labels=labels,
            similarity=None,
            verified=False if self.template is not None else None,
            user_id=self.user_id,
            fps=self._fps,
            pad_enabled=True,
            pad_status=assessment.status,
            pad_live_score=assessment.live_score,
            pad_confidence=assessment.confidence,
            pad_message=assessment.message,
            pad_verified=False,
            pad_spoof=assessment.spoof_detected,
            pad_live_streak=assessment.live_streak,
            pad_required_frames=assessment.required_live_frames,
            passive_pad_ran=True,
            identity_executed=False,
        )

    def _live_pad_result(
        self,
        frame: np.ndarray,
        detections: list[FaceDetection],
        primary: FaceDetection,
        assessment: PadAssessment,
    ) -> LiveFrameResult:
        identifications = [
            Identification(None, None, None, False) for _ in detections
        ]
        labels = ["" for _ in detections]

        feature: np.ndarray | None = None
        if self.identities or self.template is not None:
            try:
                feature = self._feature_for(frame, primary)
            except FaceEngineError:
                feature = None

        identification = Identification(None, None, None, False)
        if feature is not None and self.identities:
            identification = identify(
                self.engine, feature, self.identities, self.threshold
            )
            identifications[primary.index] = identification
        if identification.matched:
            labels[primary.index] = (
                f"{identification.display_name}  "
                f"{identification.similarity:.2f}"
            )
        elif identification.similarity is not None:
            labels[primary.index] = f"Unknown  {identification.similarity:.2f}"
        else:
            labels[primary.index] = "LIVE"

        if len(detections) > 1:
            for detection in detections:
                if detection.index != primary.index:
                    labels[detection.index] = "PAD: primary only"

        similarity: float | None = None
        verified: bool | None = None
        if feature is not None and self.template is not None:
            try:
                similarity = self.engine.cosine_similarity(feature, self.template)
                verified = similarity >= self.threshold
            except FaceEngineError:
                similarity = None
                verified = None

        if similarity is not None:
            status_text = (
                f"PAD: LIVE | {'VERIFIED' if verified else 'NOT VERIFIED'} "
                f"cosine {similarity:.3f}"
            )
        elif feature is None and (self.identities or self.template is not None):
            status_text = "PAD: LIVE | IDENTITY CHECK FAILED"
        else:
            status_text = "PAD: LIVE FACE"
        if identification.matched:
            status_text += (
                f" | {identification.display_name} "
                f"{identification.similarity:.2f}"
            )

        annotated = draw_faces(
            frame,
            detections,
            selected_index=primary.index,
            labels=labels,
        )
        annotated = draw_pad_status(annotated, status_text, PAD_LIVE)
        if similarity is not None:
            colour = (40, 200, 120) if verified else (70, 90, 255)
            cv2.putText(
                annotated,
                f"{similarity:.3f}",
                (12, annotated.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                colour,
                2,
                cv2.LINE_AA,
            )

        return LiveFrameResult(
            raw_frame=frame,
            annotated_frame=annotated,
            detections=detections,
            identifications=identifications,
            labels=labels,
            similarity=similarity,
            verified=verified,
            user_id=self.user_id,
            fps=self._fps,
            pad_enabled=True,
            pad_status=PAD_LIVE,
            pad_live_score=assessment.live_score,
            pad_confidence=assessment.confidence,
            pad_message=assessment.message,
            pad_verified=True,
            pad_spoof=False,
            pad_live_streak=assessment.live_streak,
            pad_required_frames=assessment.required_live_frames,
            passive_pad_ran=True,
            identity_executed=(
                feature is not None
                and bool(self.identities or self.template is not None)
            ),
        )

    @staticmethod
    def _pad_label(status: str) -> str:
        return {
            PAD_SPOOF: "SPOOF",
            PAD_ANALYZING: "PAD CHECK",
            PAD_UNCERTAIN: "LOW CONF",
            PAD_ERROR: "PAD ERROR",
            PAD_LOADING: "PAD LOADING",
            PAD_NO_FACE: "",
            PAD_LIVE: "LIVE",
        }.get(status, "")
