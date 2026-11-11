"""Background webcam worker used by the live, verification and capture pages."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from core.anti_spoofing import (
    PAD_ANALYZING,
    PAD_DISABLED,
    PAD_ERROR,
    PAD_LIVE,
    PAD_LOADING,
    PAD_NO_FACE,
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
from core.face_engine import FaceDetection, FaceEngine, FaceEngineError
from core.image_processor import draw_faces, draw_pad_status
from core.matcher import EnrolledIdentity, Identification, identify


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

    @property
    def recognized_count(self) -> int:
        return sum(1 for identification in self.identifications if identification.matched)


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
    ) -> None:
        super().__init__(daemon=True)
        self.engine = engine
        self.camera_index = int(camera_index)
        self.template = template
        self.user_id = user_id
        self.threshold = float(threshold)
        self.identities = list(identities or [])
        self.camera = camera_manager or CameraManager()
        self.pad_runtime = pad_runtime
        self.result_queue: queue.Queue[LiveFrameResult] = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._fps = 0.0
        self._last_time = time.perf_counter()
        self._pad_tracker = PadTracker(
            pad_runtime.config if pad_runtime is not None else None
        )
        self._pad_frame_counter = 0
        self._last_pad_prediction: PadPrediction | None = None
        self._pad_was_enabled = False
        self._pad_detector_id: int | None = None

    @property
    def stopping(self) -> bool:
        return self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()
        self.camera.release()

    def _publish(self, result: LiveFrameResult) -> None:
        try:
            self.result_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.result_queue.put_nowait(result)
        except queue.Full:
            pass

    def run(self) -> None:
        try:
            self.camera.open(self.camera_index)
            while not self._stop_event.is_set():
                frame = self.camera.read()
                now = time.perf_counter()
                delta = max(1e-6, now - self._last_time)
                self._last_time = now
                instant_fps = 1.0 / delta
                self._fps = instant_fps if self._fps == 0.0 else (
                    0.85 * self._fps + 0.15 * instant_fps
                )

                detections = self.engine.detect(frame)
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
                    self._pad_was_enabled = pad_active
                    self._pad_detector_id = detector_id

                if snapshot is not None and snapshot.enabled:
                    result = self._analyze_protected(frame, detections, snapshot)
                else:
                    result = self._analyze_unprotected(frame, detections)
                self._publish(result)
        except CameraError as exc:
            if not self._stop_event.is_set():
                self._publish(LiveFrameResult(error=str(exc)))
        except FaceEngineError as exc:
            if not self._stop_event.is_set():
                self._publish(LiveFrameResult(error=f"Face engine error: {exc}"))
        except Exception as exc:  # defensive: never leave a silent dead camera
            if not self._stop_event.is_set():
                self._publish(LiveFrameResult(error=f"Camera worker error: {exc}"))
        finally:
            self.camera.release()
            self._publish(LiveFrameResult(finished=True))

    def _analyze_unprotected(
        self, frame: np.ndarray, detections: list[FaceDetection]
    ) -> LiveFrameResult:
        """Original behaviour: no PAD overhead and no PAD decision."""
        face: FaceDetection | None = self.engine.largest_face(detections)
        features: dict[int, np.ndarray] = {}

        if self.identities:
            for detection in detections:
                try:
                    features[detection.index] = self.engine.extract_feature(
                        frame, detection
                    )
                except FaceEngineError:
                    continue
        elif face is not None and self.template is not None:
            try:
                features[face.index] = self.engine.extract_feature(frame, face)
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
                    probe = self.engine.extract_feature(frame, face)
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

        should_infer = (
            self._last_pad_prediction is None
            or (self._pad_frame_counter % config.frame_interval == 0)
        )
        self._pad_frame_counter += 1

        if should_infer:
            try:
                prediction = snapshot.detector.predict(frame, primary)
            except AntiSpoofingError as exc:
                self._last_pad_prediction = None
                assessment = self._pad_tracker.update_error(str(exc))
                return self._blocked_pad_result(
                    frame, detections, primary, assessment
                )
            self._last_pad_prediction = prediction
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
                feature = self.engine.extract_feature(frame, primary)
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
