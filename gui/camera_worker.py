"""Background webcam worker used by the live, verification and capture pages."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from core.camera_manager import CameraError, CameraManager
from core.face_engine import FaceDetection, FaceEngine, FaceEngineError
from core.image_processor import draw_faces
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

    @property
    def recognized_count(self) -> int:
        return sum(1 for identification in self.identifications if identification.matched)


class LiveCameraWorker(threading.Thread):
    """Read frames, run YuNet/SFace and publish the latest result.

    The Tkinter main thread polls ``result_queue``; no Tk widget is touched from
    this thread.
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
    ) -> None:
        super().__init__(daemon=True)
        self.engine = engine
        self.camera_index = int(camera_index)
        self.template = template
        self.user_id = user_id
        self.threshold = float(threshold)
        self.identities = list(identities or [])
        self.camera = camera_manager or CameraManager()
        self.result_queue: queue.Queue[LiveFrameResult] = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._fps = 0.0
        self._last_time = time.perf_counter()

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
                            similarity = self.engine.cosine_similarity(
                                probe, self.template
                            )
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

                self._publish(
                    LiveFrameResult(
                        raw_frame=frame,
                        annotated_frame=annotated,
                        detections=detections,
                        identifications=identifications,
                        labels=labels,
                        similarity=similarity,
                        verified=verified,
                        user_id=self.user_id,
                        fps=self._fps,
                    )
                )
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
