"""Frame-source abstraction for physical cameras and recorded videos.

The authentication worker only needs three operations from an input source:
``open``, ``read`` and ``release``.  Keeping that contract explicit lets the
Video Injection Lab feed a recorded file through the same YuNet/SFace/PAD
pipeline without creating a second recognition implementation.

This module deliberately does not claim that a source is authentic.  Source
classification is a best-effort metadata check; see
``core/source_integrity.py`` for the security policy and its limitations.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from core.camera_manager import CameraError, CameraManager

SOURCE_PHYSICAL = "PHYSICAL_CAMERA"
SOURCE_RECORDED = "RECORDED_VIDEO"
SOURCE_VIRTUAL = "VIRTUAL_CAMERA"
SOURCE_UNKNOWN = "UNKNOWN"

SOURCE_LABELS = {
    SOURCE_PHYSICAL: "Physical Camera",
    SOURCE_RECORDED: "Recorded Video",
    SOURCE_VIRTUAL: "Virtual Camera",
    SOURCE_UNKNOWN: "Unknown",
}

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}

VIRTUAL_DEVICE_MARKERS = (
    "virtual",
    "v4l2loopback",
    "v4l2 loopback",
    "loopback",
    "dummy",
    "obs virtual",
    "obs-camera",
    "manycam",
    "webcamoid",
    "akvcam",
    "chromium",
    "screen capture",
)


class FrameSourceError(RuntimeError):
    """Raised when a frame source cannot be opened or read."""


class FrameSourceEnded(FrameSourceError):
    """Raised when a finite source reaches its end."""


@dataclass(frozen=True)
class FrameSourceInfo:
    """Non-biometric metadata describing the selected input source."""

    kind: str
    identifier: str
    display_name: str
    device_path: str | None = None
    file_path: str | None = None
    device_name: str | None = None
    sysfs_path: str | None = None
    sysfs_device_path: str | None = None
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    duration_seconds: float = 0.0
    details: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def kind_label(self) -> str:
        return SOURCE_LABELS.get(self.kind, self.kind)

    @property
    def is_recorded(self) -> bool:
        return self.kind == SOURCE_RECORDED

    @property
    def is_physical(self) -> bool:
        return self.kind == SOURCE_PHYSICAL

    @property
    def is_virtual(self) -> bool:
        return self.kind == SOURCE_VIRTUAL

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "identifier": self.identifier,
            "display_name": self.display_name,
            "device_path": self.device_path,
            "file_path": self.file_path,
            "device_name": self.device_name,
            "sysfs_path": self.sysfs_path,
            "sysfs_device_path": self.sysfs_device_path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_seconds": self.duration_seconds,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class PlaybackProgress:
    frame_index: int
    frame_count: int
    seconds: float
    duration_seconds: float
    paused: bool
    finished: bool

    @property
    def fraction(self) -> float:
        if self.duration_seconds > 0:
            return max(0.0, min(1.0, self.seconds / self.duration_seconds))
        if self.frame_count > 0:
            return max(0.0, min(1.0, self.frame_index / self.frame_count))
        return 0.0


class FrameSource:
    """Minimal interface consumed by ``LiveCameraWorker``."""

    @property
    def info(self) -> FrameSourceInfo:
        raise NotImplementedError

    @property
    def is_open(self) -> bool:
        raise NotImplementedError

    def open(self) -> None:
        raise NotImplementedError

    def read(self) -> np.ndarray:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _resolve_sysfs_device(sysfs_path: Path) -> str:
    link = sysfs_path / "device"
    try:
        return os.path.realpath(link) if link.exists() else ""
    except OSError:
        return ""


def inspect_linux_video_device(index: int) -> FrameSourceInfo:
    """Build metadata for ``/dev/videoN`` without opening the device.

    The classification is heuristic.  A physical-looking sysfs path or a
    non-virtual device name does not prove that frames are authentic.
    """
    device_path = Path(f"/dev/video{index}")
    sysfs_path = Path(f"/sys/class/video4linux/video{index}")
    device_name = _read_text(sysfs_path / "name")
    resolved = _resolve_sysfs_device(sysfs_path)
    combined = " ".join(
        part for part in (device_name, str(device_path), resolved) if part
    ).lower()

    if any(marker in combined for marker in VIRTUAL_DEVICE_MARKERS):
        kind = SOURCE_VIRTUAL
    elif (
        device_path.exists()
        and os.access(device_path, os.R_OK)
        and resolved
        and ("/usb" in resolved or "/pci" in resolved or "/platform" in resolved)
    ):
        kind = SOURCE_PHYSICAL
    else:
        kind = SOURCE_UNKNOWN

    identifier = f"v4l2:{device_path}:{resolved or device_name or 'unknown'}"
    details = (
        ("device", str(device_path)),
        ("sysfs", str(sysfs_path)),
        ("sysfs_device", resolved or "unknown"),
    )
    return FrameSourceInfo(
        kind=kind,
        identifier=identifier,
        display_name=device_name or str(device_path),
        device_path=str(device_path),
        device_name=device_name or None,
        sysfs_path=str(sysfs_path) if sysfs_path.exists() else None,
        sysfs_device_path=resolved or None,
        details=details,
    )


class PhysicalCameraSource(FrameSource):
    """Adapt ``CameraManager`` to the common frame-source interface."""

    def __init__(
        self,
        manager: CameraManager,
        index: int,
        info: FrameSourceInfo | None = None,
    ) -> None:
        self.manager = manager
        self.index = int(index)
        self._info = info or inspect_linux_video_device(self.index)

    @property
    def info(self) -> FrameSourceInfo:
        return self._info

    @property
    def is_open(self) -> bool:
        return self.manager.is_open

    def open(self) -> None:
        try:
            self.manager.open(self.index)
        except CameraError:
            raise
        properties = getattr(self.manager, "properties", None) or {}
        self._info = replace(
            self._info,
            width=int(properties.get("width", 0) or 0),
            height=int(properties.get("height", 0) or 0),
            fps=float(properties.get("fps", 0.0) or 0.0),
        )

    def read(self) -> np.ndarray:
        return self.manager.read()

    def release(self) -> None:
        self.manager.release()


class RecordedVideoSource(FrameSource):
    """Decode a local video file with pacing, pause/resume and cleanup.

    Frames are paced to the file's reported FPS so the capture thread behaves
    like a camera and cannot decode an entire file into the processing queue.
    The default ``loop=True`` makes repeated attack demonstrations convenient.
    """

    def __init__(
        self,
        path: Path | str,
        loop: bool = True,
        pace: bool = True,
    ) -> None:
        self.path = Path(path)
        self.loop = bool(loop)
        self.pace = bool(pace)
        self._capture: cv2.VideoCapture | None = None
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._paused = False
        self._released = False
        self._frame_index = 0
        self._seconds = 0.0
        self._last_read_at = 0.0
        self._info = FrameSourceInfo(
            kind=SOURCE_RECORDED,
            identifier=f"file:{self.path.resolve()}",
            display_name=self.path.name,
            file_path=str(self.path),
            details=(("file", str(self.path.resolve())),),
        )

    @property
    def info(self) -> FrameSourceInfo:
        return self._info

    @property
    def is_open(self) -> bool:
        with self._lock:
            return bool(self._capture is not None and self._capture.isOpened())

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def progress(self) -> PlaybackProgress:
        with self._lock:
            finished = (
                self._info.frame_count > 0
                and self._frame_index >= self._info.frame_count
            )
            return PlaybackProgress(
                frame_index=self._frame_index,
                frame_count=self._info.frame_count,
                seconds=self._seconds,
                duration_seconds=self._info.duration_seconds,
                paused=self._paused,
                finished=finished,
            )

    def open(self) -> None:
        if not self.path.exists():
            raise FrameSourceError(f"Video file not found: {self.path}")
        if not self.path.is_file():
            raise FrameSourceError(f"Video path is not a regular file: {self.path}")
        if self.path.suffix.lower() not in VIDEO_EXTENSIONS:
            allowed = ", ".join(sorted(VIDEO_EXTENSIONS))
            raise FrameSourceError(
                f"Unsupported video type. Allowed extensions: {allowed}"
            )

        with self._lock:
            previous = self._capture
            self._capture = None
        if previous is not None:
            previous.release()

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise FrameSourceError(
                f"OpenCV could not open the video: {self.path.name}"
            )

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0.0 or fps > 240.0:
            fps = 30.0
        duration = frame_count / fps if frame_count > 0 else 0.0
        with self._lock:
            self._capture = capture
            self._released = False
            self._paused = False
            self._frame_index = 0
            self._seconds = 0.0
            self._last_read_at = 0.0
            self._info = replace(
                self._info,
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
                duration_seconds=duration,
                details=self._info.details
                + (("fps", f"{fps:.3f}"), ("frames", str(frame_count))),
            )

    def read(self) -> np.ndarray:
        with self._condition:
            while self._paused and not self._released:
                self._condition.wait(0.1)
            if self._released or self._capture is None:
                raise FrameSourceError("Recorded video source is not open.")
            capture = self._capture
            fps = self._info.fps or 30.0

        self._pace_frame(fps)

        with self._lock:
            if self._released or self._capture is None:
                raise FrameSourceError("Recorded video source was released.")
            try:
                ok, frame = capture.read()
            except cv2.error as exc:
                raise FrameSourceError(f"Video decoding failed: {exc}") from exc
            if not ok or frame is None or frame.size == 0:
                if not self.loop:
                    raise FrameSourceEnded(
                        f"Video playback finished: {self.path.name}"
                    )
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._frame_index = 0
                self._seconds = 0.0
                try:
                    ok, frame = capture.read()
                except cv2.error as exc:
                    raise FrameSourceError(
                        f"Video restart failed: {exc}"
                    ) from exc
                if not ok or frame is None or frame.size == 0:
                    raise FrameSourceEnded(
                        f"Video could not be restarted: {self.path.name}"
                    )
            self._frame_index += 1
            self._seconds = self._frame_index / fps
            return frame

    def _pace_frame(self, fps: float) -> None:
        if not self.pace or fps <= 0:
            return
        interval = 1.0 / min(fps, 120.0)
        now = time.monotonic()
        if self._last_read_at:
            remaining = interval - (now - self._last_read_at)
            if remaining > 0:
                time.sleep(remaining)
        self._last_read_at = time.monotonic()

    def pause(self) -> None:
        with self._condition:
            self._paused = True

    def resume(self) -> None:
        with self._condition:
            self._paused = False
            self._last_read_at = 0.0
            self._condition.notify_all()

    def restart(self) -> None:
        with self._lock:
            if self._capture is None or self._released:
                raise FrameSourceError("Recorded video source is not open.")
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._frame_index = 0
            self._seconds = 0.0
            self._last_read_at = 0.0
            self._paused = False
            self._condition.notify_all()

    def release(self) -> None:
        with self._condition:
            if self._released:
                return
            self._released = True
            self._paused = False
            capture = self._capture
            self._capture = None
            self._condition.notify_all()
        if capture is not None:
            try:
                capture.release()
            except cv2.error:
                pass

    def __enter__(self) -> "RecordedVideoSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def source_info_from_mapping(data: Mapping[str, object]) -> FrameSourceInfo:
    """Reconstruct source metadata for tests or persisted experiment records."""
    details = data.get("details", ())
    if isinstance(details, Mapping):
        details = tuple((str(key), str(value)) for key, value in details.items())
    return FrameSourceInfo(
        kind=str(data.get("kind", SOURCE_UNKNOWN)),
        identifier=str(data.get("identifier", "unknown")),
        display_name=str(data.get("display_name", "Unknown")),
        device_path=data.get("device_path") if isinstance(data.get("device_path"), str) else None,
        file_path=data.get("file_path") if isinstance(data.get("file_path"), str) else None,
        device_name=data.get("device_name") if isinstance(data.get("device_name"), str) else None,
        sysfs_path=data.get("sysfs_path") if isinstance(data.get("sysfs_path"), str) else None,
        sysfs_device_path=(
            data.get("sysfs_device_path")
            if isinstance(data.get("sysfs_device_path"), str)
            else None
        ),
        width=int(data.get("width", 0) or 0),
        height=int(data.get("height", 0) or 0),
        fps=float(data.get("fps", 0.0) or 0.0),
        frame_count=int(data.get("frame_count", 0) or 0),
        duration_seconds=float(data.get("duration_seconds", 0.0) or 0.0),
        details=tuple(details),  # type: ignore[arg-type]
    )
