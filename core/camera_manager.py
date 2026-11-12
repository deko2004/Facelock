"""Webcam discovery and safe camera lifecycle management.

The GUI runs camera I/O in a worker thread.  This class owns the OpenCV
``VideoCapture`` object and guarantees that ``release()`` is called on stop and
on error.
"""

from __future__ import annotations

import glob
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import settings


class CameraError(RuntimeError):
    """Raised when a camera cannot be opened or read."""


@dataclass(frozen=True)
class CameraInfo:
    index: int
    device: str
    name: str
    available: bool

    @property
    def display_name(self) -> str:
        status = "" if self.available else " (permission denied?)"
        return f"Camera {self.index}: {self.name}{status}"


def _device_name(index: int) -> str:
    sys_name = Path(f"/sys/class/video4linux/video{index}/name")
    try:
        value = sys_name.read_text(encoding="utf-8").strip()
        if value:
            return value
    except OSError:
        pass
    return f"/dev/video{index}"


def list_video_devices() -> list[CameraInfo]:
    """Return Linux video devices without opening them.

    Opening a camera just to enumerate it can steal it from another process and
    can block, so discovery only inspects ``/dev/video*``.  The actual open is
    attempted when the user presses Start.
    """
    devices: list[CameraInfo] = []
    for path_text in sorted(glob.glob("/dev/video*")):
        path = Path(path_text)
        try:
            index = int(path.name.replace("video", ""))
        except ValueError:
            continue
        devices.append(
            CameraInfo(
                index=index,
                device=str(path),
                name=_device_name(index),
                available=os.access(path, os.R_OK | os.W_OK),
            )
        )
    return devices


def probe_camera(index: int) -> bool:
    """Try to open and read one frame from *index*, then release it."""
    capture = None
    try:
        capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            return False
        ok, frame = capture.read()
        return bool(ok and frame is not None and frame.size > 0)
    except cv2.error:
        return False
    finally:
        if capture is not None:
            capture.release()


class CameraManager:
    """A small, thread-safe wrapper around ``cv2.VideoCapture``."""

    def __init__(self) -> None:
        self._capture: cv2.VideoCapture | None = None
        self._index: int | None = None
        self._lock = threading.RLock()
        self._width = 0
        self._height = 0
        self._fps = 0.0
        self._buffersize = 0
        self._active_reads: dict[int, int] = {}
        self._pending_releases: dict[int, Any] = {}

    @property
    def is_open(self) -> bool:
        with self._lock:
            return bool(self._capture is not None and self._capture.isOpened())

    @property
    def index(self) -> int | None:
        return self._index

    @property
    def properties(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "width": self._width,
                "height": self._height,
                "fps": self._fps,
                "buffersize": self._buffersize,
            }

    @staticmethod
    def _get_capture_property(capture, prop: int) -> float:
        getter = getattr(capture, "get", None)
        if not callable(getter):
            return 0.0
        try:
            return float(getter(prop))
        except (cv2.error, TypeError, ValueError):
            return 0.0

    def open(
        self,
        index: int = 0,
        width: int = settings.CAMERA_WIDTH,
        height: int = settings.CAMERA_HEIGHT,
        fps: int = settings.CAMERA_FPS,
        buffersize: int = settings.CAMERA_BUFFERSIZE,
    ) -> None:
        """Open a camera by index, trying V4L2 then the default backend."""
        self.release()
        backends: list[int | None] = []
        if hasattr(cv2, "CAP_V4L2"):
            backends.append(cv2.CAP_V4L2)
        backends.append(None)

        last_error = ""
        for backend in backends:
            try:
                capture = (
                    cv2.VideoCapture(int(index), backend)
                    if backend is not None
                    else cv2.VideoCapture(int(index))
                )
            except cv2.error as exc:
                last_error = str(exc)
                continue
            if capture.isOpened():
                try:
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
                    capture.set(cv2.CAP_PROP_FPS, int(fps))
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(buffersize)))
                except cv2.error:
                    pass
                with self._lock:
                    self._capture = capture
                    self._index = int(index)
                    self._width = int(
                        self._get_capture_property(
                            capture, cv2.CAP_PROP_FRAME_WIDTH
                        )
                    )
                    self._height = int(
                        self._get_capture_property(
                            capture, cv2.CAP_PROP_FRAME_HEIGHT
                        )
                    )
                    self._fps = self._get_capture_property(
                        capture, cv2.CAP_PROP_FPS
                    )
                    self._buffersize = int(
                        self._get_capture_property(
                            capture, cv2.CAP_PROP_BUFFERSIZE
                        )
                    )
                return
            capture.release()
            last_error = "camera did not open"

        raise CameraError(
            f"Could not open camera {index}. It may be missing, busy, or blocked "
            f"by a permission policy. {last_error}".strip()
        )

    def read(self) -> np.ndarray:
        """Read one frame, raising CameraError on failure.

        The blocking read runs outside the manager lock so ``release()`` never
        waits on camera I/O.  If release is requested mid-read, the capture
        object is released by the reader as soon as the read returns.
        """
        with self._lock:
            capture = self._capture
            if capture is None or not capture.isOpened():
                raise CameraError("Camera is not open.")
            key = id(capture)
            self._active_reads[key] = self._active_reads.get(key, 0) + 1
        try:
            try:
                ok, frame = capture.read()
            except cv2.error as exc:
                raise CameraError(f"Camera read failed: {exc}") from exc
            if not ok or frame is None or frame.size == 0:
                raise CameraError(
                    "Camera returned an empty frame. Check the connection or "
                    "whether another application is using the camera."
                )
            return frame
        finally:
            self._finish_read(key)

    def _finish_read(self, key: int) -> None:
        pending = None
        with self._lock:
            remaining = self._active_reads.get(key, 1) - 1
            if remaining > 0:
                self._active_reads[key] = remaining
            else:
                self._active_reads.pop(key, None)
                pending = self._pending_releases.pop(key, None)
        if pending is not None:
            try:
                pending.release()
            except cv2.error:
                pass

    def release(self) -> None:
        """Release the camera.  Safe to call more than once.

        If a read is in progress, release is deferred until that read returns
        so the capture object is never freed underneath a blocking backend call.
        """
        capture_to_release = None
        with self._lock:
            capture = self._capture
            self._capture = None
            self._index = None
            self._width = 0
            self._height = 0
            self._fps = 0.0
            self._buffersize = 0
            if capture is not None:
                key = id(capture)
                if self._active_reads.get(key, 0) > 0:
                    self._pending_releases[key] = capture
                else:
                    self._pending_releases.pop(key, None)
                    capture_to_release = capture
        if capture_to_release is not None:
            try:
                capture_to_release.release()
            except cv2.error:
                pass

    def __enter__(self) -> "CameraManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
