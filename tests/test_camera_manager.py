from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.camera_manager import (
    CameraError,
    CameraInfo,
    CameraManager,
    list_video_devices,
    probe_camera,
)


class FakeCapture:
    def __init__(self, opened: bool = False, frame: np.ndarray | None = None) -> None:
        self.opened = opened
        self.frame = frame
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def set(self, *args, **kwargs) -> bool:
        return True

    def read(self):
        if self.frame is None:
            return False, None
        return True, self.frame

    def release(self) -> None:
        self.released = True


def test_list_video_devices_returns_camera_info() -> None:
    devices = list_video_devices()
    assert isinstance(devices, list)
    for device in devices:
        assert isinstance(device, CameraInfo)
        assert device.device.startswith("/dev/video")
        assert "Camera" in device.display_name


def test_read_before_open_raises() -> None:
    with pytest.raises(CameraError):
        CameraManager().read()


def test_open_failure_raises(monkeypatch) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: FakeCapture(False))
    manager = CameraManager()
    with pytest.raises(CameraError):
        manager.open(0)


def test_open_and_read_with_fake_capture(monkeypatch) -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    monkeypatch.setattr(
        cv2, "VideoCapture", lambda *args, **kwargs: FakeCapture(True, frame)
    )
    manager = CameraManager()
    manager.open(0)
    assert manager.is_open
    assert manager.index == 0
    assert manager.read().shape == frame.shape
    manager.release()
    assert not manager.is_open
    manager.release()


def test_probe_camera_success(monkeypatch) -> None:
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    monkeypatch.setattr(
        cv2, "VideoCapture", lambda *args, **kwargs: FakeCapture(True, frame)
    )
    assert probe_camera(0)


def test_probe_camera_failure(monkeypatch) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: FakeCapture(False))
    assert not probe_camera(0)
