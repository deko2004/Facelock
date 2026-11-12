from __future__ import annotations

import threading

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
        self.set_calls: list[tuple[int, float]] = []
        self.values: dict[int, float] = {}

    def isOpened(self) -> bool:
        return self.opened

    def set(self, prop: int, value: float) -> bool:
        self.set_calls.append((prop, value))
        self.values[prop] = value
        return True

    def get(self, prop: int) -> float:
        return self.values.get(prop, 0.0)

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


def test_release_during_blocking_read_is_deferred(monkeypatch) -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    capture = FakeCapture(True, frame)
    started = threading.Event()
    finish = threading.Event()

    def blocking_read():
        started.set()
        assert finish.wait(timeout=2.0)
        return True, frame

    capture.read = blocking_read  # type: ignore[method-assign]
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: capture)

    manager = CameraManager()
    manager.open(0)
    result: list[np.ndarray] = []
    reader = threading.Thread(target=lambda: result.append(manager.read()))
    reader.start()
    assert started.wait(timeout=1.0)

    releaser = threading.Thread(target=manager.release)
    releaser.start()
    releaser.join(timeout=0.5)
    assert not releaser.is_alive()
    assert not capture.released
    assert not manager.is_open

    finish.set()
    reader.join(timeout=2.0)
    assert not reader.is_alive()
    assert result and result[0].shape == frame.shape
    assert capture.released


def test_open_sets_buffer_size_and_reports_properties(monkeypatch) -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    capture = FakeCapture(True, frame)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: capture)

    manager = CameraManager()
    manager.open(0, width=640, height=480, fps=30, buffersize=2)

    assert (cv2.CAP_PROP_BUFFERSIZE, 2) in capture.set_calls
    assert manager.properties == {
        "width": 640,
        "height": 480,
        "fps": 30.0,
        "buffersize": 2,
    }
    manager.release()
    assert manager.properties == {
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "buffersize": 0,
    }


def test_probe_camera_success(monkeypatch) -> None:
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    monkeypatch.setattr(
        cv2, "VideoCapture", lambda *args, **kwargs: FakeCapture(True, frame)
    )
    assert probe_camera(0)


def test_probe_camera_failure(monkeypatch) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: FakeCapture(False))
    assert not probe_camera(0)
