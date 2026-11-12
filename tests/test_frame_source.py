from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.frame_source import (
    SOURCE_RECORDED,
    FrameSourceEnded,
    FrameSourceError,
    RecordedVideoSource,
    source_info_from_mapping,
)


def _write_video(path: Path, frames: int = 5, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (64, 48),
    )
    if not writer.isOpened():
        pytest.skip("MJPG video writer is unavailable in this OpenCV build")
    try:
        for index in range(frames):
            frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_recorded_video_source_reads_and_reports_progress(tmp_path: Path) -> None:
    path = tmp_path / "attack.avi"
    _write_video(path, frames=4)
    source = RecordedVideoSource(path, loop=False, pace=False)
    source.open()
    try:
        assert source.info.kind == SOURCE_RECORDED
        assert source.info.display_name == "attack.avi"
        assert source.info.frame_count == 4
        assert source.info.width == 64
        assert source.info.height == 48
        for _ in range(4):
            frame = source.read()
            assert frame.shape == (48, 64, 3)
        assert source.progress.frame_index == 4
        with pytest.raises(FrameSourceEnded):
            source.read()
    finally:
        source.release()
    assert not source.is_open


def test_recorded_video_source_loops_and_can_restart(tmp_path: Path) -> None:
    path = tmp_path / "loop.avi"
    _write_video(path, frames=2)
    source = RecordedVideoSource(path, loop=True, pace=False)
    source.open()
    try:
        for _ in range(5):
            assert source.read().shape == (48, 64, 3)
        assert source.progress.frame_index == 1
        source.restart()
        assert source.progress.frame_index == 0
        assert source.read().shape == (48, 64, 3)
    finally:
        source.release()


def test_recorded_video_source_rejects_missing_and_unsupported(tmp_path: Path) -> None:
    source = RecordedVideoSource(tmp_path / "missing.avi")
    with pytest.raises(FrameSourceError):
        source.open()

    text = tmp_path / "not_video.txt"
    text.write_text("not a video", encoding="utf-8")
    with pytest.raises(FrameSourceError):
        RecordedVideoSource(text).open()


def test_source_info_mapping_round_trip() -> None:
    info = source_info_from_mapping(
        {
            "kind": SOURCE_RECORDED,
            "identifier": "file:/tmp/attack.avi",
            "display_name": "attack.avi",
            "file_path": "/tmp/attack.avi",
            "width": 640,
            "height": 480,
            "fps": 25.0,
            "frame_count": 50,
            "duration_seconds": 2.0,
            "details": {"codec": "MJPG"},
        }
    )
    assert info.kind == SOURCE_RECORDED
    assert info.width == 640
    assert info.details == (("codec", "MJPG"),)
