from __future__ import annotations

import numpy as np

from core.frame_buffer import LatestFrameBuffer


def test_latest_frame_replaces_old_frame_and_counts_drops() -> None:
    buffer = LatestFrameBuffer()
    first = np.zeros((10, 10, 3), dtype=np.uint8)
    second = np.ones((10, 10, 3), dtype=np.uint8)

    buffer.put(first)
    buffer.put(second)

    assert buffer.dropped == 1
    item = buffer.get(after_sequence=0, timeout=0.1)
    assert item is not None
    frame, sequence = item
    assert frame is second
    assert sequence == 2
    assert buffer.get(after_sequence=sequence, timeout=0.01) is None


def test_consumed_frames_are_not_counted_as_dropped() -> None:
    buffer = LatestFrameBuffer()
    for index in range(5):
        frame = np.full((2, 2, 3), index, dtype=np.uint8)
        buffer.put(frame)
        item = buffer.get(after_sequence=index, timeout=0.1)
        assert item is not None
        assert item[1] == index + 1

    assert buffer.dropped == 0


def test_closed_buffer_wakes_waiters_and_ignores_new_frames() -> None:
    buffer = LatestFrameBuffer()
    buffer.close()
    buffer.put(np.zeros((2, 2, 3), dtype=np.uint8))
    assert buffer.closed
    assert buffer.get(after_sequence=0, timeout=0.01) is None
    assert buffer.dropped == 0
