"""A one-frame hand-off buffer that always keeps the newest camera frame."""

from __future__ import annotations

import threading
import time

import numpy as np


class LatestFrameBuffer:
    """Thread-safe latest-frame slot.

    The capture thread never waits for the processor.  If processing falls
    behind, old frames are discarded and ``dropped`` is incremented.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._consumed_sequence = 0
        self._dropped = 0
        self._closed = False

    @property
    def dropped(self) -> int:
        with self._condition:
            return self._dropped

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def put(self, frame: np.ndarray) -> None:
        with self._condition:
            if self._closed:
                return
            if self._frame is not None and self._sequence > self._consumed_sequence:
                self._dropped += 1
            self._frame = frame
            self._sequence += 1
            self._condition.notify()

    def get(
        self, after_sequence: int = 0, timeout: float = 0.1
    ) -> tuple[np.ndarray, int] | None:
        """Return the newest frame newer than *after_sequence*.

        ``None`` is returned on timeout or after the buffer is closed.
        """
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while not self._closed and self._sequence <= after_sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if self._closed:
                return None
            assert self._frame is not None
            self._consumed_sequence = max(self._consumed_sequence, self._sequence)
            return self._frame, self._sequence

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
