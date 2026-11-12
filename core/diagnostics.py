"""Lightweight runtime metrics for the live-camera pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


def _read_rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


@dataclass(frozen=True)
class PerformanceSnapshot:
    capture_fps: float = 0.0
    process_fps: float = 0.0
    detect_ms: float | None = None
    sface_ms: float | None = None
    pad_ms: float | None = None
    process_ms: float | None = None
    dropped_frames: int = 0
    cpu_percent: float = 0.0
    rss_mb: float = 0.0

    def summary(self) -> str:
        parts = [
            f"Capture {self.capture_fps:.1f} / Process {self.process_fps:.1f} fps",
        ]
        if self.detect_ms is not None:
            parts.append(f"YuNet {self.detect_ms:.1f} ms")
        if self.sface_ms is not None:
            parts.append(f"SFace {self.sface_ms:.1f} ms")
        if self.pad_ms is not None:
            parts.append(f"PAD {self.pad_ms:.1f} ms")
        if self.process_ms is not None:
            parts.append(f"Frame {self.process_ms:.1f} ms")
        parts.append(f"CPU {self.cpu_percent:.0f}%")
        parts.append(f"RAM {self.rss_mb:.0f} MB")
        parts.append(f"Dropped {self.dropped_frames}")
        return " | ".join(parts)


class PerformanceMonitor:
    """Exponential moving averages plus sampled process CPU/RSS."""

    def __init__(self, alpha: float = 0.2, sample_interval: float = 1.0) -> None:
        self._alpha = max(0.01, min(1.0, float(alpha)))
        self._sample_interval = max(0.1, float(sample_interval))
        self._detect_ms: float | None = None
        self._sface_ms: float | None = None
        self._pad_ms: float | None = None
        self._process_ms: float | None = None
        self._cpu_percent = 0.0
        self._rss_mb = _read_rss_mb()
        self._last_sample_time = time.perf_counter()
        self._last_process_time = time.process_time()

    def _update(self, current: float | None, value: float) -> float:
        if current is None:
            return max(0.0, float(value))
        return (1.0 - self._alpha) * current + self._alpha * max(0.0, float(value))

    def record_detect(self, milliseconds: float) -> None:
        self._detect_ms = self._update(self._detect_ms, milliseconds)

    def record_sface(self, milliseconds: float) -> None:
        self._sface_ms = self._update(self._sface_ms, milliseconds)

    def record_pad(self, milliseconds: float) -> None:
        self._pad_ms = self._update(self._pad_ms, milliseconds)

    def record_process(self, milliseconds: float) -> None:
        self._process_ms = self._update(self._process_ms, milliseconds)

    def _sample_system(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._last_sample_time
        if elapsed < self._sample_interval:
            return
        process_now = time.process_time()
        cpu_seconds = max(0.0, process_now - self._last_process_time)
        self._cpu_percent = 100.0 * cpu_seconds / elapsed
        self._rss_mb = _read_rss_mb()
        self._last_sample_time = now
        self._last_process_time = process_now

    def snapshot(
        self,
        capture_fps: float,
        process_fps: float,
        dropped_frames: int,
    ) -> PerformanceSnapshot:
        self._sample_system()
        return PerformanceSnapshot(
            capture_fps=max(0.0, float(capture_fps)),
            process_fps=max(0.0, float(process_fps)),
            detect_ms=self._detect_ms,
            sface_ms=self._sface_ms,
            pad_ms=self._pad_ms,
            process_ms=self._process_ms,
            dropped_frames=max(0, int(dropped_frames)),
            cpu_percent=self._cpu_percent,
            rss_mb=self._rss_mb,
        )
