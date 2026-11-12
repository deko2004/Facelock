from __future__ import annotations

import time

from core.diagnostics import PerformanceMonitor, PerformanceSnapshot


def test_snapshot_summary_contains_core_metrics() -> None:
    snapshot = PerformanceSnapshot(
        capture_fps=16.5,
        process_fps=14.2,
        detect_ms=24.0,
        sface_ms=13.0,
        pad_ms=5.0,
        process_ms=40.0,
        dropped_frames=3,
        cpu_percent=95.0,
        rss_mb=210.0,
    )
    summary = snapshot.summary()
    assert "Capture 16.5" in summary
    assert "Process 14.2" in summary
    assert "YuNet 24.0 ms" in summary
    assert "SFace 13.0 ms" in summary
    assert "PAD 5.0 ms" in summary
    assert "Dropped 3" in summary


def test_monitor_records_and_samples() -> None:
    monitor = PerformanceMonitor(alpha=1.0, sample_interval=0.0)
    monitor.record_detect(20.0)
    monitor.record_sface(10.0)
    monitor.record_pad(4.0)
    monitor.record_process(30.0)
    time.sleep(0.01)
    snapshot = monitor.snapshot(capture_fps=15.0, process_fps=12.0, dropped_frames=2)
    assert snapshot.detect_ms == 20.0
    assert snapshot.sface_ms == 10.0
    assert snapshot.pad_ms == 4.0
    assert snapshot.process_ms == 30.0
    assert snapshot.cpu_percent >= 0.0
    assert snapshot.rss_mb >= 0.0
