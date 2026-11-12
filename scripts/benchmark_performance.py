#!/usr/bin/env python3
"""Repeatable local benchmarks for the live-camera pipeline.

Examples:
    venv/bin/python scripts/benchmark_performance.py
    venv/bin/python scripts/benchmark_performance.py --camera --seconds 5
    venv/bin/python scripts/benchmark_performance.py --camera --pad --json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from core.anti_spoofing import AntiSpoofingDetector  # noqa: E402
from core.app_context import AppContext  # noqa: E402
from core.face_engine import FaceEngine  # noqa: E402
from gui.camera_worker import LiveCameraWorker  # noqa: E402

SAMPLE = ROOT / "tests" / "assets" / "anti_spoofing" / "image_T1.jpg"


class _CountingWorker(LiveCameraWorker):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.count = 0
        self.last_live = None

    def _publish(self, result) -> None:
        if not result.finished:
            self.count += 1
            if not result.error:
                self.last_live = result
        super()._publish(result)


def _time_ms(work, repeats: int = 30) -> dict[str, float]:
    for _ in range(3):
        work()
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        work()
        values.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(values)
    return {
        "mean_ms": round(statistics.mean(values), 2),
        "median_ms": round(statistics.median(values), 2),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 2),
        "min_ms": round(min(values), 2),
    }


def _rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024.0, 1)
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def benchmark_core() -> dict:
    if not SAMPLE.is_file():
        raise SystemExit(f"Sample image not found: {SAMPLE}")
    image = cv2.imread(str(SAMPLE))
    if image is None:
        raise SystemExit(f"Could not read sample image: {SAMPLE}")

    started = time.perf_counter()
    engine = FaceEngine()
    load_ms = round((time.perf_counter() - started) * 1000.0, 2)
    detections = engine.detect(image)
    if not detections:
        raise SystemExit("No face detected in the benchmark sample.")
    face = engine.largest_face(detections)
    assert face is not None

    result = {
        "engine_load_ms": load_ms,
        "rss_mb": _rss_mb(),
        "detect": _time_ms(lambda: engine.detect(image)),
        "extract": _time_ms(lambda: engine.extract_feature(image, face)),
        "preview": _time_ms(lambda: _preview(image)),
    }

    if settings.PAD_MODEL.is_file():
        try:
            started = time.perf_counter()
            detector = AntiSpoofingDetector()
            pad_load_ms = round((time.perf_counter() - started) * 1000.0, 2)
            crop = detector.crop_face(image, face.bbox)
            result["pad_load_ms"] = pad_load_ms
            result["pad_predict"] = _time_ms(lambda: detector.predict_crop(crop))
            detector.close()
        except Exception as exc:  # optional model/runtime
            result["pad_error"] = str(exc)
    else:
        result["pad_error"] = "PAD model not found"
    return result


def _preview(image: np.ndarray, target: tuple[int, int] = (820, 470)) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(target[0] / width, target[1] / height, 1.0)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def benchmark_worker(seconds: float, use_pad: bool) -> dict:
    with tempfile.TemporaryDirectory(prefix="facelock-bench-") as temp_dir:
        temp = Path(temp_dir)
        context = AppContext.create(
            db_path=temp / "bench.db",
            key_path=temp / "bench.key",
            settings_path=temp / "bench.json",
        )
        try:
            runtime = None
            if use_pad:
                context.anti_spoofing.enable()
                context.anti_spoofing.ensure_loaded()
                runtime = context.anti_spoofing.runtime
            worker = _CountingWorker(
                engine=context.engine,
                camera_index=0,
                threshold=context.threshold,
                camera_manager=context.camera,
                pad_runtime=runtime,
            )
            started = time.perf_counter()
            worker.start()
            time.sleep(max(1.0, seconds))
            worker.stop()
            worker.join(timeout=2.0)
            elapsed = max(1e-6, time.perf_counter() - started)
            last = worker.last_live
            result = {
                "seconds": round(elapsed, 2),
                "results": worker.count,
                "throughput_per_s": round(worker.count / elapsed, 2),
                "rss_mb": _rss_mb(),
            }
            if last is not None:
                result["fps"] = round(last.fps, 2)
                result["faces"] = len(last.detections)
                result["pad_status"] = last.pad_status
                if last.diagnostics is not None:
                    result["diagnostics"] = last.diagnostics.summary()
            return result
        finally:
            context.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark FaceLock performance")
    parser.add_argument("--camera", action="store_true", help="also benchmark webcam")
    parser.add_argument("--pad", action="store_true", help="enable PAD in webcam run")
    parser.add_argument(
        "--seconds", type=float, default=5.0, help="webcam benchmark duration"
    )
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args(argv)

    report: dict = {"core": benchmark_core()}
    if args.camera:
        try:
            report["camera"] = benchmark_worker(args.seconds, args.pad)
        except Exception as exc:
            report["camera_error"] = str(exc)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        core = report["core"]
        print(f"Engine load: {core['engine_load_ms']} ms | RSS {core['rss_mb']} MB")
        for key in ("detect", "extract", "pad_predict", "preview"):
            if key in core:
                values = core[key]
                print(
                    f"{key}: mean {values['mean_ms']} ms | "
                    f"median {values['median_ms']} | p95 {values['p95_ms']}"
                )
        if "camera" in report:
            camera = report["camera"]
            print(
                f"camera: {camera['results']} results in {camera['seconds']} s "
                f"({camera['throughput_per_s']}/s), RSS {camera['rss_mb']} MB"
            )
            if "diagnostics" in camera:
                print(f"  {camera['diagnostics']}")
        if "camera_error" in report:
            print(f"camera benchmark failed: {report['camera_error']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
