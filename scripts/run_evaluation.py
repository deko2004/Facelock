#!/usr/bin/env python3
"""Run the 1:1 evaluation pipeline and export CSV + chart artifacts.

Usage:
    venv/bin/python scripts/run_evaluation.py
    venv/bin/python scripts/run_evaluation.py path/to/dataset --threshold 0.363
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from core.app_context import AppContext  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the face matching pipeline")
    parser.add_argument(
        "dataset",
        nargs="?",
        default=str(settings.IMAGES_DIR / "eval_demo"),
        help="dataset folder arranged as dataset/<identity>/<images>",
    )
    parser.add_argument("--threshold", type=float, default=settings.DEFAULT_THRESHOLD)
    parser.add_argument(
        "--output-dir",
        default=str(settings.EVALUATION_DIR),
        help="folder for pairs.csv, summary.csv and far_frr_chart.png",
    )
    args = parser.parse_args(argv)

    context = AppContext.create(threshold=args.threshold)
    try:
        report = context.evaluation.evaluate(args.dataset, threshold=args.threshold)
        output = Path(args.output_dir)
        pairs_path, summary_path = context.evaluation.write_csv(
            report, output / "pairs.csv", output / "summary.csv"
        )
        chart_path = context.evaluation.render_chart(report, output / "far_frr_chart.png")

        metrics = report.metrics
        print(f"Dataset: {report.dataset_dir}")
        print(f"Identities: {len(report.identities)}  usable images: {report.image_count}")
        print(f"Genuine pairs: {report.genuine_count}  impostor pairs: {report.impostor_count}")
        print(f"Threshold: {report.threshold:.3f}")
        print(
            "Confusion matrix: "
            f"TP={metrics.tp} FN={metrics.fn} FP={metrics.fp} TN={metrics.tn}"
        )
        print(f"FAR: {_pct(metrics.far)}  FRR: {_pct(metrics.frr)}")
        print(f"Accuracy: {_pct(metrics.accuracy)}")
        for warning in report.warnings:
            print(f"WARNING: {warning}")
        for skipped in report.skipped:
            print(f"SKIPPED: {skipped}")
        print(f"CSV: {pairs_path}")
        print(f"Summary: {summary_path}")
        print(f"Chart: {chart_path}")
        return 0
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        context.shutdown()


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())
