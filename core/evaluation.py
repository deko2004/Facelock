"""Evaluation metrics for 1:1 face comparisons.

The module never invents scores.  It reads a folder of images arranged as
``dataset/<identity>/<image>``, computes genuine and impostor pairs with the
same SFace pipeline used by the application, and reports the confusion matrix
and FAR/FRR at a chosen threshold.  A small public sample set is included for a
pipeline smoke test; it is not a biometric benchmark.
"""

from __future__ import annotations

import csv
import itertools
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import settings
from core.face_engine import FaceEngine, FaceEngineError
from core.image_processor import ImageLoadError, load_image


class EvaluationError(RuntimeError):
    """Raised when an evaluation dataset is missing or unusable."""


@dataclass(frozen=True)
class EvaluationPair:
    identity_a: str
    identity_b: str
    file_a: str
    file_b: str
    genuine: bool
    similarity: float
    accepted: bool
    correct: bool


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    tp: int
    fn: int
    fp: int
    tn: int

    @property
    def total_genuine(self) -> int:
        return self.tp + self.fn

    @property
    def total_impostor(self) -> int:
        return self.fp + self.tn

    @property
    def total(self) -> int:
        return self.tp + self.fn + self.fp + self.tn

    @property
    def far(self) -> float | None:
        if self.total_impostor == 0:
            return None
        return self.fp / self.total_impostor

    @property
    def frr(self) -> float | None:
        if self.total_genuine == 0:
            return None
        return self.fn / self.total_genuine

    @property
    def accuracy(self) -> float | None:
        if self.total == 0:
            return None
        return (self.tp + self.tn) / self.total

    @property
    def precision(self) -> float | None:
        denominator = self.tp + self.fp
        if denominator == 0:
            return None
        return self.tp / denominator

    @property
    def recall(self) -> float | None:
        denominator = self.tp + self.fn
        if denominator == 0:
            return None
        return self.tp / denominator


@dataclass(frozen=True)
class EvaluationReport:
    dataset_dir: str
    threshold: float
    identities: tuple[str, ...]
    image_count: int
    pairs: tuple[EvaluationPair, ...]
    metrics: ThresholdMetrics
    sweep: tuple[ThresholdMetrics, ...]
    warnings: tuple[str, ...]
    skipped: tuple[str, ...]
    generated_at: str
    elapsed_seconds: float

    @property
    def genuine_count(self) -> int:
        return sum(1 for pair in self.pairs if pair.genuine)

    @property
    def impostor_count(self) -> int:
        return sum(1 for pair in self.pairs if not pair.genuine)


@dataclass(frozen=True)
class _ImageSample:
    identity: str
    path: Path
    feature: np.ndarray
    detection_count: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _image_files(folder: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in settings.ALLOWED_IMAGE_EXTENSIONS:
            files.append(path)
    return files


def scan_dataset(dataset_dir: Path | str) -> dict[str, list[Path]]:
    """Return ``{identity: [images]}`` for a folder of identity subfolders."""
    root = Path(dataset_dir)
    if not root.exists() or not root.is_dir():
        raise EvaluationError(f"Dataset folder not found: {root}")
    identities: dict[str, list[Path]] = {}
    for child in sorted(root.iterdir()):
        if child.is_dir():
            images = _image_files(child)
            if images:
                identities[child.name] = images
    if not identities:
        raise EvaluationError(
            "No identity subfolders with images were found. Expected "
            "dataset/<identity>/<image files>."
        )
    return identities


def metrics_from_pairs(
    pairs: list[EvaluationPair] | tuple[EvaluationPair, ...], threshold: float
) -> ThresholdMetrics:
    """Compute the confusion matrix from pair labels and decisions."""
    tp = fn = fp = tn = 0
    for pair in pairs:
        accepted = pair.similarity >= threshold
        if pair.genuine:
            if accepted:
                tp += 1
            else:
                fn += 1
        else:
            if accepted:
                fp += 1
            else:
                tn += 1
    return ThresholdMetrics(float(threshold), tp, fn, fp, tn)


class EvaluationService:
    """Run genuine/impostor comparisons and export metrics."""

    def __init__(self, engine: FaceEngine) -> None:
        self.engine = engine

    def evaluate(
        self,
        dataset_dir: Path | str,
        threshold: float = settings.DEFAULT_THRESHOLD,
        max_impostor_pairs: int = 500,
        max_genuine_pairs: int = 500,
        max_images_per_identity: int = 30,
    ) -> EvaluationReport:
        started = time.perf_counter()
        threshold = settings.clamp_threshold(threshold)
        identities = scan_dataset(dataset_dir)
        warnings: list[str] = []
        skipped: list[str] = []
        samples: list[_ImageSample] = []

        for identity, paths in identities.items():
            for path in paths[:max_images_per_identity]:
                try:
                    image = load_image(path)
                    detections = self.engine.detect(image)
                except (ImageLoadError, FaceEngineError) as exc:
                    skipped.append(f"{path.name}: {exc}")
                    continue
                if not detections:
                    skipped.append(f"{path.name}: no face detected")
                    continue
                if len(detections) > 1:
                    warnings.append(
                        f"{path.name}: {len(detections)} faces found; used the "
                        "largest face for evaluation."
                    )
                face = self.engine.largest_face(detections)
                assert face is not None
                try:
                    feature = self.engine.extract_feature(image, face)
                except FaceEngineError as exc:
                    skipped.append(f"{path.name}: {exc}")
                    continue
                samples.append(
                    _ImageSample(
                        identity=identity,
                        path=path,
                        feature=feature,
                        detection_count=len(detections),
                    )
                )
            if len(paths) > max_images_per_identity:
                warnings.append(
                    f"{identity}: only the first {max_images_per_identity} images "
                    "were used."
                )

        if not samples:
            raise EvaluationError(
                "No usable face images were found. Check that faces are visible "
                "and the image files are valid."
            )

        genuine_pairs: list[EvaluationPair] = []
        impostor_pairs: list[EvaluationPair] = []

        by_identity: dict[str, list[_ImageSample]] = {}
        for sample in samples:
            by_identity.setdefault(sample.identity, []).append(sample)

        for identity, group in by_identity.items():
            for sample_a, sample_b in itertools.combinations(group, 2):
                similarity = self.engine.cosine_similarity(
                    sample_a.feature, sample_b.feature
                )
                genuine_pairs.append(
                    EvaluationPair(
                        identity_a=identity,
                        identity_b=identity,
                        file_a=str(sample_a.path),
                        file_b=str(sample_b.path),
                        genuine=True,
                        similarity=similarity,
                        accepted=similarity >= threshold,
                        correct=(similarity >= threshold),
                    )
                )
                if len(genuine_pairs) >= max_genuine_pairs:
                    warnings.append(
                        f"Genuine pairs capped at {max_genuine_pairs}."
                    )
                    break
            if len(genuine_pairs) >= max_genuine_pairs:
                break

        identity_names = sorted(by_identity)
        for identity_a, identity_b in itertools.combinations(identity_names, 2):
            for sample_a in by_identity[identity_a]:
                for sample_b in by_identity[identity_b]:
                    similarity = self.engine.cosine_similarity(
                        sample_a.feature, sample_b.feature
                    )
                    accepted = similarity >= threshold
                    impostor_pairs.append(
                        EvaluationPair(
                            identity_a=identity_a,
                            identity_b=identity_b,
                            file_a=str(sample_a.path),
                            file_b=str(sample_b.path),
                            genuine=False,
                            similarity=similarity,
                            accepted=accepted,
                            correct=not accepted,
                        )
                    )
                    if len(impostor_pairs) >= max_impostor_pairs:
                        warnings.append(
                            f"Impostor pairs capped at {max_impostor_pairs}."
                        )
                        break
                if len(impostor_pairs) >= max_impostor_pairs:
                    break
            if len(impostor_pairs) >= max_impostor_pairs:
                break

        pairs = tuple(genuine_pairs + impostor_pairs)
        if not genuine_pairs:
            warnings.append(
                "FRR is unavailable: the dataset needs at least two images of "
                "the same identity for genuine comparisons."
            )
        if not impostor_pairs:
            warnings.append(
                "FAR is unavailable: the dataset needs at least two identities "
                "for impostor comparisons."
            )
        if len(samples) < 10:
            warnings.append(
                "This is a very small sample. Treat the metrics as a pipeline "
                "smoke test, not as a biometric performance claim."
            )

        sweep = tuple(
            metrics_from_pairs(pairs, value / 100.0)
            for value in range(0, 101)
        )
        metrics = metrics_from_pairs(pairs, threshold)
        elapsed = time.perf_counter() - started
        return EvaluationReport(
            dataset_dir=str(Path(dataset_dir).resolve()),
            threshold=threshold,
            identities=tuple(identity_names),
            image_count=len(samples),
            pairs=pairs,
            metrics=metrics,
            sweep=sweep,
            warnings=tuple(warnings),
            skipped=tuple(skipped),
            generated_at=_utc_now(),
            elapsed_seconds=elapsed,
        )

    def write_csv(
        self,
        report: EvaluationReport,
        pairs_path: Path | str,
        summary_path: Path | str | None = None,
    ) -> tuple[Path, Path | None]:
        """Write pair-level and summary CSV files."""
        pairs_file = Path(pairs_path)
        pairs_file.parent.mkdir(parents=True, exist_ok=True)
        with pairs_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "identity_a",
                    "identity_b",
                    "file_a",
                    "file_b",
                    "pair_type",
                    "similarity",
                    "threshold",
                    "accepted",
                    "correct",
                ]
            )
            for pair in report.pairs:
                writer.writerow(
                    [
                        pair.identity_a,
                        pair.identity_b,
                        pair.file_a,
                        pair.file_b,
                        "genuine" if pair.genuine else "impostor",
                        f"{pair.similarity:.8f}",
                        f"{report.threshold:.4f}",
                        int(pair.accepted),
                        int(pair.correct),
                    ]
                )

        summary_file: Path | None = None
        if summary_path is not None:
            summary_file = Path(summary_path)
            summary_file.parent.mkdir(parents=True, exist_ok=True)
            metrics = report.metrics
            with summary_file.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["metric", "value"])
                writer.writerow(["dataset_dir", report.dataset_dir])
                writer.writerow(["generated_at", report.generated_at])
                writer.writerow(["threshold", f"{report.threshold:.4f}"])
                writer.writerow(["identities", len(report.identities)])
                writer.writerow(["usable_images", report.image_count])
                writer.writerow(["genuine_comparisons", report.genuine_count])
                writer.writerow(["impostor_comparisons", report.impostor_count])
                writer.writerow(["true_accepts", metrics.tp])
                writer.writerow(["false_rejects", metrics.fn])
                writer.writerow(["false_accepts", metrics.fp])
                writer.writerow(["true_rejects", metrics.tn])
                writer.writerow(["FAR", _format_optional(metrics.far)])
                writer.writerow(["FRR", _format_optional(metrics.frr)])
                writer.writerow(["accuracy", _format_optional(metrics.accuracy)])
                writer.writerow(["precision", _format_optional(metrics.precision)])
                writer.writerow(["recall", _format_optional(metrics.recall)])
                for warning in report.warnings:
                    writer.writerow(["warning", warning])
                for skipped in report.skipped:
                    writer.writerow(["skipped", skipped])
        return pairs_file, summary_file

    def render_chart(
        self, report: EvaluationReport, output_path: Path | str
    ) -> Path:
        """Render a FAR/FRR vs threshold chart without matplotlib."""
        from PIL import Image, ImageDraw

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        width, height = 920, 440
        margin_left, margin_right = 70, 30
        margin_top, margin_bottom = 50, 60
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom

        image = Image.new("RGB", (width, height), "#0b1220")
        draw = ImageDraw.Draw(image)
        draw.text((margin_left, 16), "FAR / FRR trade-off vs threshold", fill="#e8eefc")
        draw.text(
            (margin_left, height - 28),
            "Cosine threshold (0.00 to 1.00)",
            fill="#9fb0cc",
        )
        draw.text((12, margin_top - 28), "Rate", fill="#9fb0cc")

        def x_for(value: float) -> float:
            return margin_left + value * plot_w

        def y_for(value: float) -> float:
            return margin_top + (1.0 - value) * plot_h

        for step in range(0, 11):
            value = step / 10.0
            x = x_for(value)
            y = y_for(value)
            draw.line((x, margin_top, x, margin_top + plot_h), fill="#1b2a44")
            draw.line((margin_left, y, margin_left + plot_w, y), fill="#1b2a44")
            draw.text((x - 12, margin_top + plot_h + 8), f"{value:.1f}", fill="#8fa3c4")
            draw.text((margin_left - 42, y - 6), f"{value:.1f}", fill="#8fa3c4")

        def draw_series(
            values: list[float | None], colour: str, label: str, offset: int
        ) -> None:
            points = [
                (x_for(index / 100.0), y_for(value))
                for index, value in enumerate(values)
                if value is not None
            ]
            if len(points) >= 2:
                draw.line(points, fill=colour, width=3)
            if points:
                draw.ellipse(
                    (
                        points[-1][0] - 3,
                        points[-1][1] - 3,
                        points[-1][0] + 3,
                        points[-1][1] + 3,
                    ),
                    fill=colour,
                )
            draw.rectangle(
                (margin_left + plot_w - 160, margin_top + 8 + offset,
                 margin_left + plot_w - 146, margin_top + 20 + offset),
                fill=colour,
            )
            draw.text(
                (margin_left + plot_w - 138, margin_top + 7 + offset),
                label,
                fill="#dbe6fb",
            )

        far_values = [metric.far for metric in report.sweep]
        frr_values = [metric.frr for metric in report.sweep]
        draw_series(far_values, "#ff6b6b", "FAR", 0)
        draw_series(frr_values, "#4cc9f0", "FRR", 24)

        threshold_x = x_for(report.threshold)
        draw.line(
            (threshold_x, margin_top, threshold_x, margin_top + plot_h),
            fill="#ffd166",
            width=2,
        )
        draw.text(
            (min(threshold_x + 6, width - 140), margin_top + plot_h - 18),
            f"selected {report.threshold:.3f}",
            fill="#ffd166",
        )
        draw.text(
            (margin_left, margin_top + plot_h + 30),
            f"genuine={report.genuine_count}  impostor={report.impostor_count}  "
            f"images={report.image_count}",
            fill="#9fb0cc",
        )
        image.save(output)
        return output


def _format_optional(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"
