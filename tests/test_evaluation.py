from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from core.evaluation import (
    EvaluationError,
    EvaluationPair,
    EvaluationService,
    metrics_from_pairs,
    scan_dataset,
)
from tests.conftest import EVAL_DEMO


def _pair(genuine: bool, similarity: float) -> EvaluationPair:
    return EvaluationPair(
        identity_a="a",
        identity_b="a" if genuine else "b",
        file_a="a.jpg",
        file_b="b.jpg",
        genuine=genuine,
        similarity=similarity,
        accepted=similarity >= 0.363,
        correct=(similarity >= 0.363) == genuine,
    )


def test_metrics_from_pairs() -> None:
    pairs = [
        _pair(True, 0.9),
        _pair(True, 0.2),
        _pair(False, 0.5),
        _pair(False, 0.1),
    ]
    metrics = metrics_from_pairs(pairs, 0.363)
    assert (metrics.tp, metrics.fn, metrics.fp, metrics.tn) == (1, 1, 1, 1)
    assert metrics.far == pytest.approx(0.5)
    assert metrics.frr == pytest.approx(0.5)
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)


def test_metrics_handles_empty_classes() -> None:
    metrics = metrics_from_pairs([_pair(True, 0.9)], 0.363)
    assert metrics.far is None
    assert metrics.frr == pytest.approx(0.0)


def test_scan_missing_dataset(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError):
        scan_dataset(tmp_path / "missing")


def test_scan_empty_dataset(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError):
        scan_dataset(tmp_path)


def test_scan_eval_demo() -> None:
    identities = scan_dataset(EVAL_DEMO)
    assert len(identities) == 4
    assert len(identities["obama"]) == 2


def test_evaluate_eval_demo(engine) -> None:
    report = EvaluationService(engine).evaluate(EVAL_DEMO, threshold=0.363)
    assert report.image_count == 5
    assert report.genuine_count == 1
    assert report.impostor_count == 9
    assert report.metrics.total == 10
    assert report.metrics.far is not None
    assert report.metrics.frr is not None
    assert report.metrics.far == pytest.approx(0.0)
    assert report.metrics.frr == pytest.approx(0.0)
    assert len(report.sweep) == 101
    assert report.warnings


def test_evaluate_without_genuine_pairs(
    engine, tmp_path: Path, sample_image_path: Path
) -> None:
    identity_dir = tmp_path / "solo"
    identity_dir.mkdir()
    shutil.copy(sample_image_path, identity_dir / "one.jpg")
    report = EvaluationService(engine).evaluate(tmp_path, threshold=0.363)
    assert report.genuine_count == 0
    assert report.impostor_count == 0
    assert report.metrics.far is None
    assert report.metrics.frr is None
    assert any("FRR is unavailable" in warning for warning in report.warnings)
    assert any("FAR is unavailable" in warning for warning in report.warnings)


def test_export_csv_and_chart(engine, tmp_path: Path) -> None:
    service = EvaluationService(engine)
    report = service.evaluate(EVAL_DEMO, threshold=0.363)
    pairs_csv, summary_csv = service.write_csv(
        report, tmp_path / "pairs.csv", tmp_path / "summary.csv"
    )
    chart = service.render_chart(report, tmp_path / "chart.png")
    assert pairs_csv.exists() and pairs_csv.stat().st_size > 0
    assert summary_csv is not None and summary_csv.exists()
    assert chart.exists() and chart.stat().st_size > 0
    assert "pair_type" in pairs_csv.read_text(encoding="utf-8")
    assert "FAR" in summary_csv.read_text(encoding="utf-8")
