from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from core.app_context import AppContext
from tests.conftest import BIDEN


def _enroll_alice(context: AppContext, image_path: Path):
    return context.enrollment.enroll_from_image(
        "alice", "Alice Example", image_path, consent=True
    )


def test_enroll_from_image(context: AppContext, sample_image_path: Path) -> None:
    result = _enroll_alice(context, sample_image_path)
    assert result.success, result.message
    assert result.user_id == "alice"
    assert result.summary is not None
    assert context.database.count_users() == 1


def test_enroll_requires_consent(context: AppContext, sample_image_path: Path) -> None:
    result = context.enrollment.enroll_from_image(
        "alice", "Alice", sample_image_path, consent=False
    )
    assert not result.success
    assert "Consent" in result.message
    assert context.database.count_users() == 0


def test_enroll_rejects_duplicate(context: AppContext, sample_image_path: Path) -> None:
    assert _enroll_alice(context, sample_image_path).success
    result = _enroll_alice(context, sample_image_path)
    assert not result.success
    assert "already enrolled" in result.message


def test_enroll_rejects_invalid_id(context: AppContext, sample_image_path: Path) -> None:
    result = context.enrollment.enroll_from_image(
        "../alice", "Alice", sample_image_path, consent=True
    )
    assert not result.success


def test_enroll_rejects_no_face(context: AppContext, no_face_image: Path) -> None:
    result = context.enrollment.enroll_from_image(
        "alice", "Alice", no_face_image, consent=True
    )
    assert not result.success
    assert "No face" in result.message


def test_enroll_rejects_missing_file(context: AppContext, tmp_path: Path) -> None:
    result = context.enrollment.enroll_from_image(
        "alice", "Alice", tmp_path / "missing.jpg", consent=True
    )
    assert not result.success


def test_enroll_from_frame(context: AppContext, sample_image_path: Path) -> None:
    frame = cv2.imread(str(sample_image_path))
    assert frame is not None
    result = context.enrollment.enroll_from_frame(
        "frame_user", "Frame User", frame, consent=True
    )
    assert result.success, result.message


def test_verify_same_person(context: AppContext, sample_image_path: Path) -> None:
    assert _enroll_alice(context, sample_image_path).success
    result = context.verification.verify_image("alice", sample_image_path)
    assert result.success, result.message
    assert result.verified
    assert result.similarity is not None
    assert result.similarity >= result.threshold
    assert result.decision == "VERIFIED"


def test_verify_different_person(context: AppContext, sample_image_path: Path) -> None:
    assert _enroll_alice(context, sample_image_path).success
    result = context.verification.verify_image("alice", BIDEN)
    assert result.success, result.message
    assert not result.verified
    assert result.similarity is not None
    assert result.similarity < result.threshold
    assert result.decision == "NOT VERIFIED"


def test_verify_unknown_user(context: AppContext, sample_image_path: Path) -> None:
    result = context.verification.verify_image("nobody", sample_image_path)
    assert not result.success
    assert not result.verified


def test_verify_no_face(
    context: AppContext, sample_image_path: Path, no_face_image: Path
) -> None:
    assert _enroll_alice(context, sample_image_path).success
    result = context.verification.verify_image("alice", no_face_image)
    assert not result.success
    assert "No face" in result.message


def test_verify_after_delete(context: AppContext, sample_image_path: Path) -> None:
    assert _enroll_alice(context, sample_image_path).success
    assert context.database.delete_user("alice")
    result = context.verification.verify_image("alice", sample_image_path)
    assert not result.success


def test_threshold_update_is_shared(context: AppContext) -> None:
    value = context.set_threshold(0.5, persist=False)
    assert value == pytest.approx(0.5)
    assert context.matcher.threshold == pytest.approx(0.5)
    assert context.enrollment.threshold == pytest.approx(0.5)
    assert context.verification.threshold == pytest.approx(0.5)


def test_threshold_persists(context: AppContext, tmp_path: Path) -> None:
    from config import settings

    context.set_threshold(0.44)
    assert settings.load_threshold(tmp_path / "settings.json") == pytest.approx(0.44)
