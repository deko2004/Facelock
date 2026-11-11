from __future__ import annotations

import numpy as np
import pytest

from core.face_engine import FaceEngine, FaceEngineError
from core.matcher import (
    EnrolledIdentity,
    FaceMatcher,
    Identification,
    MatchResult,
    decide,
    identify,
)


def test_decide_threshold() -> None:
    assert decide(0.5, 0.363)
    assert decide(0.363, 0.363)
    assert not decide(0.3629, 0.363)


def test_cosine_similarity_known_vectors() -> None:
    a = np.array([[1.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0]])
    c = np.array([[0.0, 1.0, 0.0]])
    assert FaceEngine.cosine_similarity(a, b) == pytest.approx(1.0)
    assert FaceEngine.cosine_similarity(a, c) == pytest.approx(0.0)
    assert FaceEngine.cosine_similarity(a, -b) == pytest.approx(-1.0)


def test_cosine_similarity_shape_mismatch() -> None:
    with pytest.raises(FaceEngineError):
        FaceEngine.cosine_similarity(np.ones((1, 3)), np.ones((1, 4)))


def test_cosine_similarity_zero_vector() -> None:
    with pytest.raises(FaceEngineError):
        FaceEngine.cosine_similarity(np.zeros((1, 3)), np.ones((1, 3)))


def test_match_result_properties() -> None:
    result = MatchResult(similarity=0.5, threshold=0.363, matched=True)
    assert result.decision == "MATCH"
    assert "0.5000" in result.explanation
    rejected = MatchResult(similarity=0.1, threshold=0.363, matched=False)
    assert rejected.decision == "NO MATCH"


def test_matcher_clamps_threshold(engine: FaceEngine) -> None:
    matcher = FaceMatcher(engine, threshold=2.0)
    assert matcher.threshold == 1.0
    matcher.set_threshold(-1.0)
    assert matcher.threshold == 0.0


def test_matcher_compare_features(engine: FaceEngine) -> None:
    a = np.ones((1, 128), dtype=np.float32)
    b = np.ones((1, 128), dtype=np.float32)
    matcher = FaceMatcher(engine, threshold=0.5)
    result = matcher.compare_features(a, b)
    assert result.matched
    assert result.similarity == pytest.approx(1.0)


class _CosineEngine:
    cosine_similarity = staticmethod(FaceEngine.cosine_similarity)


def test_identify_returns_best_match_above_threshold() -> None:
    alice = EnrolledIdentity(
        "1", "Alice", np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    )
    bob = EnrolledIdentity(
        "2", "Bob", np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    )
    probe = np.array([[0.9, 0.1, 0.0]], dtype=np.float32)
    result = identify(_CosineEngine, probe, [alice, bob], 0.5)
    assert isinstance(result, Identification)
    assert result.matched
    assert result.user_id == "1"
    assert result.display_name == "Alice"
    assert result.similarity == pytest.approx(0.9938837, rel=1e-5)


def test_identify_withholds_name_below_threshold() -> None:
    alice = EnrolledIdentity(
        "1", "Alice", np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    )
    probe = np.array([[0.5, 0.5, 0.7071]], dtype=np.float32)
    result = identify(_CosineEngine, probe, [alice], 0.9)
    assert not result.matched
    assert result.user_id is None
    assert result.display_name is None
    assert result.similarity is not None
    assert result.similarity < 0.9


def test_identify_without_enrolled_users() -> None:
    result = identify(
        _CosineEngine,
        np.ones((1, 3), dtype=np.float32),
        [],
        0.363,
    )
    assert not result.matched
    assert result.similarity is None
