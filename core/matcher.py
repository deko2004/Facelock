"""Cosine-similarity matching and threshold decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from config import settings
from core.face_engine import FaceEngine, FaceEngineError

SCORE_NOTE = (
    "Cosine similarity is a geometric score, not a probability that two faces "
    "belong to the same person."
)


@dataclass(frozen=True)
class MatchResult:
    similarity: float
    threshold: float
    matched: bool
    method: str = "cosine similarity (SFace embeddings)"

    @property
    def decision(self) -> str:
        return "MATCH" if self.matched else "NO MATCH"

    @property
    def explanation(self) -> str:
        comparator = ">=" if self.matched else "<"
        return (
            f"similarity {self.similarity:.4f} {comparator} "
            f"threshold {self.threshold:.3f}"
        )


@dataclass(frozen=True)
class EnrolledIdentity:
    """An enrolled user's display name and decrypted template."""

    user_id: str
    display_name: str
    embedding: np.ndarray


@dataclass(frozen=True)
class Identification:
    """Best enrolled match for one probe, if it passes the threshold."""

    user_id: str | None
    display_name: str | None
    similarity: float | None
    matched: bool


def decide(similarity: float, threshold: float) -> bool:
    """Apply the configured acceptance threshold."""
    if not np.isfinite(similarity):
        raise ValueError("Similarity must be a finite number.")
    return float(similarity) >= float(threshold)


def identify(
    engine: FaceEngine,
    probe: np.ndarray,
    identities: Sequence[EnrolledIdentity],
    threshold: float,
) -> Identification:
    """Return the closest enrolled identity that passes *threshold*.

    This is a 1:N identification helper, not a probability estimate. If no
    identity passes, the best score is still returned for diagnostics, but the
    name is deliberately withheld.
    """
    if not identities:
        return Identification(None, None, None, False)

    limit = settings.clamp_threshold(threshold)
    best_identity: EnrolledIdentity | None = None
    best_score = -np.inf
    for identity in identities:
        try:
            score = engine.cosine_similarity(probe, identity.embedding)
        except FaceEngineError:
            continue
        if score > best_score:
            best_score = score
            best_identity = identity

    if best_identity is None:
        return Identification(None, None, None, False)

    matched = decide(best_score, limit)
    return Identification(
        user_id=best_identity.user_id if matched else None,
        display_name=best_identity.display_name if matched else None,
        similarity=float(best_score),
        matched=matched,
    )


class FaceMatcher:
    """Compare SFace embeddings using cosine similarity."""

    def __init__(
        self, engine: FaceEngine, threshold: float = settings.DEFAULT_THRESHOLD
    ) -> None:
        self.engine = engine
        self.threshold = settings.clamp_threshold(threshold)

    def set_threshold(self, threshold: float) -> None:
        self.threshold = settings.clamp_threshold(threshold)

    def compare_features(
        self, feature_a: np.ndarray, feature_b: np.ndarray
    ) -> MatchResult:
        similarity = self.engine.cosine_similarity(feature_a, feature_b)
        return MatchResult(
            similarity=similarity,
            threshold=self.threshold,
            matched=decide(similarity, self.threshold),
        )
