"""1:1 identity verification against an enrolled encrypted template."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import settings
from core.database import DatabaseError, FaceDatabase, UserNotFoundError
from core.face_engine import FaceEngine, FaceEngineError
from core.image_processor import (
    ImageLoadError,
    QualityReport,
    assess_quality,
    load_image,
)
from core.matcher import SCORE_NOTE, FaceMatcher
from core.security import SecurityError, ValidationError, validate_user_id


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    message: str
    user_id: str = ""
    display_name: str = ""
    similarity: float | None = None
    threshold: float = settings.DEFAULT_THRESHOLD
    verified: bool = False
    detection_count: int = 0
    face_index: int = 0
    quality: QualityReport | None = None
    score_note: str = SCORE_NOTE

    @property
    def decision(self) -> str:
        if not self.success:
            return "NOT VERIFIED"
        return "VERIFIED" if self.verified else "NOT VERIFIED"


class VerificationService:
    """Compare a probe face with one enrolled user's template (1:1)."""

    def __init__(
        self,
        database: FaceDatabase,
        engine: FaceEngine,
        matcher: FaceMatcher,
        threshold: float = settings.DEFAULT_THRESHOLD,
    ) -> None:
        self.database = database
        self.engine = engine
        self.matcher = matcher
        self.threshold = settings.clamp_threshold(threshold)
        self.matcher.set_threshold(self.threshold)

    def set_threshold(self, threshold: float) -> None:
        self.threshold = settings.clamp_threshold(threshold)
        self.matcher.set_threshold(self.threshold)

    def verify_image(
        self,
        user_id: str,
        image_path: Path | str,
        face_index: int = 0,
    ) -> VerificationResult:
        try:
            image = load_image(image_path)
        except ImageLoadError as exc:
            return VerificationResult(False, str(exc), user_id=user_id)
        return self._verify(user_id, image, face_index, source="upload")

    def verify_frame(
        self,
        user_id: str,
        frame: np.ndarray,
        face_index: int = 0,
    ) -> VerificationResult:
        return self._verify(user_id, frame, face_index, source="webcam")

    def _verify(
        self,
        user_id: str,
        image: np.ndarray,
        face_index: int,
        source: str,
    ) -> VerificationResult:
        try:
            cleaned_id = validate_user_id(user_id)
        except ValidationError as exc:
            return VerificationResult(False, str(exc), user_id=user_id)

        try:
            user = self.database.get_user(cleaned_id)
        except UserNotFoundError as exc:
            return VerificationResult(False, str(exc), user_id=cleaned_id)
        except (DatabaseError, SecurityError) as exc:
            return VerificationResult(
                False, f"Cannot load template: {exc}", user_id=cleaned_id
            )

        try:
            detections = self.engine.detect(image)
        except FaceEngineError as exc:
            return VerificationResult(False, str(exc), user_id=cleaned_id)

        if not detections:
            self.database.log_event(
                "verification_no_face", cleaned_id, f"No face in {source} probe"
            )
            return VerificationResult(
                False,
                "No face was detected in the probe image.",
                user_id=cleaned_id,
                display_name=user.display_name,
                threshold=self.threshold,
                detection_count=0,
            )
        if face_index < 0 or face_index >= len(detections):
            return VerificationResult(
                False,
                f"Face {face_index + 1} does not exist; "
                f"{len(detections)} face(s) were detected.",
                user_id=cleaned_id,
                display_name=user.display_name,
                threshold=self.threshold,
                detection_count=len(detections),
            )

        detection = detections[face_index]
        quality = assess_quality(image, detection)
        try:
            probe = self.engine.extract_feature(image, detection)
            match = self.matcher.compare_features(probe, user.embedding)
        except FaceEngineError as exc:
            return VerificationResult(
                False,
                f"Verification failed: {exc}",
                user_id=cleaned_id,
                display_name=user.display_name,
                threshold=self.threshold,
                detection_count=len(detections),
                face_index=face_index,
                quality=quality,
            )

        note = ""
        if not quality.passed:
            note = " Low-quality probe image: " + "; ".join(quality.reasons)
        message = (
            f"{match.decision}: cosine similarity {match.similarity:.4f} "
            f"against threshold {match.threshold:.3f}.{note}"
        )
        self.database.log_event(
            "verification_verified" if match.matched else "verification_rejected",
            cleaned_id,
            f"1:1 {source} comparison; score={match.similarity:.4f}; "
            f"threshold={match.threshold:.3f}",
        )
        return VerificationResult(
            True,
            message,
            user_id=cleaned_id,
            display_name=user.display_name,
            similarity=match.similarity,
            threshold=match.threshold,
            verified=match.matched,
            detection_count=len(detections),
            face_index=face_index,
            quality=quality,
        )
