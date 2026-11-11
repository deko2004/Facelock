"""User enrolment: validate, detect, extract and store an encrypted template."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import settings
from core.database import (
    DatabaseError,
    DuplicateUserError,
    FaceDatabase,
    UserSummary,
)
from core.face_engine import FaceEngine, FaceEngineError
from core.image_processor import (
    ImageLoadError,
    QualityReport,
    assess_quality,
    load_image,
)
from core.security import (
    SecurityError,
    ValidationError,
    validate_display_name,
    validate_user_id,
)


@dataclass(frozen=True)
class EnrollmentResult:
    success: bool
    message: str
    user_id: str = ""
    display_name: str = ""
    quality: QualityReport | None = None
    detection_count: int = 0
    face_index: int = 0
    summary: UserSummary | None = None


class EnrollmentService:
    """Create encrypted SFace templates for new users."""

    def __init__(
        self,
        database: FaceDatabase,
        engine: FaceEngine,
        threshold: float = settings.DEFAULT_THRESHOLD,
    ) -> None:
        self.database = database
        self.engine = engine
        self.threshold = settings.clamp_threshold(threshold)

    def set_threshold(self, threshold: float) -> None:
        self.threshold = settings.clamp_threshold(threshold)

    def enroll_from_image(
        self,
        user_id: str,
        display_name: str,
        image_path: Path | str,
        face_index: int = 0,
        consent: bool = False,
    ) -> EnrollmentResult:
        if not consent:
            return EnrollmentResult(
                False,
                "Consent is required before a face can be enrolled.",
                user_id=user_id,
            )
        try:
            image = load_image(image_path)
        except ImageLoadError as exc:
            return EnrollmentResult(False, str(exc), user_id=user_id)
        return self._enroll(
            user_id=user_id,
            display_name=display_name,
            image=image,
            face_index=face_index,
            consent=consent,
            source="upload",
        )

    def enroll_from_frame(
        self,
        user_id: str,
        display_name: str,
        frame: np.ndarray,
        face_index: int = 0,
        consent: bool = False,
    ) -> EnrollmentResult:
        if not consent:
            return EnrollmentResult(
                False,
                "Consent is required before a face can be enrolled.",
                user_id=user_id,
            )
        return self._enroll(
            user_id=user_id,
            display_name=display_name,
            image=frame,
            face_index=face_index,
            consent=consent,
            source="webcam",
        )

    def _enroll(
        self,
        user_id: str,
        display_name: str,
        image: np.ndarray,
        face_index: int,
        consent: bool,
        source: str,
    ) -> EnrollmentResult:
        try:
            cleaned_id = validate_user_id(user_id)
            cleaned_name = validate_display_name(display_name)
        except ValidationError as exc:
            return EnrollmentResult(False, str(exc), user_id=user_id)

        if self.database.user_exists(cleaned_id):
            return EnrollmentResult(
                False,
                f"User ID '{cleaned_id}' is already enrolled. Choose another ID.",
                user_id=cleaned_id,
                display_name=cleaned_name,
            )

        try:
            detections = self.engine.detect(image)
        except FaceEngineError as exc:
            return EnrollmentResult(False, str(exc), user_id=cleaned_id)

        if not detections:
            return EnrollmentResult(
                False,
                "No face was detected. Use a clear, front-facing photograph.",
                user_id=cleaned_id,
                display_name=cleaned_name,
                detection_count=0,
            )
        if face_index < 0 or face_index >= len(detections):
            return EnrollmentResult(
                False,
                f"Face {face_index + 1} does not exist; "
                f"{len(detections)} face(s) were detected.",
                user_id=cleaned_id,
                display_name=cleaned_name,
                detection_count=len(detections),
            )

        detection = detections[face_index]
        quality = assess_quality(image, detection)
        if not quality.passed:
            return EnrollmentResult(
                False,
                quality.summary,
                user_id=cleaned_id,
                display_name=cleaned_name,
                quality=quality,
                detection_count=len(detections),
                face_index=face_index,
            )

        try:
            feature = self.engine.extract_feature(image, detection)
            summary = self.database.add_user(
                user_id=cleaned_id,
                display_name=cleaned_name,
                embedding=feature,
                model=settings.MODEL_LABEL,
                threshold=self.threshold,
            )
        except DuplicateUserError as exc:
            return EnrollmentResult(False, str(exc), user_id=cleaned_id)
        except (FaceEngineError, DatabaseError, SecurityError) as exc:
            return EnrollmentResult(
                False,
                f"Enrolment failed: {exc}",
                user_id=cleaned_id,
                display_name=cleaned_name,
                detection_count=len(detections),
                face_index=face_index,
            )

        self.database.log_event(
            "enrollment_source",
            cleaned_id,
            f"Template created from {source} image",
        )
        return EnrollmentResult(
            True,
            f"User '{cleaned_name}' enrolled successfully. "
            "Only an encrypted template was stored.",
            user_id=cleaned_id,
            display_name=cleaned_name,
            quality=quality,
            detection_count=len(detections),
            face_index=face_index,
            summary=summary,
        )
