"""Application composition root: build and share the core services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import settings
from core.anti_spoofing import AntiSpoofingService
from core.camera_manager import CameraManager
from core.database import FaceDatabase
from core.enrollment import EnrollmentService
from core.evaluation import EvaluationService
from core.face_engine import FaceEngine
from core.matcher import FaceMatcher
from core.security import TemplateCipher
from core.verification import VerificationService


@dataclass
class AppContext:
    engine: FaceEngine
    database: FaceDatabase
    cipher: TemplateCipher
    matcher: FaceMatcher
    enrollment: EnrollmentService
    verification: VerificationService
    evaluation: EvaluationService
    camera: CameraManager
    anti_spoofing: AntiSpoofingService
    threshold: float
    settings_path: Path | None

    @classmethod
    def create(
        cls,
        db_path: Path | str | None = None,
        key_path: Path | str | None = None,
        settings_path: Path | str | None = None,
        threshold: float | None = None,
        detector_path: Path | str | None = None,
        recognizer_path: Path | str | None = None,
        pad_model_path: Path | str | None = None,
    ) -> "AppContext":
        settings.ensure_directories()
        cipher = TemplateCipher(key_path or settings.TEMPLATE_KEY_FILE)
        database = FaceDatabase(db_path or settings.DATABASE_FILE, cipher)
        engine = FaceEngine(
            detector_path=detector_path or settings.YUNET_MODEL,
            recognizer_path=recognizer_path or settings.SFACE_MODEL,
        )
        if threshold is None:
            threshold = settings.load_threshold(settings_path)
        threshold = settings.clamp_threshold(threshold)
        matcher = FaceMatcher(engine, threshold)
        enrollment = EnrollmentService(database, engine, threshold)
        verification = VerificationService(database, engine, matcher, threshold)
        evaluation = EvaluationService(engine)
        camera = CameraManager()
        anti_spoofing = AntiSpoofingService(model_path=pad_model_path)
        return cls(
            engine=engine,
            database=database,
            cipher=cipher,
            matcher=matcher,
            enrollment=enrollment,
            verification=verification,
            evaluation=evaluation,
            camera=camera,
            anti_spoofing=anti_spoofing,
            threshold=threshold,
            settings_path=(
                Path(settings_path)
                if settings_path is not None
                else settings.APP_SETTINGS_FILE
            ),
        )

    def set_threshold(self, value: float, persist: bool = True) -> float:
        """Update the shared threshold and optionally persist it."""
        self.threshold = settings.clamp_threshold(value)
        self.matcher.set_threshold(self.threshold)
        self.enrollment.set_threshold(self.threshold)
        self.verification.set_threshold(self.threshold)
        if persist and self.settings_path is not None:
            settings.save_threshold(self.threshold, self.settings_path)
        return self.threshold

    def shutdown(self) -> None:
        self.camera.release()
        self.anti_spoofing.close()
