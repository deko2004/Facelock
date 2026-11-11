"""Image loading, quality checks and annotation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from config import settings
from core.face_engine import FaceDetection


class ImageLoadError(RuntimeError):
    """Raised when an image cannot be read or fails validation."""


@dataclass(frozen=True)
class QualityReport:
    passed: bool
    face_width: int
    face_height: int
    blur_variance: float
    brightness: float
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def summary(self) -> str:
        if self.passed:
            return (
                f"Quality OK: {self.face_width}x{self.face_height}px, "
                f"sharpness {self.blur_variance:.1f}, brightness {self.brightness:.1f}"
            )
        return "Quality rejected: " + "; ".join(self.reasons)


def load_image(
    path: Path | str, max_dimension: int = settings.MAX_IMAGE_DIMENSION
) -> np.ndarray:
    """Load an image from disk with extension, existence and size validation."""
    image_path = Path(path)
    if not image_path.exists():
        raise ImageLoadError(f"Image not found: {image_path}")
    if not image_path.is_file():
        raise ImageLoadError(f"Not a regular file: {image_path}")
    if image_path.suffix.lower() not in settings.ALLOWED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(settings.ALLOWED_IMAGE_EXTENSIONS))
        raise ImageLoadError(f"Unsupported image type. Allowed: {allowed}")

    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ImageLoadError(f"OpenCV could not read the image: {exc}") from exc
    if image is None or image.size == 0:
        raise ImageLoadError(
            f"Could not decode image: {image_path.name}. The file may be corrupt."
        )
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageLoadError("Image must be an 8-bit colour image.")

    height, width = image.shape[:2]
    if height < 20 or width < 20:
        raise ImageLoadError("Image is too small to analyse.")
    longest = max(height, width)
    if longest > max_dimension:
        scale = max_dimension / float(longest)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image


def save_image(path: Path | str, image: np.ndarray) -> Path:
    """Save an annotated image and return its path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise ImageLoadError(f"Could not write image to {output}")
    return output


def assess_quality(
    image: np.ndarray,
    detection: FaceDetection,
    min_face_size: int = settings.MIN_FACE_SIZE_PX,
    min_blur_variance: float = settings.MIN_BLUR_VARIANCE,
    min_brightness: float = settings.MIN_BRIGHTNESS,
    max_brightness: float = settings.MAX_BRIGHTNESS,
) -> QualityReport:
    """Check that a detected face is large, sharp and reasonably exposed."""
    height, width = image.shape[:2]
    x = max(0, detection.x)
    y = max(0, detection.y)
    w = max(0, min(detection.w, width - x))
    h = max(0, min(detection.h, height - y))
    if w == 0 or h == 0:
        return QualityReport(
            False, w, h, 0.0, 0.0, ("Face bounding box is outside the image.",)
        )

    crop = image[y : y + h, x : x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    reasons: list[str] = []
    if w < min_face_size or h < min_face_size:
        reasons.append(
            f"face is smaller than {min_face_size}px (got {w}x{h}px)"
        )
    if blur_variance < min_blur_variance:
        reasons.append(
            f"image looks blurry (sharpness {blur_variance:.1f} < "
            f"{min_blur_variance:.1f})"
        )
    if brightness < min_brightness:
        reasons.append(f"face is too dark (brightness {brightness:.1f})")
    if brightness > max_brightness:
        reasons.append(f"face is too bright (brightness {brightness:.1f})")

    return QualityReport(
        passed=not reasons,
        face_width=w,
        face_height=h,
        blur_variance=blur_variance,
        brightness=brightness,
        reasons=tuple(reasons),
    )


def draw_faces(
    image: np.ndarray,
    detections: list[FaceDetection],
    selected_index: int | None = None,
    status_text: str | None = None,
    labels: Sequence[str | None] | None = None,
) -> np.ndarray:
    """Return an annotated copy of *image* with bounding boxes and labels.

    ``labels`` is optional and indexed by ``FaceDetection.index``. An empty or
    missing entry falls back to the default ``Face N score`` label.
    """
    annotated = image.copy()
    for detection in detections:
        is_selected = selected_index is None or detection.index == selected_index
        colour = (0, 210, 140) if is_selected else (0, 170, 255)
        thickness = 3 if is_selected else 2
        x, y, w, h = detection.bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, thickness)
        custom_label = None
        if labels is not None and 0 <= detection.index < len(labels):
            custom_label = labels[detection.index]
        label = custom_label or f"Face {detection.index + 1}  {detection.score:.2f}"
        text_y = max(22, y - 8)
        cv2.putText(
            annotated,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
            cv2.LINE_AA,
        )
    if status_text:
        cv2.putText(
            annotated,
            status_text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def draw_pad_status(image: np.ndarray, text: str, status: str) -> np.ndarray:
    """Draw a compact anti-spoofing banner across the top of a frame."""
    colours = {
        "LIVE": (40, 200, 120),
        "SPOOF": (60, 60, 255),
        "ERROR": (60, 60, 255),
        "ANALYZING": (0, 180, 255),
        "UNCERTAIN": (0, 210, 255),
        "NO_FACE": (0, 170, 255),
        "LOADING": (0, 180, 255),
    }
    colour = colours.get(status, (255, 255, 255))
    annotated = image.copy()
    cv2.rectangle(
        annotated,
        (0, 0),
        (annotated.shape[1], 40),
        (12, 20, 36),
        thickness=-1,
    )
    cv2.putText(
        annotated,
        text,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        colour,
        2,
        cv2.LINE_AA,
    )
    return annotated


def draw_comparison_result(
    image: np.ndarray,
    score: float,
    threshold: float,
    matched: bool,
) -> np.ndarray:
    """Overlay a comparison decision on an image for export."""
    annotated = image.copy()
    colour = (0, 210, 140) if matched else (60, 80, 255)
    decision = "MATCH" if matched else "NO MATCH"
    text = f"{decision} | cosine {score:.4f} | threshold {threshold:.3f}"
    cv2.rectangle(
        annotated,
        (0, 0),
        (annotated.shape[1], 44),
        (12, 20, 36),
        thickness=-1,
    )
    cv2.putText(
        annotated,
        text,
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        colour,
        2,
        cv2.LINE_AA,
    )
    return annotated
