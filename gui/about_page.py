"""About page with objectives, technology and academic disclaimers."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from config import settings
from gui.styles import COLORS, font
from gui.widgets import BasePage, Card, PageHeader, ScrollableFrame, bullet_text


class AboutPage(BasePage):
    def __init__(self, parent: tk.Misc, controller) -> None:
        super().__init__(parent, controller)
        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True)
        body = scroll.inner
        body.grid_columnconfigure(0, weight=1)

        PageHeader(
            body,
            "About the Project",
            "Biometric Authentication: Face Recognition and Verification System",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 14))

        overview = Card(body)
        overview.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(overview, text="Project objectives", style="CardTitle.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            overview,
            text=(
                "This Master's-level Cybersecurity demonstration implements a "
                "complete, local-only face recognition and 1:1 verification "
                "workflow using OpenCV YuNet for detection and SFace for feature "
                "extraction. It covers image comparison, live webcam identification "
                "and verification, encrypted enrolment, identity verification, "
                "evaluation metrics, and an explicit analysis of spoofing and "
                "privacy limitations."
            ),
            style="Card.TLabel",
            wraplength=1100,
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        tech = Card(body)
        tech.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(tech, text="Technology stack", style="CardTitle.TLabel").pack(
            anchor="w"
        )
        bullet_text(
            tech,
            [
                "Python 3.12 and a local virtual environment.",
                "OpenCV 5.0.0 (contrib) with YuNet and SFace ONNX models.",
                "NumPy for vector operations and cosine similarity.",
                "Tkinter/ttk for the desktop interface, with worker threads for camera and heavy processing.",
                "SQLite for local metadata and encrypted templates.",
                "cryptography/Fernet for encryption at rest.",
                "Pillow for image previews and the evaluation chart.",
                "pytest for automated tests.",
            ],
            wraplength=1100,
        ).pack(anchor="w", pady=(8, 0))

        models = Card(body)
        models.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(models, text="Models and matching", style="CardTitle.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            models,
            text=(
                f"Detector: {settings.DETECTOR_LABEL} (ONNX, OpenCV Zoo). "
                f"Recognizer: {settings.MODEL_LABEL}. The recognizer returns a "
                "128-dimensional embedding. Matching uses cosine similarity: "
                "sim(a,b) = (a . b) / (||a|| ||b||). The decision is "
                "sim >= threshold. The threshold is a configurable demonstration "
                "value and must be calibrated on validation data. The score is not "
                "a probability of identity."
            ),
            style="Card.TLabel",
            wraplength=1100,
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        facelock = Card(body)
        facelock.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(
            facelock, text="Why not the Facelock package?", style="CardTitle.TLabel"
        ).pack(anchor="w")
        ttk.Label(
            facelock,
            text=(
                "The proposal mentioned Facelock. The published PyPI package "
                "(0.1.2, April 2025, MIT) is a small InsightFace-based wrapper "
                "whose documented workflow is a webcam authentication check that "
                "returns a boolean. It does not expose image-to-image comparison, "
                "multi-user templates, configurable thresholds or a GUI, and its "
                "opencv-python-headless dependency conflicts with the existing "
                "opencv-contrib-python environment. The OpenCV YuNet + SFace "
                "pipeline is therefore used and documented in "
                "docs/tool_selection.md. No compatibility claim is made beyond the "
                "evidence recorded there."
            ),
            style="Card.TLabel",
            wraplength=1100,
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        disclaimer = Card(body)
        disclaimer.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(disclaimer, text="Academic disclaimer", style="CardTitle.TLabel").pack(
            anchor="w"
        )
        tk.Label(
            disclaimer,
            text=(
                "This is an academic demonstration. It is not a production identity "
                "verification system. It provides no liveness/PAD protection, no "
                "certified bias evaluation, no secure hardware key storage and no "
                "operational threshold calibration. Do not use it to protect real "
                "accounts, buildings or sensitive data."
            ),
            bg=COLORS["danger_dark"],
            fg=COLORS["danger"],
            font=font(9, "bold"),
            wraplength=1050,
            justify="left",
            padx=10,
            pady=8,
        ).pack(anchor="w", pady=(8, 0))

        references = Card(body)
        references.grid(row=6, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(references, text="References", style="CardTitle.TLabel").pack(
            anchor="w"
        )
        bullet_text(
            references,
            [
                "OpenCV FaceDetectorYN / YuNet documentation and OpenCV Zoo model card.",
                "OpenCV FaceRecognizerSF / SFace documentation and model card.",
                "OpenCV Zoo: face_detection_yunet_2023mar.onnx and face_recognition_sface_2021dec.onnx.",
                "NIST FRVT / FATE PAD resources for presentation-attack terminology.",
                "cryptography.io Fernet documentation.",
                "Python sqlite3 and tkinter documentation.",
                "Facelock PyPI page and GitHub repository (Ranuja01/facelock).",
            ],
            wraplength=1100,
        ).pack(anchor="w", pady=(8, 0))
