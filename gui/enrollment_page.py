"""Module C: user enrolment from an uploaded image or a webcam capture."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from core.enrollment import EnrollmentResult
from core.face_engine import FaceDetection
from core.image_processor import assess_quality, draw_faces, load_image
from gui.camera_dialog import CameraCaptureDialog
from gui.styles import COLORS, font
from gui.widgets import BasePage, Card, ImagePanel, PageHeader


class EnrollmentPage(BasePage):
    def __init__(self, parent: tk.Misc, controller) -> None:
        super().__init__(parent, controller)
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(1, weight=1)
        self.image = None
        self.detections: list[FaceDetection] = []
        self.selected_index = 0
        self.source_path: str | None = None
        self.source_kind = "upload"

        PageHeader(
            self,
            "Register User",
            "Create an encrypted SFace template. Only the embedding is stored; "
            "the original photograph is never written to the database.",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))

        form = Card(self)
        form.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        form.columnconfigure(0, weight=1)
        ttk.Label(form, text="Enrolment details", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 12)
        )

        ttk.Label(form, text="User ID", style="Card.TLabel").grid(
            row=1, column=0, sticky="w"
        )
        self.user_id_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.user_id_var).grid(
            row=2, column=0, sticky="ew", pady=(4, 10)
        )
        ttk.Label(
            form,
            text="Letters, numbers, dot, dash or underscore; 1-64 characters.",
            style="CardMuted.TLabel",
        ).grid(row=3, column=0, sticky="w", pady=(0, 10))

        ttk.Label(form, text="Display name", style="Card.TLabel").grid(
            row=4, column=0, sticky="w"
        )
        self.display_name_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.display_name_var).grid(
            row=5, column=0, sticky="ew", pady=(4, 14)
        )

        ttk.Label(form, text="Face image source", style="Card.TLabel").grid(
            row=6, column=0, sticky="w"
        )
        source_row = ttk.Frame(form, style="Card.TFrame")
        source_row.grid(row=7, column=0, sticky="ew", pady=(6, 8))
        source_row.columnconfigure(0, weight=1)
        source_row.columnconfigure(1, weight=1)
        ttk.Button(
            source_row,
            text="Browse image",
            style="Primary.TButton",
            command=self._browse,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(
            source_row,
            text="Capture from webcam",
            style="Secondary.TButton",
            command=self._capture,
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.source_label = ttk.Label(
            form, text="No image selected", style="CardMuted.TLabel", wraplength=340
        )
        self.source_label.grid(row=8, column=0, sticky="w", pady=(0, 8))

        self.face_var = tk.StringVar()
        self.face_combo = ttk.Combobox(
            form, textvariable=self.face_var, state="readonly"
        )
        self.face_combo.grid(row=9, column=0, sticky="ew", pady=(0, 8))
        self.face_combo.bind("<<ComboboxSelected>>", self._face_changed)
        self.face_combo.grid_remove()

        self.consent_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            form,
            text=(
                "I confirm that the person has given informed consent to enrol "
                "their face in this academic demonstration."
            ),
            variable=self.consent_var,
        ).grid(row=10, column=0, sticky="w", pady=(8, 12))

        self.enroll_button = ttk.Button(
            form,
            text="Create encrypted template",
            style="Primary.TButton",
            command=self._enroll,
            state="disabled",
        )
        self.enroll_button.grid(row=11, column=0, sticky="ew")

        self.status_label = tk.Label(
            form,
            text="Select an image to begin.",
            bg=COLORS["card"],
            fg=COLORS["muted"],
            font=font(9),
            wraplength=360,
            justify="left",
            anchor="w",
        )
        self.status_label.grid(row=12, column=0, sticky="ew", pady=(12, 0))

        preview = Card(self)
        preview.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(1, weight=1)
        ttk.Label(preview, text="Preview and quality", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )
        self.panel = ImagePanel(
            preview,
            height=390,
            max_size=(560, 390),
            placeholder="Upload or capture a clear, front-facing face image",
        )
        self.panel.grid(row=1, column=0, sticky="nsew")

        self.detection_label = ttk.Label(
            preview, text="Detection: -", style="Card.TLabel"
        )
        self.detection_label.grid(row=2, column=0, sticky="w", pady=(10, 4))
        self.quality_label = ttk.Label(
            preview,
            text="Quality: -",
            style="CardMuted.TLabel",
            wraplength=560,
            justify="left",
        )
        self.quality_label.grid(row=3, column=0, sticky="w")
        ttk.Label(
            preview,
            text=(
                "Quality checks require a face of at least 80x80 px, sufficient "
                "sharpness and reasonable brightness. These checks reduce bad "
                "enrolments but do not guarantee biometric quality."
            ),
            style="CardMuted.TLabel",
            wraplength=560,
            justify="left",
        ).grid(row=4, column=0, sticky="w", pady=(10, 0))

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a face image for enrolment",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.source_kind = "upload"
        self.source_path = path
        self.source_label.configure(text=str(Path(path)))
        self.status_label.configure(text="Detecting face...", fg=COLORS["muted"])
        self._load_and_detect(path)

    def _load_and_detect(self, path: str) -> None:
        def work():
            image = load_image(path)
            detections = self.ctx.engine.detect(image)
            return image, detections

        self.run_background(work, self._image_ready, self._image_failed)

    def _capture(self) -> None:
        dialog = CameraCaptureDialog(self, self.ctx.engine)
        frame = dialog.show()
        if frame is None:
            return
        self.source_kind = "webcam"
        self.source_path = None
        self.source_label.configure(text="Captured from webcam (not saved to disk)")
        self.status_label.configure(text="Detecting face...", fg=COLORS["muted"])

        def work():
            return frame, self.ctx.engine.detect(frame)

        self.run_background(work, self._image_ready, self._image_failed)

    def _image_ready(self, result) -> None:
        image, detections = result
        self.image = image
        self.detections = detections
        self.selected_index = 0
        if detections:
            self.face_combo.configure(
                values=[face.label for face in detections]
            )
            self.face_combo.current(0)
            self.face_combo.grid()
        else:
            self.face_combo.grid_remove()
        self._update_preview()

    def _image_failed(self, exc: Exception) -> None:
        self.image = None
        self.detections = []
        self.panel.set_message("Could not load the image")
        self.detection_label.configure(text="Detection: failed")
        self.quality_label.configure(text=str(exc))
        self.status_label.configure(text=str(exc), fg=COLORS["danger"])
        self.enroll_button.configure(state="disabled")
        messagebox.showerror("Image error", str(exc))

    def _face_changed(self, _event=None) -> None:
        current = self.face_combo.current()
        if current >= 0:
            self.selected_index = current
            self._update_preview()

    def _update_preview(self) -> None:
        if self.image is None:
            return
        if not self.detections:
            self.panel.show_bgr(self.image, max_size=(560, 390))
            self.detection_label.configure(
                text="Detection: no face found", foreground=COLORS["danger"]
            )
            self.quality_label.configure(text="Enrolment cannot continue.")
            self.enroll_button.configure(state="disabled")
            self.status_label.configure(
                text="No face detected. Try another image.", fg=COLORS["danger"]
            )
            return

        face = self.detections[self.selected_index]
        annotated = draw_faces(
            self.image, self.detections, selected_index=face.index
        )
        self.panel.show_bgr(annotated, max_size=(560, 390))
        quality = assess_quality(self.image, face)
        self.detection_label.configure(
            text=(
                f"Detection: {len(self.detections)} face(s), selected face "
                f"{face.w}x{face.h}px, detector score {face.score:.2f}"
            ),
            foreground=COLORS["text"],
        )
        self.quality_label.configure(
            text=quality.summary,
            foreground=COLORS["success"] if quality.passed else COLORS["danger"],
        )
        self.enroll_button.configure(state="normal")
        self.status_label.configure(
            text="Ready to create a template.", fg=COLORS["muted"]
        )

    def _enroll(self) -> None:
        if self.image is None or not self.detections:
            messagebox.showwarning("No face", "Select an image containing a face.")
            return
        user_id = self.user_id_var.get().strip()
        display_name = self.display_name_var.get().strip()
        consent = bool(self.consent_var.get())
        if not consent:
            messagebox.showwarning(
                "Consent required",
                "Confirm informed consent before enrolling a face.",
            )
            return
        self.enroll_button.configure(state="disabled")
        self.status_label.configure(text="Creating encrypted template...", fg=COLORS["muted"])

        def work() -> EnrollmentResult:
            if self.source_kind == "upload" and self.source_path:
                return self.ctx.enrollment.enroll_from_image(
                    user_id,
                    display_name,
                    self.source_path,
                    face_index=self.selected_index,
                    consent=consent,
                )
            return self.ctx.enrollment.enroll_from_frame(
                user_id,
                display_name,
                self.image,
                face_index=self.selected_index,
                consent=consent,
            )

        self.run_background(work, self._enrolled, self._enroll_failed)

    def _enrolled(self, result: EnrollmentResult) -> None:
        self.enroll_button.configure(state="normal")
        if result.success:
            self.status_label.configure(text=result.message, fg=COLORS["success"])
            self.consent_var.set(False)
            messagebox.showinfo("Enrolment complete", result.message)
        else:
            self.status_label.configure(text=result.message, fg=COLORS["danger"])

    def _enroll_failed(self, exc: Exception) -> None:
        self.enroll_button.configure(state="normal")
        self.status_label.configure(text=str(exc), fg=COLORS["danger"])
        messagebox.showerror("Enrolment failed", str(exc))
