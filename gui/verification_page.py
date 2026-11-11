"""Module D: 1:1 identity verification with an image or the webcam."""

from __future__ import annotations

import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from core.database import DatabaseError, UserNotFoundError
from core.face_engine import FaceDetection
from core.image_processor import draw_faces, load_image
from core.security import SecurityError
from core.verification import VerificationResult
from gui.camera_worker import LiveCameraWorker, LiveFrameResult
from gui.styles import COLORS, font
from gui.widgets import Badge, BasePage, Card, ImagePanel, PageHeader


class VerificationPage(BasePage):
    def __init__(self, parent: tk.Misc, controller) -> None:
        super().__init__(parent, controller)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self._user_map: dict[str, str] = {}
        self._image = None
        self._detections: list[FaceDetection] = []
        self._selected_index = 0
        self._image_path: str | None = None
        self._worker: LiveCameraWorker | None = None
        self._poll_id: str | None = None

        PageHeader(
            self,
            "Verify Identity",
            "1:1 verification: compare a new face against one selected enrolled "
            "template. This is not 1:N identification.",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))

        selector = Card(self, padding=14)
        selector.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        selector.columnconfigure(1, weight=1)
        ttk.Label(selector, text="Enrolled user", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.user_var = tk.StringVar()
        self.user_combo = ttk.Combobox(
            selector, textvariable=self.user_var, state="readonly"
        )
        self.user_combo.grid(row=0, column=1, sticky="ew")
        self.user_combo.bind("<<ComboboxSelected>>", self._user_changed)
        ttk.Button(
            selector, text="Refresh", style="Secondary.TButton", command=self._refresh_users
        ).grid(row=0, column=2, padx=8)
        ttk.Label(
            selector,
            text=f"Threshold: {self.ctx.threshold:.3f}",
            style="CardMuted.TLabel",
        ).grid(row=0, column=3, padx=(12, 0))
        ttk.Label(
            selector,
            text=(
                "The threshold is shared with the Compare and Live pages and is "
                "persisted locally. It is a starting value, not a calibrated "
                "operational threshold."
            ),
            style="CardMuted.TLabel",
            wraplength=1000,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(10, 0))

        image_card = Card(self)
        image_card.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        image_card.columnconfigure(0, weight=1)
        image_card.rowconfigure(2, weight=1)
        ttk.Label(image_card, text="Verify with an image", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        top = ttk.Frame(image_card, style="Card.TFrame")
        top.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        top.columnconfigure(0, weight=1)
        ttk.Button(
            top, text="Browse image", style="Secondary.TButton", command=self._browse
        ).grid(row=0, column=0, sticky="w")
        self.image_path_label = ttk.Label(
            top, text="No image selected", style="CardMuted.TLabel", wraplength=420
        )
        self.image_path_label.grid(row=0, column=1, sticky="w", padx=(10, 0))

        self.image_panel = ImagePanel(
            image_card,
            height=300,
            max_size=(520, 300),
            placeholder="Choose a probe image",
        )
        self.image_panel.grid(row=2, column=0, sticky="nsew")

        self.image_detection_label = ttk.Label(
            image_card, text="Detection: -", style="Card.TLabel"
        )
        self.image_detection_label.grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.face_var = tk.StringVar()
        self.face_combo = ttk.Combobox(
            image_card, textvariable=self.face_var, state="readonly"
        )
        self.face_combo.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        self.face_combo.bind("<<ComboboxSelected>>", self._face_changed)
        self.face_combo.grid_remove()

        self.verify_image_button = ttk.Button(
            image_card,
            text="Verify selected image",
            style="Primary.TButton",
            command=self._verify_image,
            state="disabled",
        )
        self.verify_image_button.grid(row=5, column=0, sticky="ew", pady=(10, 0))

        self.image_result_badge = Badge(image_card, "NOT VERIFIED", "muted")
        self.image_result_badge.grid(row=6, column=0, sticky="w", pady=(10, 0))
        self.image_score_label = ttk.Label(
            image_card, text="Similarity: -", style="CardValueSmall.TLabel"
        )
        self.image_score_label.grid(row=7, column=0, sticky="w", pady=(4, 0))
        self.image_message_label = ttk.Label(
            image_card,
            text="Select a user and a probe image.",
            style="CardMuted.TLabel",
            wraplength=520,
            justify="left",
        )
        self.image_message_label.grid(row=8, column=0, sticky="w", pady=(4, 0))

        camera_card = Card(self)
        camera_card.grid(row=2, column=1, sticky="nsew", padx=(8, 0))
        camera_card.columnconfigure(0, weight=1)
        camera_card.rowconfigure(2, weight=1)
        ttk.Label(
            camera_card, text="Verify with the webcam", style="CardTitle.TLabel"
        ).grid(row=0, column=0, sticky="w")
        camera_controls = ttk.Frame(camera_card, style="Card.TFrame")
        camera_controls.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        self.start_camera_button = ttk.Button(
            camera_controls,
            text="Start verification",
            style="Primary.TButton",
            command=self._start_camera,
        )
        self.start_camera_button.pack(side="left")
        self.stop_camera_button = ttk.Button(
            camera_controls,
            text="Stop",
            style="Danger.TButton",
            command=self._stop_camera,
            state="disabled",
        )
        self.stop_camera_button.pack(side="left", padx=(8, 0))

        self.camera_panel = ImagePanel(
            camera_card,
            height=300,
            max_size=(520, 300),
            placeholder="Webcam verification is stopped",
        )
        self.camera_panel.grid(row=2, column=0, sticky="nsew")

        self.camera_status_label = ttk.Label(
            camera_card, text="Status: idle", style="Card.TLabel"
        )
        self.camera_status_label.grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.camera_result_badge = Badge(camera_card, "NOT VERIFIED", "muted")
        self.camera_result_badge.grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.camera_score_label = ttk.Label(
            camera_card, text="Similarity: -", style="CardValueSmall.TLabel"
        )
        self.camera_score_label.grid(row=5, column=0, sticky="w", pady=(4, 0))
        ttk.Label(
            camera_card,
            text=(
                "The live worker compares the largest detected face with the "
                "selected template. It does not prove that the person is physically "
                "present."
            ),
            style="CardMuted.TLabel",
            wraplength=520,
            justify="left",
        ).grid(row=6, column=0, sticky="w", pady=(8, 0))

        self._refresh_users()

    def on_show(self) -> None:
        self._refresh_users()

    def on_hide(self) -> None:
        self._stop_camera(silent=True)

    def _refresh_users(self) -> None:
        try:
            users = self.ctx.database.list_users()
        except DatabaseError as exc:
            messagebox.showerror("Database error", str(exc))
            users = []
        self._user_map = {
            f"{user.display_name} ({user.user_id})": user.user_id for user in users
        }
        values = list(self._user_map) if self._user_map else ["No users enrolled"]
        self.user_combo.configure(values=values)
        if self.user_var.get() not in values:
            self.user_combo.current(0)
        self._update_image_button()

    def _selected_user_id(self) -> str | None:
        return self._user_map.get(self.user_var.get())

    def _user_changed(self, _event=None) -> None:
        self._stop_camera(silent=True)
        self._update_image_button()

    def _update_image_button(self) -> None:
        ready = (
            self._selected_user_id() is not None
            and self._detections
        )
        self.verify_image_button.configure(state="normal" if ready else "disabled")

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a probe image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._image_path = path
        self.image_path_label.configure(text=str(Path(path)))
        self.image_panel.set_message("Loading and detecting faces...")
        self.image_detection_label.configure(text="Detection: running...")

        def work():
            image = load_image(path)
            detections = self.ctx.engine.detect(image)
            return image, detections

        self.run_background(work, self._image_ready, self._image_failed)

    def _image_ready(self, result) -> None:
        self._image, self._detections = result
        self._selected_index = 0
        if self._detections:
            self.face_combo.configure(
                values=[face.label for face in self._detections]
            )
            self.face_combo.current(0)
            self.face_combo.grid()
            self.image_detection_label.configure(
                text=f"Detection: {len(self._detections)} face(s) found",
                foreground=COLORS["success"],
            )
        else:
            self.face_combo.grid_remove()
            self.image_detection_label.configure(
                text="Detection: no face found", foreground=COLORS["danger"]
            )
            self.image_panel.show_bgr(self._image, max_size=(520, 300))
            self._update_image_button()
            return
        self._redraw_image()
        self._update_image_button()

    def _image_failed(self, exc: Exception) -> None:
        self._image = None
        self._detections = []
        self.image_panel.set_message("Could not load the image")
        self.image_detection_label.configure(
            text="Detection: failed", foreground=COLORS["danger"]
        )
        self._update_image_button()
        messagebox.showerror("Image error", str(exc))

    def _face_changed(self, _event=None) -> None:
        current = self.face_combo.current()
        if current >= 0:
            self._selected_index = current
            self._redraw_image()

    def _redraw_image(self) -> None:
        if self._image is None:
            return
        if self._detections:
            annotated = draw_faces(
                self._image,
                self._detections,
                selected_index=self._selected_index,
            )
            self.image_panel.show_bgr(annotated, max_size=(520, 300))
        else:
            self.image_panel.show_bgr(self._image, max_size=(520, 300))

    def _verify_image(self) -> None:
        user_id = self._selected_user_id()
        if user_id is None:
            messagebox.showwarning("No user", "Select an enrolled user first.")
            return
        if not self._image_path or not self._detections:
            messagebox.showwarning("No probe", "Select a probe image with a face.")
            return
        face_index = self._selected_index
        self.verify_image_button.configure(state="disabled")
        self.image_result_badge.set("CHECKING", "warning")
        self.image_score_label.configure(text="Similarity: ...")
        self.image_message_label.configure(text="Comparing SFace embeddings...")

        self.run_background(
            lambda: self.ctx.verification.verify_image(
                user_id, self._image_path, face_index=face_index
            ),
            self._show_image_result,
            self._image_verify_failed,
        )

    def _show_image_result(self, result: VerificationResult) -> None:
        self.verify_image_button.configure(state="normal")
        if not result.success:
            self.image_result_badge.set("NOT VERIFIED", "danger")
            self.image_score_label.configure(text="Similarity: -")
            self.image_message_label.configure(text=result.message)
            return
        if result.verified:
            self.image_result_badge.set("VERIFIED", "success")
        else:
            self.image_result_badge.set("NOT VERIFIED", "danger")
        assert result.similarity is not None
        self.image_score_label.configure(
            text=f"Similarity: {result.similarity:.4f}  (threshold {result.threshold:.3f})"
        )
        self.image_message_label.configure(
            text=result.message + " " + result.score_note
        )

    def _image_verify_failed(self, exc: Exception) -> None:
        self.verify_image_button.configure(state="normal")
        self.image_result_badge.set("ERROR", "danger")
        self.image_message_label.configure(text=str(exc))
        messagebox.showerror("Verification failed", str(exc))

    def _start_camera(self) -> None:
        user_id = self._selected_user_id()
        if user_id is None:
            messagebox.showwarning("No user", "Select an enrolled user first.")
            return
        try:
            user = self.ctx.database.get_user(user_id)
        except (UserNotFoundError, DatabaseError, SecurityError) as exc:
            messagebox.showerror("Cannot load template", str(exc))
            return
        if self._worker is not None and self._worker.is_alive():
            return
        self.camera_panel.set_message("Opening camera...")
        self.camera_status_label.configure(text="Status: opening camera...")
        self.camera_result_badge.set("STARTING", "warning")
        self.camera_score_label.configure(text="Similarity: -")
        self.start_camera_button.configure(state="disabled")
        self.stop_camera_button.configure(state="normal")
        self.user_combo.configure(state="disabled")
        self._worker = LiveCameraWorker(
            engine=self.ctx.engine,
            camera_index=0,
            template=user.embedding,
            user_id=user_id,
            threshold=self.ctx.threshold,
        )
        self._worker.start()
        self._poll()

    def _poll(self) -> None:
        if self._worker is None:
            return
        while True:
            try:
                result = self._worker.result_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_camera(result)
        if self._worker is not None and self._worker.is_alive():
            self._poll_id = self.after(30, self._poll)
        else:
            self._poll_id = None

    def _handle_camera(self, result: LiveFrameResult) -> None:
        if result.error:
            self.camera_status_label.configure(
                text=f"Status: {result.error}", foreground=COLORS["danger"]
            )
            self.camera_result_badge.set("CAMERA ERROR", "danger")
            messagebox.showerror("Camera error", result.error)
            self._stop_camera(silent=True)
            return
        if result.finished:
            return
        if result.annotated_frame is not None:
            self.camera_panel.show_bgr(result.annotated_frame, max_size=(520, 300))
        self.camera_status_label.configure(
            text=(
                f"Status: {len(result.detections)} face(s) detected, "
                f"{result.fps:.1f} FPS"
            ),
            foreground=COLORS["text"],
        )
        if result.similarity is None:
            self.camera_score_label.configure(text="Similarity: -")
            if result.detections:
                self.camera_result_badge.set("FACE DETECTED", "info")
            else:
                self.camera_result_badge.set("NO FACE", "warning")
        else:
            self.camera_score_label.configure(
                text=(
                    f"Similarity: {result.similarity:.4f}  "
                    f"(threshold {self.ctx.threshold:.3f})"
                )
            )
            if result.verified:
                self.camera_result_badge.set("VERIFIED", "success")
            else:
                self.camera_result_badge.set("NOT VERIFIED", "danger")

    def _stop_camera(self, silent: bool = False) -> None:
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.stop()
            try:
                worker.join(timeout=1.5)
            except RuntimeError:
                pass
        self.start_camera_button.configure(state="normal")
        self.stop_camera_button.configure(state="disabled")
        self.user_combo.configure(state="readonly")
        if not silent:
            self.camera_panel.set_message("Webcam verification is stopped")
            self.camera_status_label.configure(text="Status: idle")
            self.camera_result_badge.set("NOT VERIFIED", "muted")
            self.camera_score_label.configure(text="Similarity: -")
