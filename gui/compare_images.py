"""Module A: compare two still images with YuNet + SFace."""

from __future__ import annotations

import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from config import settings
from core.face_engine import FaceDetection
from core.image_processor import draw_faces, load_image, save_image
from core.matcher import MatchResult
from gui.styles import COLORS, font
from gui.widgets import BasePage, Card, ImagePanel, PageHeader


class _ImageSide(Card):
    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        on_browse,
        on_selection,
    ) -> None:
        super().__init__(parent, padding=14)
        self.on_browse = on_browse
        self.on_selection = on_selection
        self.image = None
        self.detections: list[FaceDetection] = []
        self.path: str | None = None
        self.selected_index = 0

        self.columnconfigure(0, weight=1)
        header = ttk.Frame(self, style="Card.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=title, style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(
            header,
            text="Browse...",
            style="Secondary.TButton",
            command=lambda: self.on_browse(self),
        ).grid(row=0, column=1, sticky="e")

        self.path_label = ttk.Label(
            self,
            text="No file selected",
            style="CardMuted.TLabel",
            wraplength=380,
        )
        self.path_label.grid(row=1, column=0, sticky="w", pady=(6, 8))

        self.panel = ImagePanel(
            self,
            height=275,
            max_size=(400, 275),
            placeholder="Choose an image to detect a face",
        )
        self.panel.grid(row=2, column=0, sticky="nsew")

        self.status_label = tk.Label(
            self,
            text="Waiting for an image",
            bg=COLORS["card"],
            fg=COLORS["muted"],
            font=font(9),
            anchor="w",
        )
        self.status_label.grid(row=3, column=0, sticky="ew", pady=(8, 0))

        self.face_var = tk.StringVar()
        self.face_combo = ttk.Combobox(
            self, textvariable=self.face_var, state="readonly"
        )
        self.face_combo.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        self.face_combo.bind("<<ComboboxSelected>>", self._face_changed)
        self.face_combo.grid_remove()

    def set_loading(self, path: str) -> None:
        self.path = path
        self.image = None
        self.detections = []
        self.path_label.configure(text=Path(path).name)
        self.panel.set_message("Loading and detecting faces...")
        self.status_label.configure(
            text="Running YuNet face detection...", fg=COLORS["muted"]
        )
        self.face_combo.grid_remove()

    def set_loaded(
        self, image, detections: list[FaceDetection], path: str
    ) -> None:
        self.image = image
        self.detections = detections
        self.path = path
        self.selected_index = 0
        self.path_label.configure(text=str(Path(path)))
        if detections:
            values = [face.label for face in detections]
            self.face_combo.configure(values=values)
            self.face_combo.current(0)
            self.face_combo.grid()
            if len(detections) > 1:
                self.status_label.configure(
                    text=(
                        f"{len(detections)} faces detected. Select the face to "
                        "compare using the list below."
                    ),
                    fg=COLORS["warning"],
                )
            else:
                self.status_label.configure(
                    text="1 face detected.", fg=COLORS["success"]
                )
        else:
            self.face_combo.grid_remove()
            self.status_label.configure(
                text="No face detected in this image.", fg=COLORS["danger"]
            )
        self.redraw()

    def set_error(self, message: str) -> None:
        self.image = None
        self.detections = []
        self.panel.set_message("Could not load this image")
        self.status_label.configure(text=message, fg=COLORS["danger"])
        self.face_combo.grid_remove()

    def selected_face(self) -> FaceDetection | None:
        if not self.detections:
            return None
        if 0 <= self.selected_index < len(self.detections):
            return self.detections[self.selected_index]
        return self.detections[0]

    def redraw(self) -> None:
        if self.image is None:
            return
        if self.detections:
            annotated = draw_faces(
                self.image,
                self.detections,
                selected_index=self.selected_index,
            )
            self.panel.show_bgr(annotated, max_size=(400, 275))
        else:
            self.panel.show_bgr(self.image, max_size=(400, 275))

    def _face_changed(self, _event=None) -> None:
        current = self.face_combo.current()
        if current >= 0:
            self.selected_index = current
            self.redraw()
            self.on_selection()


class CompareImagesPage(BasePage):
    def __init__(self, parent: tk.Misc, controller) -> None:
        super().__init__(parent, controller)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.last_result: MatchResult | None = None

        PageHeader(
            self,
            "Compare Two Images",
            "Detect one face in each image, extract SFace embeddings and compare "
            "them with cosine similarity. Multiple faces require an explicit choice.",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 14))

        panels = ttk.Frame(self, style="App.TFrame")
        panels.grid(row=1, column=0, sticky="nsew")
        panels.grid_columnconfigure(0, weight=1, uniform="sides")
        panels.grid_columnconfigure(1, weight=1, uniform="sides")
        panels.grid_rowconfigure(0, weight=1)

        self.left = _ImageSide(
            panels, "Image 1", self._browse, self._update_compare_state
        )
        self.left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.right = _ImageSide(
            panels, "Image 2", self._browse, self._update_compare_state
        )
        self.right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        controls = Card(self, padding=14)
        controls.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        controls.columnconfigure(3, weight=1)
        ttk.Label(controls, text="Decision threshold", style="Card.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.threshold_var = tk.StringVar(value=f"{self.ctx.threshold:.3f}")
        ttk.Entry(controls, textvariable=self.threshold_var, width=8).grid(
            row=0, column=1, padx=(8, 8)
        )
        ttk.Label(
            controls,
            text="Cosine similarity in [0, 1]; the default is not calibrated for this demo.",
            style="CardMuted.TLabel",
        ).grid(row=0, column=2, sticky="w")
        self.compare_button = ttk.Button(
            controls,
            text="Compare faces",
            style="Primary.TButton",
            command=self._compare,
            state="disabled",
        )
        self.compare_button.grid(row=0, column=4, padx=(12, 8))
        self.save_button = ttk.Button(
            controls,
            text="Save annotated images",
            style="Secondary.TButton",
            command=self._save,
            state="disabled",
        )
        self.save_button.grid(row=0, column=5)

        result = Card(self)
        result.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        result.columnconfigure(0, weight=1)
        result.columnconfigure(1, weight=2)
        ttk.Label(result, text="Comparison result", style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        self.decision_label = ttk.Label(
            result, text="Waiting for two images", style="CardValue.TLabel"
        )
        self.decision_label.grid(row=1, column=0, sticky="w", pady=(8, 4))
        self.score_label = ttk.Label(
            result, text="Similarity: -", style="CardValueSmall.TLabel"
        )
        self.score_label.grid(row=2, column=0, sticky="w")
        self.detail_label = ttk.Label(
            result,
            text="The decision compares the cosine similarity with the configured threshold.",
            style="CardMuted.TLabel",
            wraplength=560,
            justify="left",
        )
        self.detail_label.grid(row=1, column=1, rowspan=2, sticky="w", padx=(20, 0))
        self.note_label = ttk.Label(
            result,
            text=(
                "Similarity is not a probability of identity. A high score means "
                "the two embeddings point in a similar direction."
            ),
            style="CardMuted.TLabel",
            wraplength=900,
            justify="left",
        )
        self.note_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def on_show(self) -> None:
        self.threshold_var.set(f"{self.ctx.threshold:.3f}")

    def _browse(self, side: _ImageSide) -> None:
        path = filedialog.askopenfilename(
            title="Choose a face image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        side.set_loading(path)
        self.compare_button.configure(state="disabled")
        self.save_button.configure(state="disabled")

        def work():
            image = load_image(path)
            detections = self.ctx.engine.detect(image)
            return image, detections

        self.run_background(
            work,
            lambda result: self._loaded(side, path, result),
            lambda exc: self._load_failed(side, exc),
        )

    def _loaded(self, side: _ImageSide, path: str, result) -> None:
        image, detections = result
        side.set_loaded(image, detections, path)
        self._update_compare_state()

    def _load_failed(self, side: _ImageSide, exc: Exception) -> None:
        side.set_error(str(exc))
        messagebox.showerror("Image error", str(exc))

    def _update_compare_state(self) -> None:
        ready = (
            self.left.selected_face() is not None
            and self.right.selected_face() is not None
        )
        self.compare_button.configure(state="normal" if ready else "disabled")
        if not ready:
            self.save_button.configure(state="disabled")

    def _threshold(self) -> float | None:
        try:
            value = float(self.threshold_var.get().strip())
        except ValueError:
            messagebox.showerror(
                "Invalid threshold", "Enter a number between 0.0 and 1.0."
            )
            return None
        if value < settings.MIN_THRESHOLD or value > settings.MAX_THRESHOLD:
            messagebox.showerror(
                "Invalid threshold", "The threshold must be between 0.0 and 1.0."
            )
            return None
        return self.controller.set_threshold(value)

    def _compare(self) -> None:
        face_left = self.left.selected_face()
        face_right = self.right.selected_face()
        if face_left is None or face_right is None:
            messagebox.showwarning(
                "Missing face", "Both images must contain a selected face."
            )
            return
        threshold = self._threshold()
        if threshold is None:
            return
        self.threshold_var.set(f"{threshold:.3f}")
        self.compare_button.configure(state="disabled")
        self.decision_label.configure(text="Comparing...", foreground=COLORS["accent"])
        self.score_label.configure(text="Similarity: ...")
        self.detail_label.configure(text="Extracting SFace embeddings...")

        def work():
            feature_left = self.ctx.engine.extract_feature(
                self.left.image, face_left
            )
            feature_right = self.ctx.engine.extract_feature(
                self.right.image, face_right
            )
            return self.ctx.matcher.compare_features(feature_left, feature_right)

        self.run_background(work, self._show_result, self._compare_failed)

    def _show_result(self, result: MatchResult) -> None:
        self.last_result = result
        colour = COLORS["success"] if result.matched else COLORS["danger"]
        self.decision_label.configure(text=result.decision, foreground=colour)
        self.score_label.configure(
            text=f"Cosine similarity: {result.similarity:.4f}"
        )
        self.detail_label.configure(
            text=(
                f"{result.explanation}.  Method: {result.method}.  "
                "Threshold set by the user for this comparison."
            )
        )
        self.compare_button.configure(state="normal")
        self.save_button.configure(state="normal")

    def _compare_failed(self, exc: Exception) -> None:
        self.decision_label.configure(
            text="COMPARISON FAILED", foreground=COLORS["danger"]
        )
        self.score_label.configure(text="Similarity: -")
        self.detail_label.configure(text=str(exc))
        self.compare_button.configure(state="normal")
        messagebox.showerror("Comparison failed", str(exc))

    def _save(self) -> None:
        if self.last_result is None:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = settings.RESULTS_DIR / f"compare_{stamp}"
        left_face = self.left.selected_face()
        right_face = self.right.selected_face()
        if left_face is None or right_face is None:
            return
        left = draw_faces(
            self.left.image, self.left.detections, selected_index=left_face.index
        )
        right = draw_faces(
            self.right.image, self.right.detections, selected_index=right_face.index
        )
        try:
            save_image(output_dir / "image_1_annotated.jpg", left)
            save_image(output_dir / "image_2_annotated.jpg", right)
            (output_dir / "result.txt").write_text(
                f"{self.last_result.decision}\n"
                f"cosine_similarity={self.last_result.similarity:.8f}\n"
                f"threshold={self.last_result.threshold:.4f}\n"
                "note=cosine similarity is not a probability of identity\n",
                encoding="utf-8",
            )
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        messagebox.showinfo("Saved", f"Annotated images saved to:\n{output_dir}")
