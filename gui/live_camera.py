"""Module B: live webcam detection, 1:N name labels and optional 1:1 verification."""

from __future__ import annotations

import queue
import re
import tkinter as tk
from tkinter import messagebox, ttk

from config import settings
from core.anti_spoofing import (
    PAD_ANALYZING,
    PAD_DISABLED,
    PAD_ERROR,
    PAD_LIVE,
    PAD_LOADING,
    PAD_NO_FACE,
    PAD_READY,
    PAD_SPOOF,
    PAD_UNCERTAIN,
)
from core.camera_manager import CameraError, list_video_devices
from core.database import DatabaseError, UserNotFoundError
from core.matcher import EnrolledIdentity
from core.security import SecurityError
from gui.camera_worker import LiveCameraWorker, LiveFrameResult
from gui.styles import COLORS, font
from gui.widgets import Badge, BasePage, Card, ImagePanel, PageHeader


class LiveCameraPage(BasePage):
    def __init__(self, parent: tk.Misc, controller) -> None:
        super().__init__(parent, controller)
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(2, weight=1)
        self._worker: LiveCameraWorker | None = None
        self._poll_id: str | None = None
        self._user_map: dict[str, str] = {}
        self._camera_values: list[str] = []
        self._diagnostics_enabled = settings.PERFORMANCE_DIAGNOSTICS
        self._last_state: tuple[str, str] | None = None
        self._last_pad_ui_key: tuple | None = None
        self._last_labels: dict[str, str] = {}

        PageHeader(
            self,
            "Live Camera",
            "YuNet detects faces in each frame. Enrolled faces are labelled with "
            "the closest matching name. Select a user for an additional 1:1 "
            "verification. Camera I/O runs off the GUI thread.",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))

        controls = Card(self, padding=14)
        controls.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(4, weight=1)

        ttk.Label(controls, text="Camera", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(
            controls, textvariable=self.camera_var, state="readonly"
        )
        self.camera_combo.grid(row=0, column=1, sticky="ew")
        ttk.Button(
            controls, text="Refresh", style="Secondary.TButton", command=self._refresh_cameras
        ).grid(row=0, column=2, padx=8)

        ttk.Label(controls, text="Verify user", style="Card.TLabel").grid(
            row=0, column=3, sticky="w", padx=(12, 8)
        )
        self.user_var = tk.StringVar()
        self.user_combo = ttk.Combobox(
            controls, textvariable=self.user_var, state="readonly"
        )
        self.user_combo.grid(row=0, column=4, sticky="ew")
        ttk.Button(
            controls, text="Refresh", style="Secondary.TButton", command=self._refresh_users
        ).grid(row=0, column=5, padx=8)

        self.start_button = ttk.Button(
            controls, text="Start camera", style="Primary.TButton", command=self._start
        )
        self.start_button.grid(row=0, column=6, padx=(8, 6))
        self.stop_button = ttk.Button(
            controls,
            text="Stop",
            style="Danger.TButton",
            command=self._stop,
            state="disabled",
        )
        self.stop_button.grid(row=0, column=7)

        self.pad_var = tk.BooleanVar(value=False)
        self.pad_toggle = ttk.Checkbutton(
            controls,
            text="Anti-Spoofing Protection",
            variable=self.pad_var,
            command=self._on_pad_toggle,
        )
        self.pad_toggle.grid(row=1, column=0, columnspan=2, sticky="w", pady=(12, 0))
        self.pad_toggle_badge = Badge(controls, "OFF", "muted")
        self.pad_toggle_badge.grid(
            row=1, column=2, sticky="w", padx=(4, 8), pady=(12, 0)
        )
        ttk.Label(
            controls,
            text=(
                "MiniFASNetV2 checks the primary face for presentation attacks "
                "before SFace identity verification."
            ),
            style="CardMuted.TLabel",
            wraplength=760,
        ).grid(row=1, column=3, columnspan=5, sticky="w", pady=(12, 0))

        self.pad_hint = ttk.Label(
            controls,
            text=(
                "Live mode labels any detected face whose best enrolled match "
                "passes the threshold. The optional target user runs 1:1 "
                "verification on the largest face. These labels are not proof of "
                "identity."
            ),
            style="CardMuted.TLabel",
            wraplength=1100,
        )
        self.pad_hint.grid(row=2, column=0, columnspan=8, sticky="w", pady=(10, 0))

        self.panel = ImagePanel(
            self,
            height=470,
            max_size=(820, 470),
            placeholder="Camera preview is stopped",
        )
        self.panel.grid(row=2, column=0, sticky="nsew", padx=(0, 8))

        status_card = Card(self)
        status_card.grid(row=2, column=1, sticky="nsew", padx=(8, 0))
        status_card.columnconfigure(0, weight=1)
        ttk.Label(status_card, text="Live status", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.state_badge = Badge(status_card, "IDLE", "muted")
        self.state_badge.grid(row=1, column=0, sticky="w", pady=(10, 16))

        self.face_value = self._status_row(status_card, 2, "Faces detected", "-")
        self.fps_value = self._status_row(status_card, 4, "Camera FPS", "-")
        self.score_value = self._status_row(status_card, 6, "Cosine similarity", "-")
        self.threshold_value = self._status_row(
            status_card, 8, "Threshold", f"{self.ctx.threshold:.3f}"
        )
        self.user_value = self._status_row(
            status_card, 10, "Target user", "No target (identification only)"
        )
        self.pad_value = self._status_row(status_card, 12, "Anti-spoofing", "OFF")
        self.diagnostics_value: ttk.Label | None = None
        if self._diagnostics_enabled:
            self.diagnostics_value = self._status_row(
                status_card, 14, "Performance", "-"
            )

        self._refresh_cameras()
        self._refresh_users()
        self._refresh_pad_state()

    def _status_row(self, parent: tk.Misc, row: int, title: str, value: str) -> ttk.Label:
        ttk.Label(parent, text=title.upper(), style="CardMuted.TLabel").grid(
            row=row, column=0, sticky="w", pady=(8, 0)
        )
        label = ttk.Label(parent, text=value, style="CardValueSmall.TLabel")
        label.grid(row=row + 1, column=0, sticky="w")
        return label

    def _set_label(self, label: ttk.Label, key: str, value: str) -> None:
        if self._last_labels.get(key) == value:
            return
        self._last_labels[key] = value
        label.configure(text=value)

    def _set_state(self, text: str, kind: str) -> None:
        if self._last_state == (text, kind):
            return
        self._last_state = (text, kind)
        self.state_badge.set(text, kind)

    def on_show(self) -> None:
        self._refresh_cameras()
        self._refresh_users()
        self.threshold_value.configure(text=f"{self.ctx.threshold:.3f}")
        self._refresh_pad_state()

    def on_hide(self) -> None:
        self._stop(silent=True)

    def _refresh_cameras(self) -> None:
        devices = list_video_devices()
        self._camera_values = (
            [device.display_name for device in devices]
            if devices
            else ["Camera 0 (no device detected)"]
        )
        self.camera_combo.configure(values=self._camera_values)
        if self._camera_values:
            self.camera_combo.current(0)

    def _refresh_users(self) -> None:
        try:
            users = self.ctx.database.list_users()
        except DatabaseError as exc:
            messagebox.showerror("Database error", str(exc))
            users = []
        self._user_map = {
            f"{user.display_name} ({user.user_id})": user.user_id for user in users
        }
        values = ["No target user (identification only)"] + list(self._user_map)
        self.user_combo.configure(values=values)
        if self.user_var.get() not in values:
            self.user_combo.current(0)

    def _camera_index(self) -> int:
        match = re.search(r"Camera\s+(\d+)", self.camera_var.get())
        return int(match.group(1)) if match else 0

    def _selected_user(self) -> tuple[str | None, object | None]:
        label = self.user_var.get()
        if label not in self._user_map:
            return None, None
        user_id = self._user_map[label]
        try:
            return user_id, self.ctx.database.get_user(user_id)
        except (UserNotFoundError, DatabaseError, SecurityError) as exc:
            messagebox.showerror("Cannot load template", str(exc))
            return None, None

    def _load_identities(self) -> list[EnrolledIdentity]:
        identities: list[EnrolledIdentity] = []
        for summary in self.ctx.database.list_users():
            try:
                user = self.ctx.database.get_user(summary.user_id)
            except (UserNotFoundError, DatabaseError, SecurityError):
                continue
            identities.append(
                EnrolledIdentity(
                    user_id=user.user_id,
                    display_name=user.display_name,
                    embedding=user.embedding,
                )
            )
        return identities

    def _pad_status_info(self, status: str) -> tuple[str, str, str]:
        return {
            PAD_DISABLED: (
                "OFF",
                "muted",
                "Protection is off. Static photos and screen replays may be accepted.",
            ),
            PAD_READY: (
                "ON",
                "success",
                "Protection is on. Start the camera to analyze faces.",
            ),
            PAD_LOADING: (
                "LOADING",
                "warning",
                "Loading MiniFASNetV2...",
            ),
            PAD_ERROR: (
                "PAD ERROR",
                "danger",
                "The model could not be loaded. Authentication is blocked while "
                "protection is on.",
            ),
            PAD_NO_FACE: (
                "NO FACE",
                "warning",
                "No face detected. Authentication is blocked.",
            ),
            PAD_ANALYZING: (
                "ANALYZING",
                "info",
                "Checking liveness. Hold still.",
            ),
            PAD_LIVE: (
                "LIVE FACE",
                "success",
                "Live face confirmed. SFace identity verification may proceed.",
            ),
            PAD_SPOOF: (
                "SPOOF DETECTED",
                "danger",
                "Presentation attack detected. Authentication is blocked.",
            ),
            PAD_UNCERTAIN: (
                "LOW CONFIDENCE",
                "warning",
                "Liveness confidence is too low. Hold still and improve lighting.",
            ),
        }.get(status, (status, "muted", ""))

    def _update_pad_ui(
        self,
        status: str,
        message: str | None = None,
        live_score: float | None = None,
        streak: int = 0,
        required: int = 0,
    ) -> None:
        key = (
            status,
            message,
            live_score,
            streak,
            required,
            self.pad_var.get(),
        )
        if key == self._last_pad_ui_key:
            return
        self._last_pad_ui_key = key
        badge_text, kind, default_hint = self._pad_status_info(status)
        colours = {
            "success": COLORS["success"],
            "danger": COLORS["danger"],
            "warning": COLORS["warning"],
            "info": COLORS["accent"],
            "muted": COLORS["muted"],
        }
        self.pad_value.configure(
            text=badge_text,
            foreground=colours.get(kind, COLORS["text"]),
        )
        if status == PAD_DISABLED:
            self.pad_toggle_badge.set("OFF", "muted")
        elif status == PAD_LOADING:
            self.pad_toggle_badge.set("LOADING", "warning")
        elif status == PAD_ERROR:
            self.pad_toggle_badge.set("ERROR", "danger")
        elif self.pad_var.get():
            self.pad_toggle_badge.set("ON", "success")
        else:
            self.pad_toggle_badge.set("OFF", "muted")
        hint = message or default_hint
        if live_score is not None:
            hint = f"{hint}  Live confidence {live_score:.3f}."
            if required:
                hint = f"{hint} Evidence {streak}/{required} live frames."
        self.pad_hint.configure(text=hint)

    def _set_start_button_state(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            self.start_button.configure(state="disabled")
            return
        snapshot = self.ctx.anti_spoofing.runtime.snapshot()
        if snapshot.enabled and snapshot.detector is None:
            self.start_button.configure(state="disabled")
        else:
            self.start_button.configure(state="normal")

    def _refresh_pad_state(self) -> None:
        snapshot = self.ctx.anti_spoofing.runtime.snapshot()
        if not snapshot.enabled:
            self.pad_var.set(False)
            self._update_pad_ui(PAD_DISABLED)
            self._set_start_button_state()
            return
        self.pad_var.set(True)
        if snapshot.detector is not None:
            self._update_pad_ui(PAD_READY)
        elif snapshot.error:
            self._update_pad_ui(PAD_ERROR, snapshot.error)
        else:
            self._update_pad_ui(PAD_LOADING)
        self._set_start_button_state()

    def _on_pad_toggle(self) -> None:
        if not self.pad_var.get():
            self.ctx.anti_spoofing.disable()
            self._refresh_pad_state()
            return

        self.ctx.anti_spoofing.enable()
        snapshot = self.ctx.anti_spoofing.runtime.snapshot()
        if snapshot.detector is not None:
            self._refresh_pad_state()
            return

        self._update_pad_ui(PAD_LOADING)
        self._set_start_button_state()
        self.run_background(
            self.ctx.anti_spoofing.ensure_loaded,
            self._on_pad_loaded,
            self._on_pad_load_error,
        )

    def _on_pad_loaded(self, _detector) -> None:
        if self._worker is not None and self._worker.is_alive():
            self.pad_var.set(True)
            self._update_pad_ui(PAD_ANALYZING, "Starting liveness checks...")
            self._set_start_button_state()
        else:
            self._refresh_pad_state()

    def _on_pad_load_error(self, exc: Exception) -> None:
        if self.pad_var.get():
            self._update_pad_ui(PAD_ERROR, str(exc))
            self._set_start_button_state()
            messagebox.showerror("Anti-spoofing unavailable", str(exc))
        else:
            self._refresh_pad_state()

    def _start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        pad_snapshot = self.ctx.anti_spoofing.runtime.snapshot()
        if pad_snapshot.enabled and pad_snapshot.detector is None:
            messagebox.showerror(
                "Anti-spoofing unavailable",
                pad_snapshot.error
                or "The anti-spoofing model is still loading. Wait a moment.",
            )
            return
        user_id, user = self._selected_user()
        if self.user_var.get() in self._user_map and user is None:
            return
        try:
            identities = self._load_identities()
        except DatabaseError as exc:
            messagebox.showerror("Database error", str(exc))
            return
        template = user.embedding if user is not None else None
        self.panel.set_message("Opening camera...")
        self._set_state("STARTING", "warning")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.camera_combo.configure(state="disabled")
        self.user_combo.configure(state="disabled")
        self.user_value.configure(
            text=(
                user.display_name
                if user is not None
                else "No target (identification only)"
            )
        )
        self._set_label(self.score_value, "score", "-")
        if pad_snapshot.enabled:
            self._update_pad_ui(PAD_ANALYZING, "Starting liveness checks...")
        else:
            self._update_pad_ui(PAD_DISABLED)
        self._worker = LiveCameraWorker(
            engine=self.ctx.engine,
            camera_index=self._camera_index(),
            template=template,
            user_id=user_id,
            threshold=self.ctx.threshold,
            identities=identities,
            pad_runtime=self.ctx.anti_spoofing.runtime,
        )
        self._worker.start()
        self._poll()

    def _poll(self) -> None:
        if self._worker is None:
            return
        latest: LiveFrameResult | None = None
        error: LiveFrameResult | None = None
        finished = False
        while True:
            try:
                result = self._worker.result_queue.get_nowait()
            except queue.Empty:
                break
            if result.error:
                error = result
            elif result.finished:
                finished = True
            else:
                latest = result
        if error is not None:
            self._handle(error)
        elif latest is not None:
            self._handle(latest)
        elif finished:
            self._handle(LiveFrameResult(finished=True))
        if self._worker is not None and self._worker.is_alive():
            self._poll_id = self.after(settings.PREVIEW_REFRESH_MS, self._poll)
        else:
            self._poll_id = None

    def _handle(self, result: LiveFrameResult) -> None:
        if result.error:
            self._set_state("CAMERA ERROR", "danger")
            self.panel.set_message("Camera unavailable")
            messagebox.showerror("Camera error", result.error)
            self._stop(silent=True)
            return
        if result.finished:
            return
        if result.annotated_frame is not None:
            self.panel.show_bgr(
                result.annotated_frame, max_size=(820, 470), fast=True
            )
        if result.identifications:
            face_text = (
                f"{len(result.detections)} "
                f"({result.recognized_count} recognized)"
            )
        else:
            face_text = str(len(result.detections))
        self._set_label(self.face_value, "faces", face_text)
        self._set_label(self.fps_value, "fps", f"{result.fps:.1f}")
        if self.diagnostics_value is not None and result.diagnostics is not None:
            self._set_label(
                self.diagnostics_value,
                "diagnostics",
                result.diagnostics.summary(),
            )

        if result.pad_enabled:
            self._update_pad_ui(
                result.pad_status,
                result.pad_message,
                result.pad_live_score,
                result.pad_live_streak,
                result.pad_required_frames,
            )
            if result.pad_status == PAD_LIVE:
                if result.similarity is not None:
                    self._set_label(
                        self.score_value, "score", f"{result.similarity:.4f}"
                    )
                    if result.verified:
                        self._set_state("VERIFIED", "success")
                    else:
                        self._set_state("NOT VERIFIED", "danger")
                elif result.recognized_count:
                    self._set_label(self.score_value, "score", "-")
                    self._set_state("FACE RECOGNIZED", "success")
                else:
                    self._set_label(self.score_value, "score", "-")
                    self._set_state("LIVE FACE", "success")
                return
            blocked = {
                PAD_SPOOF: ("AUTH BLOCKED", "danger"),
                PAD_ERROR: ("PAD ERROR", "danger"),
                PAD_LOADING: ("PAD LOADING", "warning"),
                PAD_NO_FACE: ("NO FACE", "warning"),
                PAD_ANALYZING: ("PAD CHECK", "info"),
                PAD_UNCERTAIN: ("HOLD STILL", "warning"),
            }
            if result.pad_status in blocked:
                self._set_label(self.score_value, "score", "-")
                text, kind = blocked[result.pad_status]
                self._set_state(text, kind)
                return
        else:
            self._update_pad_ui(PAD_DISABLED)

        if result.similarity is None:
            self._set_label(self.score_value, "score", "-")
            if result.recognized_count:
                self._set_state("FACE RECOGNIZED", "success")
            elif result.detections:
                self._set_state("FACE DETECTED", "info")
            else:
                self._set_state("NO FACE", "warning")
        else:
            self._set_label(
                self.score_value, "score", f"{result.similarity:.4f}"
            )
            if result.verified:
                self._set_state("VERIFIED", "success")
            else:
                self._set_state("NOT VERIFIED", "danger")

    def _stop(self, silent: bool = False) -> None:
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
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.camera_combo.configure(state="readonly")
        self.user_combo.configure(state="readonly")
        if not silent:
            self.panel.set_message("Camera stopped")
            self._set_state("IDLE", "muted")
            self._set_label(self.face_value, "faces", "-")
            self._set_label(self.fps_value, "fps", "-")
            self._set_label(self.score_value, "score", "-")
        self._refresh_pad_state()
