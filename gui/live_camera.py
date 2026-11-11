"""Module B: live webcam detection, 1:N name labels and optional 1:1 verification."""

from __future__ import annotations

import queue
import re
import tkinter as tk
from tkinter import messagebox, ttk

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

        ttk.Label(
            controls,
            text=(
                "Live mode labels any detected face whose best enrolled match "
                "passes the threshold. The optional target user runs 1:1 "
                "verification on the largest face. These labels are not proof of "
                "identity."
            ),
            style="CardMuted.TLabel",
            wraplength=1100,
        ).grid(row=1, column=0, columnspan=8, sticky="w", pady=(10, 0))

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

        ttk.Label(
            status_card,
            text=(
                "A static photograph can also produce a match. YuNet and SFace "
                "do not provide liveness detection. 1:N name labels are a "
                "demonstration, not an access-control decision."
            ),
            style="CardMuted.TLabel",
            wraplength=330,
            justify="left",
        ).grid(row=12, column=0, sticky="w", pady=(18, 0))

        self._refresh_cameras()
        self._refresh_users()

    def _status_row(self, parent: tk.Misc, row: int, title: str, value: str) -> ttk.Label:
        ttk.Label(parent, text=title.upper(), style="CardMuted.TLabel").grid(
            row=row, column=0, sticky="w", pady=(8, 0)
        )
        label = ttk.Label(parent, text=value, style="CardValueSmall.TLabel")
        label.grid(row=row + 1, column=0, sticky="w")
        return label

    def on_show(self) -> None:
        self._refresh_cameras()
        self._refresh_users()
        self.threshold_value.configure(text=f"{self.ctx.threshold:.3f}")

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

    def _start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
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
        self.state_badge.set("STARTING", "warning")
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
        self.score_value.configure(text="-")
        self._worker = LiveCameraWorker(
            engine=self.ctx.engine,
            camera_index=self._camera_index(),
            template=template,
            user_id=user_id,
            threshold=self.ctx.threshold,
            identities=identities,
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
            self._handle(result)
        if self._worker is not None and self._worker.is_alive():
            self._poll_id = self.after(30, self._poll)
        else:
            self._poll_id = None

    def _handle(self, result: LiveFrameResult) -> None:
        if result.error:
            self.state_badge.set("CAMERA ERROR", "danger")
            self.panel.set_message("Camera unavailable")
            messagebox.showerror("Camera error", result.error)
            self._stop(silent=True)
            return
        if result.finished:
            return
        if result.annotated_frame is not None:
            self.panel.show_bgr(result.annotated_frame, max_size=(820, 470))
        if result.identifications:
            self.face_value.configure(
                text=(
                    f"{len(result.detections)} "
                    f"({result.recognized_count} recognized)"
                )
            )
        else:
            self.face_value.configure(text=str(len(result.detections)))
        self.fps_value.configure(text=f"{result.fps:.1f}")
        if result.similarity is None:
            self.score_value.configure(text="-")
            if result.recognized_count:
                self.state_badge.set("FACE RECOGNIZED", "success")
            elif result.detections:
                self.state_badge.set("FACE DETECTED", "info")
            else:
                self.state_badge.set("NO FACE", "warning")
        else:
            self.score_value.configure(text=f"{result.similarity:.4f}")
            if result.verified:
                self.state_badge.set("VERIFIED", "success")
            else:
                self.state_badge.set("NOT VERIFIED", "danger")

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
            self.state_badge.set("IDLE", "muted")
            self.face_value.configure(text="-")
            self.fps_value.configure(text="-")
            self.score_value.configure(text="-")
