"""Modal webcam capture dialog used during enrolment."""

from __future__ import annotations

import queue
import re
import tkinter as tk
from tkinter import ttk

import numpy as np

from config import settings
from core.camera_manager import list_video_devices
from core.face_engine import FaceEngine
from gui.camera_worker import LiveCameraWorker, LiveFrameResult
from gui.styles import COLORS, font
from gui.widgets import ImagePanel


class CameraCaptureDialog(tk.Toplevel):
    """Show a live preview and return a single captured BGR frame."""

    def __init__(self, parent: tk.Misc, engine: FaceEngine) -> None:
        super().__init__(parent)
        self.engine = engine
        self.captured_frame: np.ndarray | None = None
        self._worker: LiveCameraWorker | None = None
        self._latest: LiveFrameResult | None = None
        self._closed = False
        self._poll_id: str | None = None
        self._devices = list_video_devices()

        self.title("Capture face image")
        self.configure(bg=COLORS["bg"])
        self.geometry("780x650")
        self.minsize(640, 560)
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._close)

        self._build()
        self.after(120, self._start)

    def _build(self) -> None:
        frame = ttk.Frame(self, style="App.TFrame", padding=20)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)

        ttk.Label(
            frame,
            text="Capture an enrolment image",
            style="PageTitle.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            frame,
            text=(
                "Look at the camera in even lighting. The image is used only to "
                "extract an encrypted template; it is not saved."
            ),
            style="PageSubtitle.TLabel",
            wraplength=700,
        ).grid(row=1, column=0, sticky="w", pady=(4, 12))

        self.panel = ImagePanel(
            frame,
            height=380,
            max_size=(720, 380),
            placeholder="Starting camera...",
        )
        self.panel.grid(row=2, column=0, sticky="nsew", pady=(0, 12))

        controls = ttk.Frame(frame, style="App.TFrame")
        controls.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Camera", style="PageSubtitle.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(
            controls,
            textvariable=self.camera_var,
            state="readonly",
            values=self._camera_values(),
        )
        self.camera_combo.grid(row=0, column=1, sticky="ew")
        if self._devices:
            self.camera_combo.current(0)
        ttk.Button(
            controls, text="Restart", style="Secondary.TButton", command=self._start
        ).grid(row=0, column=2, padx=(8, 0))

        self.status_label = tk.Label(
            frame,
            text="Starting camera...",
            bg=COLORS["card"],
            fg=COLORS["muted"],
            font=font(9),
            anchor="w",
            padx=10,
            pady=6,
        )
        self.status_label.grid(row=4, column=0, sticky="ew", pady=(0, 10))

        buttons = ttk.Frame(frame, style="App.TFrame")
        buttons.grid(row=5, column=0, sticky="e")
        ttk.Button(
            buttons, text="Cancel", style="Secondary.TButton", command=self._close
        ).pack(side="left", padx=(0, 8))
        self.capture_button = ttk.Button(
            buttons,
            text="Capture frame",
            style="Primary.TButton",
            command=self._capture,
            state="disabled",
        )
        self.capture_button.pack(side="left")

    def _camera_values(self) -> list[str]:
        if not self._devices:
            return ["Camera 0 (no device detected)"]
        return [device.display_name for device in self._devices]

    def _selected_index(self) -> int:
        text = self.camera_var.get()
        match = re.search(r"Camera\s+(\d+)", text)
        if match:
            return int(match.group(1))
        return 0

    def _start(self) -> None:
        self._stop_worker()
        index = self._selected_index()
        self.panel.set_message("Starting camera...")
        self.status_label.configure(
            text=f"Opening camera {index}...", fg=COLORS["muted"]
        )
        self.capture_button.configure(state="disabled")
        self._latest = None
        self._worker = LiveCameraWorker(
            engine=self.engine,
            camera_index=index,
        )
        self._worker.start()
        self._poll()

    def _poll(self) -> None:
        if self._closed:
            return
        if self._worker is not None:
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
        self._poll_id = self.after(settings.PREVIEW_REFRESH_MS, self._poll)

    def _handle(self, result: LiveFrameResult) -> None:
        if result.finished:
            return
        if result.error:
            self.status_label.configure(text=result.error, fg=COLORS["danger"])
            self.panel.set_message("Camera unavailable")
            self.capture_button.configure(state="disabled")
            return
        self._latest = result
        if result.annotated_frame is not None:
            self.panel.show_bgr(
                result.annotated_frame, max_size=(720, 380), fast=True
            )
            self.capture_button.configure(state="normal")
        if result.detections:
            self.status_label.configure(
                text=(
                    f"{len(result.detections)} face(s) detected. "
                    "Capture when the face is clear and centred."
                ),
                fg=COLORS["success"],
            )
        else:
            self.status_label.configure(
                text="No face detected yet. Move closer and face the camera.",
                fg=COLORS["warning"],
            )

    def _capture(self) -> None:
        if self._latest is None or self._latest.raw_frame is None:
            self.status_label.configure(
                text="No frame is available yet.", fg=COLORS["warning"]
            )
            return
        self.captured_frame = self._latest.raw_frame.copy()
        self._close()

    def _stop_worker(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            try:
                self._worker.join(timeout=1.5)
            except RuntimeError:
                pass
            self._worker = None

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        self._stop_worker()
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()

    def show(self) -> np.ndarray | None:
        self.wait_window()
        return self.captured_frame
