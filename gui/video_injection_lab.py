"""Video Injection Attack Lab: replay a recorded file through the real pipeline."""

from __future__ import annotations

import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from config import settings
from core.active_liveness import CHALLENGE_LABELS, LIVENESS_PASSED
from core.anti_spoofing import (
    PAD_ANALYZING,
    PAD_DISABLED,
    PAD_ERROR,
    PAD_LIVE,
    PAD_LOADING,
    PAD_NO_FACE,
    PAD_NOT_RUN,
    PAD_READY,
    PAD_SPOOF,
    PAD_UNCERTAIN,
)
from core.database import DatabaseError, UserNotFoundError
from core.frame_source import VIDEO_EXTENSIONS, RecordedVideoSource
from core.protection import (
    FINAL_ACCEPTED,
    FINAL_NOT_RUN,
    FINAL_REJECTED,
    ProtectionConfig,
)
from core.security import SecurityError
from core.source_integrity import (
    SOURCE_NOT_RUN,
    SOURCE_REJECTED,
    SOURCE_TRUSTED,
    SOURCE_UNCERTAIN,
)
from gui.camera_worker import LiveCameraWorker, LiveFrameResult
from gui.styles import COLORS, font
from gui.widgets import (
    Badge,
    BasePage,
    Card,
    ImagePanel,
    PageHeader,
    ScrollableFrame,
)

MODE_ORIGINAL = "Original / Vulnerable"
MODE_FULL = "Full Protection"
MODE_LIVENESS = "Liveness Simulation"
MODE_CUSTOM = "Custom"

MODES = (MODE_ORIGINAL, MODE_FULL, MODE_LIVENESS, MODE_CUSTOM)

_MODE_DEFAULTS = {
    MODE_ORIGINAL: (False, False, False),
    MODE_FULL: (True, True, True),
    MODE_LIVENESS: (False, True, False),
}


class VideoInjectionLabPage(BasePage):
    def __init__(self, parent: tk.Misc, controller) -> None:
        super().__init__(parent, controller)
        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True)
        self.body = scroll.inner
        self.body.grid_columnconfigure(0, weight=3)
        self.body.grid_columnconfigure(1, weight=2)
        self.body.grid_rowconfigure(3, weight=1)

        self._worker: LiveCameraWorker | None = None
        self._source: RecordedVideoSource | None = None
        self._poll_id: str | None = None
        self._user_map: dict[str, str] = {}
        self._video_path: str | None = None
        self._session = None
        self._results: dict[str, LiveFrameResult] = {}
        self._starting = False
        self._updating_mode = False
        self._last_pipeline_key: tuple | None = None

        PageHeader(
            self.body,
            "Video Injection Attack Lab",
            "Feed a recorded video file through the same YuNet, SFace, PAD and "
            "decision pipeline as the Live Camera page. Compare the original "
            "configuration with source-integrity, active-liveness and PAD defenses.",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))

        self._build_controls()
        self._build_preview()
        self._build_pipeline()
        self._build_results()

        self._refresh_users()
        self._refresh_pad_state()
        self._apply_mode(MODE_ORIGINAL)
        self._update_challenge_display()

    def _build_controls(self) -> None:
        controls = Card(self.body, padding=14)
        controls.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(4, weight=1)

        ttk.Label(controls, text="Recorded video", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.path_var = tk.StringVar(value="No video selected")
        ttk.Entry(
            controls, textvariable=self.path_var, state="readonly"
        ).grid(row=0, column=1, columnspan=2, sticky="ew")
        ttk.Button(
            controls,
            text="Choose video",
            style="Secondary.TButton",
            command=self._browse_video,
        ).grid(row=0, column=3, padx=8)

        ttk.Label(controls, text="Enrolled target", style="Card.TLabel").grid(
            row=0, column=4, sticky="w", padx=(12, 8)
        )
        self.user_var = tk.StringVar()
        self.user_combo = ttk.Combobox(
            controls, textvariable=self.user_var, state="readonly"
        )
        self.user_combo.grid(row=0, column=5, sticky="ew")
        ttk.Button(
            controls,
            text="Refresh",
            style="Secondary.TButton",
            command=self._refresh_users,
        ).grid(row=0, column=6, padx=8)

        ttk.Label(controls, text="Mode", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", pady=(10, 0)
        )
        self.mode_var = tk.StringVar(value=MODE_ORIGINAL)
        self.mode_combo = ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            values=MODES,
            state="readonly",
        )
        self.mode_combo.grid(row=1, column=1, sticky="w", pady=(10, 0))
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_mode_selected)

        self.source_integrity_var = tk.BooleanVar(value=False)
        self.liveness_var = tk.BooleanVar(value=False)
        self.pad_var = tk.BooleanVar(value=False)
        self.source_integrity_check = ttk.Checkbutton(
            controls,
            text="Source integrity",
            variable=self.source_integrity_var,
            command=self._on_protection_changed,
        )
        self.source_integrity_check.grid(
            row=1, column=2, sticky="w", padx=(12, 0), pady=(10, 0)
        )
        self.liveness_check = ttk.Checkbutton(
            controls,
            text="Active liveness",
            variable=self.liveness_var,
            command=self._on_protection_changed,
        )
        self.liveness_check.grid(
            row=1, column=3, sticky="w", padx=(12, 0), pady=(10, 0)
        )
        self.pad_check = ttk.Checkbutton(
            controls,
            text="Passive PAD (MiniFASNetV2)",
            variable=self.pad_var,
            command=self._on_protection_changed,
        )
        self.pad_check.grid(
            row=1, column=4, columnspan=2, sticky="w", padx=(12, 0), pady=(10, 0)
        )
        self.pad_badge = Badge(controls, "OFF", "muted")
        self.pad_badge.grid(row=1, column=6, sticky="w", pady=(10, 0))

        self.start_button = ttk.Button(
            controls,
            text="Run attack",
            style="Primary.TButton",
            command=self._start,
        )
        self.start_button.grid(row=2, column=0, sticky="w", pady=(12, 0))
        self.stop_button = ttk.Button(
            controls,
            text="Stop",
            style="Danger.TButton",
            command=self._stop,
            state="disabled",
        )
        self.stop_button.grid(row=2, column=1, sticky="w", pady=(12, 0))
        self.pause_button = ttk.Button(
            controls,
            text="Pause",
            style="Secondary.TButton",
            command=self._pause,
            state="disabled",
        )
        self.pause_button.grid(row=2, column=2, sticky="w", padx=(8, 0), pady=(12, 0))
        self.resume_button = ttk.Button(
            controls,
            text="Resume",
            style="Secondary.TButton",
            command=self._resume,
            state="disabled",
        )
        self.resume_button.grid(row=2, column=3, sticky="w", padx=(8, 0), pady=(12, 0))
        self.replay_button = ttk.Button(
            controls,
            text="Replay",
            style="Secondary.TButton",
            command=self._replay,
            state="disabled",
        )
        self.replay_button.grid(row=2, column=4, sticky="w", padx=(8, 0), pady=(12, 0))
        self.challenge_button = ttk.Button(
            controls,
            text="New liveness challenge",
            style="Secondary.TButton",
            command=self._new_challenge,
        )
        self.challenge_button.grid(
            row=2, column=5, columnspan=2, sticky="w", padx=(8, 0), pady=(12, 0)
        )

        self.hint_label = ttk.Label(
            controls,
            text=(
                "The recorded file is decoded by the real camera worker and passes "
                "through the same detection, recognition and decision code. Source "
                "integrity is an application-level check, not cryptographic proof "
                "of camera authenticity."
            ),
            style="CardMuted.TLabel",
            wraplength=1050,
            justify="left",
        )
        self.hint_label.grid(
            row=3, column=0, columnspan=7, sticky="w", pady=(10, 0)
        )

    def _build_preview(self) -> None:
        preview_card = Card(self.body)
        preview_card.grid(row=3, column=0, sticky="nsew", padx=(0, 8))
        preview_card.grid_rowconfigure(1, weight=1)
        preview_card.grid_columnconfigure(0, weight=1)
        ttk.Label(
            preview_card, text="Attack playback", style="CardTitle.TLabel"
        ).grid(row=0, column=0, sticky="w")
        self.panel = ImagePanel(
            preview_card,
            height=390,
            max_size=(720, 390),
            placeholder="Choose a recorded video and run an experiment",
        )
        self.panel.grid(row=1, column=0, sticky="nsew", pady=(10, 8))
        self.progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(
            preview_card,
            variable=self.progress_var,
            maximum=100.0,
        ).grid(row=2, column=0, sticky="ew")
        self.playback_label = ttk.Label(
            preview_card,
            text="Playback: stopped",
            style="CardMuted.TLabel",
        )
        self.playback_label.grid(row=3, column=0, sticky="w", pady=(6, 0))

    def _build_pipeline(self) -> None:
        card = Card(self.body)
        card.grid(row=3, column=1, sticky="nsew", padx=(8, 0))
        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=1)
        ttk.Label(
            card, text="Authentication pipeline", style="CardTitle.TLabel"
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        self.pipeline_badge = Badge(card, "IDLE", "muted")
        self.pipeline_badge.grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(10, 8)
        )
        self.source_value = self._status_row(card, 2, "Source", "-", column=0)
        self.source_integrity_value = self._status_row(
            card, 4, "Source integrity", "Not Run", column=0
        )
        self.liveness_value = self._status_row(
            card, 6, "Active liveness", "Not Run", column=0
        )
        self.challenge_value = self._status_row(
            card, 8, "Challenge", "Not issued", column=0
        )
        self.pad_value = self._status_row(card, 2, "Passive PAD", "Off", column=1)
        self.identity_value = self._status_row(
            card, 4, "Identity check", "Not Run", column=1
        )
        self.final_value = self._status_row(
            card, 6, "Final decision", "Not Run", column=1
        )
        self.layer_value = self._status_row(card, 8, "Detected by", "-", column=1)
        self.processing_value = self._status_row(
            card, 10, "Processing time", "-", column=1
        )
        self.reason_label = ttk.Label(
            card,
            text="",
            style="CardMuted.TLabel",
            wraplength=720,
            justify="left",
        )
        self.reason_label.grid(
            row=12, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

    def _build_results(self) -> None:
        card = Card(self.body)
        card.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        card.columnconfigure(0, weight=1)
        ttk.Label(
            card, text="Attack results and analysis", style="CardTitle.TLabel"
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            card,
            text="Clear results",
            style="Secondary.TButton",
            command=self._clear_results,
        ).grid(row=0, column=1, sticky="e")

        columns = (
            "mode",
            "source",
            "source_integrity",
            "liveness",
            "pad",
            "identity",
            "decision",
            "layer",
            "ms",
        )
        self.results_tree = ttk.Treeview(
            card, columns=columns, show="headings", height=4
        )
        headings = {
            "mode": "Mode",
            "source": "Source",
            "source_integrity": "Source integrity",
            "liveness": "Active liveness",
            "pad": "Passive PAD",
            "identity": "Identity ran",
            "decision": "Decision",
            "layer": "Detected by",
            "ms": "ms",
        }
        widths = {
            "mode": 150,
            "source": 110,
            "source_integrity": 110,
            "liveness": 110,
            "pad": 90,
            "identity": 80,
            "decision": 120,
            "layer": 110,
            "ms": 60,
        }
        for column in columns:
            self.results_tree.heading(column, text=headings[column])
            self.results_tree.column(
                column, width=widths[column], anchor="w", stretch=False
            )
        self.results_tree.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 8))

        self.analysis_label = ttk.Label(
            card,
            text=(
                "Run the same video in Original, Full Protection and Liveness "
                "Simulation modes to compare which layer detects the injection."
            ),
            style="CardMuted.TLabel",
            wraplength=1050,
            justify="left",
        )
        self.analysis_label.grid(row=2, column=0, columnspan=2, sticky="w")

    def _status_row(
        self,
        parent: tk.Misc,
        row: int,
        title: str,
        value: str,
        column: int = 0,
    ) -> ttk.Label:
        ttk.Label(parent, text=title.upper(), style="CardMuted.TLabel").grid(
            row=row, column=column, sticky="w", padx=(0, 8), pady=(5, 0)
        )
        label = ttk.Label(parent, text=value, style="CardValueSmall.TLabel")
        label.grid(row=row + 1, column=column, sticky="w", padx=(0, 8))
        return label

    def on_show(self) -> None:
        self._refresh_users()
        self._refresh_pad_state()
        self._update_challenge_display()

    def on_hide(self) -> None:
        self._stop(silent=True)

    def _browse_video(self) -> None:
        patterns = " ".join(f"*{extension}" for extension in sorted(VIDEO_EXTENSIONS))
        path = filedialog.askopenfilename(
            title="Choose a recorded video for the injection experiment",
            filetypes=[
                ("Video files", patterns),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._video_path = path
        self.path_var.set(str(Path(path)))

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

    def _on_mode_selected(self, _event=None) -> None:
        mode = self.mode_var.get()
        if mode in _MODE_DEFAULTS:
            self._apply_mode(mode)
        else:
            self._update_mode_label()

    def _apply_mode(self, mode: str) -> None:
        if mode not in _MODE_DEFAULTS:
            self._update_mode_label()
            return
        source, liveness, pad = _MODE_DEFAULTS[mode]
        self._updating_mode = True
        try:
            self.mode_var.set(mode)
            self.source_integrity_var.set(source)
            self.liveness_var.set(liveness)
            self.pad_var.set(pad)
        finally:
            self._updating_mode = False
        self._refresh_pad_state()
        self._update_mode_label()

    def _on_protection_changed(self) -> None:
        if self._updating_mode:
            return
        self.mode_var.set(MODE_CUSTOM)
        self._update_mode_label()
        self._refresh_pad_state()

    def _update_mode_label(self) -> None:
        mode = self.mode_var.get()
        descriptions = {
            MODE_ORIGINAL: "Recognition only; source integrity, active liveness and PAD are off.",
            MODE_FULL: "Source integrity, active liveness and MiniFASNetV2 PAD are all enforced.",
            MODE_LIVENESS: (
                "Source integrity and PAD are off; active liveness is enforced in a "
                "clearly labelled simulation, not a trusted-camera mode."
            ),
            MODE_CUSTOM: "Custom combination of the three protections.",
        }
        self.hint_label.configure(text=descriptions.get(mode, descriptions[MODE_CUSTOM]))

    def _refresh_pad_state(self) -> None:
        snapshot = self.ctx.anti_spoofing.runtime.snapshot()
        if not self.pad_var.get():
            if snapshot.enabled:
                self.ctx.anti_spoofing.disable()
            self.pad_badge.set("OFF", "muted")
            return
        self.ctx.anti_spoofing.enable()
        snapshot = self.ctx.anti_spoofing.runtime.snapshot()
        if snapshot.detector is not None:
            self.pad_badge.set("READY", "success")
        elif snapshot.error:
            self.pad_badge.set("ERROR", "danger")
        else:
            self.pad_badge.set("LOADING", "warning")

    def _start(self) -> None:
        if self._starting or (self._worker is not None and self._worker.is_alive()):
            return
        if not self._video_path:
            messagebox.showerror("No video selected", "Choose a recorded video file first.")
            return
        path = Path(self._video_path)
        if not path.exists() or not path.is_file():
            messagebox.showerror("Video not found", str(path))
            return
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            messagebox.showerror(
                "Unsupported video",
                "Choose a supported video extension: "
                + ", ".join(sorted(VIDEO_EXTENSIONS)),
            )
            return
        user_id = self._user_map.get(self.user_var.get())
        if not user_id:
            messagebox.showerror(
                "No target user",
                "Enroll a user first, then select that enrolled user as the target.",
            )
            return
        try:
            user = self.ctx.database.get_user(user_id)
        except (UserNotFoundError, DatabaseError, SecurityError) as exc:
            messagebox.showerror("Cannot load template", str(exc))
            return

        self._starting = True
        self._user_id = user_id
        self._template = user.embedding
        self._display_name = user.display_name
        self.start_button.configure(state="disabled")

        if self.pad_var.get():
            self.ctx.anti_spoofing.enable()
            snapshot = self.ctx.anti_spoofing.runtime.snapshot()
            if snapshot.detector is None:
                if snapshot.error:
                    self._on_pad_error(RuntimeError(snapshot.error))
                    return
                self.pipeline_badge.set("LOADING PAD", "warning")
                self.run_background(
                    self.ctx.anti_spoofing.ensure_loaded,
                    lambda _detector: self._start_worker(),
                    self._on_pad_error,
                )
                return
        else:
            self.ctx.anti_spoofing.disable()
        self._start_worker()

    def _on_pad_error(self, exc: Exception) -> None:
        self._starting = False
        self.start_button.configure(state="normal")
        self.pad_badge.set("ERROR", "danger")
        self.pipeline_badge.set("PAD ERROR", "danger")
        messagebox.showerror("Anti-spoofing unavailable", str(exc))

    def _protection_config(self) -> ProtectionConfig:
        return ProtectionConfig(
            source_integrity_enabled=bool(self.source_integrity_var.get()),
            active_liveness_enabled=bool(self.liveness_var.get()),
            label=self.mode_var.get(),
            simulation=self.mode_var.get() == MODE_LIVENESS,
        )

    def _start_worker(self) -> None:
        if not self._starting:
            return
        try:
            if not self.winfo_exists():
                self._starting = False
                return
        except tk.TclError:
            self._starting = False
            return
        if self._worker is not None and self._worker.is_alive():
            self._starting = False
            return
        config = self._protection_config()
        source = RecordedVideoSource(self._video_path or "", loop=True)
        session = self.ctx.source_integrity.start_session(source.info)
        if config.active_liveness_enabled:
            self.ctx.active_liveness.issue_challenge()
        else:
            self.ctx.active_liveness.reset("Active liveness is off for this run.")
        self._source = source
        self._session = session
        self._starting = False
        self.panel.set_message("Opening recorded video...")
        self.pipeline_badge.set("STARTING", "warning")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.pause_button.configure(state="normal")
        self.resume_button.configure(state="normal")
        self.replay_button.configure(state="normal")
        self._set_controls_enabled(False)
        self._update_challenge_display()
        self._worker = LiveCameraWorker(
            engine=self.ctx.engine,
            camera_index=0,
            template=self._template,
            user_id=self._user_id,
            threshold=self.ctx.threshold,
            frame_source=source,
            source_integrity=self.ctx.source_integrity,
            source_session=session,
            liveness=self.ctx.active_liveness,
            protection=config,
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
            self._handle_error(error.error or "Unknown video error")
            return
        if latest is not None:
            self._handle_result(latest)
        elif finished:
            self._on_finished()
            return
        self._update_playback()
        if self._worker is not None and self._worker.is_alive():
            self._poll_id = self.after(settings.PREVIEW_REFRESH_MS, self._poll)
        else:
            self._poll_id = None
            self._on_finished()

    def _update_playback(self) -> None:
        source = self._source
        if source is None:
            return
        progress = source.progress
        self.progress_var.set(progress.fraction * 100.0)
        state = "paused" if progress.paused else "playing"
        if progress.frame_count:
            text = (
                f"Playback: {state} | frame {progress.frame_index}/"
                f"{progress.frame_count} | {progress.seconds:.1f}s / "
                f"{progress.duration_seconds:.1f}s"
            )
        else:
            text = f"Playback: {state} | {progress.seconds:.1f}s"
        self.playback_label.configure(text=text)

    def _handle_result(self, result: LiveFrameResult) -> None:
        if result.annotated_frame is not None:
            self.panel.show_bgr(
                result.annotated_frame, max_size=(720, 390), fast=True
            )
        self._update_pipeline(result)
        self._update_playback()
        if result.final_decision in {FINAL_ACCEPTED, FINAL_REJECTED}:
            key = result.protection_mode
            previous = self._results.get(key)
            signature = (
                result.final_decision,
                result.layer_detected,
                result.source_integrity_status,
                result.liveness_status,
                result.pad_status,
                result.identity_executed,
            )
            if previous is None or signature != self._result_signature(previous):
                self._results[key] = result
                self._refresh_results()

    @staticmethod
    def _result_signature(result: LiveFrameResult) -> tuple:
        return (
            result.final_decision,
            result.layer_detected,
            result.source_integrity_status,
            result.liveness_status,
            result.pad_status,
            result.identity_executed,
        )

    def _update_pipeline(self, result: LiveFrameResult) -> None:
        key = (
            result.source_kind,
            result.source_integrity_status,
            result.source_integrity_reason,
            result.liveness_status,
            result.liveness_challenge,
            result.liveness_progress,
            result.liveness_required_frames,
            result.pad_status,
            result.pad_message,
            result.identity_executed,
            result.final_decision,
            result.layer_detected,
            result.decision_reason,
            round(result.processing_ms, 1),
        )
        if key == self._last_pipeline_key:
            return
        self._last_pipeline_key = key

        self.source_value.configure(
            text=f"{result.source_label}  ({result.source_kind})"
        )
        source_kind = self._status_kind(result.source_integrity_status)
        reason = result.source_integrity_reason
        if len(reason) > 72:
            reason = reason[:69] + "..."
        self.source_integrity_value.configure(
            text=f"{result.source_status_label}  {reason}".strip(),
            foreground=source_kind,
        )
        liveness_kind = self._status_kind(result.liveness_status)
        self.liveness_value.configure(
            text=result.liveness_status_label, foreground=liveness_kind
        )
        challenge = result.liveness_challenge_label or "Not issued"
        if result.liveness_required_frames:
            challenge += (
                f"  ({result.liveness_progress}/"
                f"{result.liveness_required_frames})"
            )
        if result.liveness_remaining_seconds is not None:
            challenge += f"  {result.liveness_remaining_seconds:.1f}s left"
        self.challenge_value.configure(text=challenge)
        self.pad_value.configure(
            text=self._pad_text(result.pad_status),
            foreground=self._pad_kind(result.pad_status),
        )
        self.identity_value.configure(
            text="Executed" if result.identity_executed else "Not executed",
            foreground=(
                COLORS["text"] if result.identity_executed else COLORS["muted"]
            ),
        )
        final_text, final_kind = self._decision_text(result)
        self.final_value.configure(text=final_text, foreground=final_kind)
        self.layer_value.configure(text=result.layer_detected or "-")
        self.processing_value.configure(text=f"{result.processing_ms:.1f} ms")
        self.reason_label.configure(text=result.decision_reason or result.pad_message)
        self.pipeline_badge.set(final_text, self._badge_kind(result))
        self._update_challenge_display()

    def _update_challenge_display(self) -> None:
        snapshot = self.ctx.active_liveness.snapshot()
        if not self.liveness_var.get():
            self.challenge_button.configure(state="disabled")
            return
        self.challenge_button.configure(state="normal")
        if snapshot.challenge is None:
            self.challenge_value.configure(text="Not issued")
            return
        challenge = CHALLENGE_LABELS.get(
            snapshot.challenge, snapshot.challenge_label or snapshot.challenge
        )
        if snapshot.passed:
            self.challenge_value.configure(text=f"Passed: {challenge}")
        elif snapshot.failed:
            self.challenge_value.configure(
                text=f"Failed: {challenge} | {snapshot.status_label}"
            )
        else:
            self.challenge_value.configure(
                text=(
                    f"{challenge}  ({snapshot.progress}/"
                    f"{snapshot.required_frames})"
                )
            )

    def _new_challenge(self) -> None:
        if not self.liveness_var.get():
            return
        assessment = self.ctx.active_liveness.issue_challenge()
        self.challenge_value.configure(
            text=f"{assessment.challenge_label}  (0/{assessment.required_frames})"
        )
        self.pipeline_badge.set("CHALLENGE ISSUED", "info")

    def _refresh_results(self) -> None:
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        order = [MODE_ORIGINAL, MODE_FULL, MODE_LIVENESS, MODE_CUSTOM]
        for mode in order:
            result = self._results.get(mode)
            if result is None:
                continue
            decision, _kind = self._decision_text(result)
            self.results_tree.insert(
                "",
                "end",
                values=(
                    mode,
                    result.source_label,
                    result.source_status_label,
                    result.liveness_status_label,
                    self._pad_text(result.pad_status),
                    "Yes" if result.identity_executed else "No",
                    decision,
                    result.layer_detected,
                    f"{result.processing_ms:.0f}",
                ),
            )
        self._refresh_analysis()

    def _refresh_analysis(self) -> None:
        if not self._results:
            self.analysis_label.configure(
                text=(
                    "Run the same video in Original, Full Protection and Liveness "
                    "Simulation modes to compare which layer detects the injection."
                )
            )
            return
        lines: list[str] = []
        original = self._results.get(MODE_ORIGINAL)
        full = self._results.get(MODE_FULL)
        liveness = self._results.get(MODE_LIVENESS)
        if original is not None:
            if original.final_decision == FINAL_ACCEPTED:
                lines.append(
                    "Original / Vulnerable: the replayed video was ACCEPTED. "
                    "Recognition alone did not prove a live person was present."
                )
            else:
                lines.append(
                    f"Original / Vulnerable: blocked by {original.layer_detected}."
                )
        if full is not None:
            if full.final_decision == FINAL_REJECTED:
                lines.append(
                    f"Full Protection: BLOCKED at {full.layer_detected}. "
                    "Identity verification did not run before the failed gate."
                )
            else:
                lines.append(
                    "Full Protection: the run was not blocked; inspect the pipeline "
                    "and source metadata before drawing a conclusion."
                )
        if liveness is not None:
            if liveness.final_decision == FINAL_REJECTED:
                lines.append(
                    f"Liveness Simulation: BLOCKED at {liveness.layer_detected}. "
                    "This is a controlled simulation because source integrity was off."
                )
            else:
                lines.append(
                    "Liveness Simulation: not blocked. A recording that happens to "
                    "satisfy the random challenge can still pass this layer."
                )
        if full is not None and full.source_integrity_status == SOURCE_REJECTED:
            lines.append(
                "Full Protection rejects the recorded source immediately, so active "
                "liveness is not reached; use Liveness Simulation to exercise that "
                "layer separately."
            )
        self.analysis_label.configure(text="  ".join(lines))

    def _clear_results(self) -> None:
        self._results.clear()
        self._refresh_results()

    def _handle_error(self, message: str) -> None:
        self.pipeline_badge.set("ERROR", "danger")
        self.panel.set_message("Video unavailable")
        messagebox.showerror("Video injection error", message)
        self._stop(silent=True)

    def _on_finished(self) -> None:
        self._stop(silent=True)
        self.pipeline_badge.set("STOPPED", "muted")
        self.playback_label.configure(text="Playback: stopped")

    def _pause(self) -> None:
        if self._source is not None:
            self._source.pause()
            self.playback_label.configure(text="Playback: paused")

    def _resume(self) -> None:
        if self._source is not None:
            self._source.resume()
            self.playback_label.configure(text="Playback: playing")

    def _replay(self) -> None:
        if self._source is not None:
            try:
                self._source.restart()
            except Exception as exc:
                messagebox.showerror("Cannot replay", str(exc))

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
        if self._source is not None:
            self._source.release()
        if self._session is not None:
            self.ctx.source_integrity.end_session(self._session, "video lab stopped")
            self._session = None
        self._starting = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.pause_button.configure(state="disabled")
        self.resume_button.configure(state="disabled")
        self.replay_button.configure(state="disabled")
        self._set_controls_enabled(True)
        if not silent:
            self.panel.set_message("Experiment stopped")
            self.pipeline_badge.set("IDLE", "muted")
            self.playback_label.configure(text="Playback: stopped")

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        combo_state = "readonly" if enabled else "disabled"
        self.mode_combo.configure(state=combo_state)
        self.user_combo.configure(state=combo_state)
        self.source_integrity_check.configure(state=state)
        self.liveness_check.configure(state=state)
        self.pad_check.configure(state=state)

    def _status_kind(self, status: str) -> str:
        if status in {SOURCE_TRUSTED, LIVENESS_PASSED}:
            return COLORS["success"]
        if status in {SOURCE_REJECTED, SOURCE_UNCERTAIN, "FAILED", "EXPIRED"}:
            return COLORS["danger"]
        if status in {"PENDING", "TRACKING_LOST", "FACE_CHANGED"}:
            return COLORS["warning"]
        if status == SOURCE_NOT_RUN:
            return COLORS["muted"]
        return COLORS["text"]

    def _pad_text(self, status: str) -> str:
        return {
            PAD_NOT_RUN: "Not Run",
            PAD_DISABLED: "Off",
            PAD_READY: "Ready",
            PAD_LOADING: "Loading",
            PAD_ERROR: "Error",
            PAD_NO_FACE: "No face",
            PAD_ANALYZING: "Analyzing",
            PAD_LIVE: "Live",
            PAD_SPOOF: "Spoof",
            PAD_UNCERTAIN: "Uncertain",
        }.get(status, status)

    def _pad_kind(self, status: str) -> str:
        if status == PAD_LIVE:
            return COLORS["success"]
        if status in {PAD_SPOOF, PAD_ERROR}:
            return COLORS["danger"]
        if status in {PAD_ANALYZING, PAD_LOADING, PAD_UNCERTAIN, PAD_NO_FACE}:
            return COLORS["warning"]
        return COLORS["muted"]

    def _decision_text(self, result: LiveFrameResult) -> tuple[str, str]:
        if result.final_decision == FINAL_ACCEPTED:
            return "ACCEPTED (attack succeeded)", COLORS["danger"]
        if result.final_decision == FINAL_REJECTED:
            return "BLOCKED", COLORS["success"]
        if result.final_decision == FINAL_NOT_RUN:
            return "PENDING / NOT RUN", COLORS["warning"]
        return result.final_decision, COLORS["text"]

    def _badge_kind(self, result: LiveFrameResult) -> str:
        if result.final_decision == FINAL_ACCEPTED:
            return "danger"
        if result.final_decision == FINAL_REJECTED:
            return "success"
        return "warning"
