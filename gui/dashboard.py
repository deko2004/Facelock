"""Dashboard page with live counts, pipeline summary and audit activity."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from core.camera_manager import list_video_devices
from gui.styles import COLORS, font
from gui.widgets import BasePage, Card, PageHeader, StatCard, bullet_text


class DashboardPage(BasePage):
    def __init__(self, parent: tk.Misc, controller) -> None:
        super().__init__(parent, controller)
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=1)

        PageHeader(
            self,
            "Dashboard",
            "A local-only demonstration of face detection, SFace embeddings, "
            "1:1 verification and threshold-based authentication decisions.",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 18))

        stats = ttk.Frame(self, style="App.TFrame")
        stats.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 16))
        for index in range(4):
            stats.grid_columnconfigure(index, weight=1, uniform="stats")

        self.users_stat = StatCard(
            stats, "Enrolled users", "0", "Encrypted templates in the local database"
        )
        self.users_stat.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.threshold_stat = StatCard(
            stats, "Match threshold", f"{self.ctx.threshold:.3f}",
            "Cosine similarity acceptance boundary",
        )
        self.threshold_stat.grid(row=0, column=1, sticky="nsew", padx=8)
        self.camera_stat = StatCard(
            stats, "Camera devices", "0", "Detected Linux /dev/video* nodes"
        )
        self.camera_stat.grid(row=0, column=2, sticky="nsew", padx=8)
        self.storage_stat = StatCard(
            stats, "Template storage", "Fernet", "AES-128-CBC + HMAC-SHA256 at rest"
        )
        self.storage_stat.grid(row=0, column=3, sticky="nsew", padx=(8, 0))

        pipeline = Card(self)
        pipeline.grid(row=2, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        ttk.Label(pipeline, text="Biometric pipeline", style="CardTitle.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            pipeline,
            text=(
                "1. Image acquisition  ->  2. YuNet detection  ->  "
                "3. SFace alignment  ->  4. 128-D embedding  ->  "
                "5. Encrypted template  ->  6. Cosine matching  ->  "
                "7. Threshold comparison  ->  8. Authentication decision"
            ),
            style="Card.TLabel",
            wraplength=560,
            justify="left",
        ).pack(anchor="w", pady=(10, 12))
        ttk.Label(
            pipeline,
            text=(
                f"Detector: {self.ctx.engine.detector_label}    "
                f"Recognizer: {self.ctx.engine.recognizer_label}"
            ),
            style="CardMuted.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            pipeline,
            text=(
                "The score is a cosine similarity, not a probability of identity. "
                "Optional MiniFASNetV2 PAD is available on Live Camera; it is off "
                "by default and is not a certified liveness guarantee."
            ),
            style="CardMuted.TLabel",
            wraplength=560,
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        security = Card(self)
        security.grid(row=2, column=1, sticky="nsew", padx=(8, 0), pady=(0, 8))
        ttk.Label(security, text="Security posture", style="CardTitle.TLabel").pack(
            anchor="w"
        )
        bullet_text(
            security,
            [
                "All processing stays on this computer.",
                "Only encrypted templates are stored, not enrolment photographs.",
                "The Fernet key is stored separately from the database.",
                "Raw feature vectors are never written to logs.",
                "Users can be deleted together with their template.",
                "Optional PAD on Live Camera; OFF mode can still match a printed photo.",
            ],
            wraplength=380,
        ).pack(anchor="w", pady=(10, 0))

        activity = Card(self)
        activity.grid(row=3, column=0, sticky="nsew", padx=(0, 8), pady=(8, 0))
        ttk.Label(activity, text="Recent audit activity", style="CardTitle.TLabel").pack(
            anchor="w", pady=(0, 8)
        )
        self.activity_tree = ttk.Treeview(
            activity,
            columns=("time", "event", "user", "detail"),
            show="headings",
            height=7,
        )
        for column, title, width in (
            ("time", "Time", 135),
            ("event", "Event", 120),
            ("user", "User", 90),
            ("detail", "Detail", 280),
        ):
            self.activity_tree.heading(column, text=title)
            self.activity_tree.column(column, width=width, anchor="w")
        self.activity_tree.pack(fill="both", expand=True)

        actions = Card(self)
        actions.grid(row=3, column=1, sticky="nsew", padx=(8, 0), pady=(8, 0))
        ttk.Label(actions, text="Quick actions", style="CardTitle.TLabel").pack(
            anchor="w", pady=(0, 8)
        )
        ttk.Button(
            actions,
            text="Compare two images",
            style="Primary.TButton",
            command=lambda: self.controller.show_page("compare"),
        ).pack(fill="x", pady=(0, 6))
        ttk.Button(
            actions,
            text="Register a user",
            style="Secondary.TButton",
            command=lambda: self.controller.show_page("enrollment"),
        ).pack(fill="x", pady=4)
        ttk.Button(
            actions,
            text="Start live camera",
            style="Secondary.TButton",
            command=lambda: self.controller.show_page("live"),
        ).pack(fill="x", pady=4)
        ttk.Button(
            actions,
            text="Run evaluation",
            style="Secondary.TButton",
            command=lambda: self.controller.show_page("evaluation"),
        ).pack(fill="x", pady=4)

    def on_show(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        try:
            user_count = self.ctx.database.count_users()
            events = self.ctx.database.recent_events(8)
        except Exception as exc:
            user_count = 0
            events = []
            self.users_stat.set_value("Error", COLORS["danger"])
            self.users_stat.caption_label.configure(text=str(exc))
        else:
            self.users_stat.set_value(str(user_count))
            self.users_stat.caption_label.configure(
                text="Encrypted templates in the local database"
            )
        self.threshold_stat.set_value(f"{self.ctx.threshold:.3f}")
        try:
            cameras = list_video_devices()
        except Exception:
            cameras = []
        self.camera_stat.set_value(str(len(cameras)))
        self.camera_stat.caption_label.configure(
            text="Detected Linux /dev/video* nodes"
            if cameras
            else "No camera device detected"
        )

        for item in self.activity_tree.get_children():
            self.activity_tree.delete(item)
        if not events:
            self.activity_tree.insert(
                "", "end", values=("-", "No activity yet", "-", "-")
            )
        else:
            for event in events:
                timestamp = event.created_at.replace("T", " ")[:19]
                self.activity_tree.insert(
                    "",
                    "end",
                    values=(
                        timestamp,
                        event.event_type,
                        event.user_id or "-",
                        event.detail or "-",
                    ),
                )
