"""Security, privacy and spoofing-limitation page with a safe demo."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from core.database import DatabaseError
from core.verification import VerificationResult
from gui.styles import COLORS, font
from gui.widgets import (
    Badge,
    BasePage,
    Card,
    PageHeader,
    ScrollableFrame,
    bullet_text,
)


class SecurityPage(BasePage):
    def __init__(self, parent: tk.Misc, controller) -> None:
        super().__init__(parent, controller)
        self._user_map: dict[str, str] = {}
        self._demo_path: str | None = None

        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True)
        body = scroll.inner
        body.grid_columnconfigure(0, weight=1)

        PageHeader(
            body,
            "Security & Privacy",
            "Security is part of the design, but this is an academic demonstration "
            "and not a production identity-verification system.",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 14))

        controls = Card(body)
        controls.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(controls, text="Implemented controls", style="CardTitle.TLabel").pack(
            anchor="w"
        )
        bullet_text(
            controls,
            [
                "Local-only processing: images and embeddings never leave this computer.",
                "No cloud API, telemetry or network upload of biometric data.",
                "Biometric templates are encrypted at rest with Fernet "
                "(AES-128-CBC + HMAC-SHA256).",
                "The encryption key is stored in database/templates.key, separately "
                "from biometric.db.",
                "Database and key files are restricted to the current user "
                "(0600 / 0700) on supported filesystems.",
                "Only encrypted templates are stored; enrolment photographs are not saved.",
                "Raw feature vectors are never written to logs or shown in the interface.",
                "Informed-consent checkbox before enrolment and a delete-user action.",
                "Input validation and safe image decoding reduce malformed-input risk.",
            ],
            wraplength=1100,
        ).pack(anchor="w", pady=(8, 0))

        limits = Card(body)
        limits.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(
            limits, text="Encryption limitations and key management", style="CardTitle.TLabel"
        ).pack(anchor="w")
        ttk.Label(
            limits,
            text=(
                "Encryption protects a copied database from casual inspection. It "
                "does not protect against an attacker who can read the key file, "
                "who controls the running process, or who has administrator access. "
                "There is no hardware-backed key store, no secure enclave, no key "
                "rotation and no tamper-evident audit log. A stolen key makes the "
                "templates recoverable. Templates are also not cancellable in the "
                "cryptographic sense: if a template is compromised, the underlying "
                "face cannot be reissued like a password."
            ),
            style="Card.TLabel",
            wraplength=1100,
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        spoof = Card(body)
        spoof.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(
            spoof, text="Spoofing and liveness detection", style="CardTitle.TLabel"
        ).pack(anchor="w")
        bullet_text(
            spoof,
            [
                "Printed-photo attack: a photograph of the enrolled person is shown to the camera.",
                "Face-mask attack: a mask or synthetic face is presented.",
                "Replayed-video attack: a recorded video is shown on a phone or monitor.",
                "Manipulated/deepfake samples: synthetic or edited imagery is presented.",
                "Presentation Attack Detection (PAD) tries to detect these attacks before matching.",
                "Liveness detection tries to prove that a live person is physically present.",
            ],
            wraplength=1100,
        ).pack(anchor="w", pady=(8, 0))
        tk.Label(
            spoof,
            text=(
                "LIMITATION: YuNet and SFace perform detection and recognition only. "
                "The Live Camera page has an optional MiniFASNetV2 PAD toggle, but it "
                "is a demonstration, not certified liveness detection. With PAD OFF, "
                "a static photograph can be accepted; even with PAD ON, determined "
                "presentation attacks may succeed. A successful match is not proof "
                "that the person is alive or physically present."
            ),
            bg=COLORS["warning_dark"],
            fg=COLORS["warning"],
            font=font(9, "bold"),
            wraplength=1050,
            justify="left",
            padx=10,
            pady=8,
        ).pack(anchor="w", pady=(10, 0))

        demo = Card(body)
        demo.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        demo.columnconfigure(1, weight=1)
        ttk.Label(
            demo,
            text="Controlled presentation-attack demonstration (educational)",
            style="CardTitle.TLabel",
        ).grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(
            demo,
            text=(
                "Choose an enrolled user and a still photograph of that person. "
                "The system will probably accept the still image because this demo "
                "calls the verification service directly and bypasses the Live "
                "Camera PAD check. It is a safe demonstration of why recognition "
                "alone is not liveness, not a real attack tool."
            ),
            style="CardMuted.TLabel",
            wraplength=1100,
            justify="left",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 10))

        ttk.Label(demo, text="Enrolled user", style="Card.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 8)
        )
        self.demo_user_var = tk.StringVar()
        self.demo_user_combo = ttk.Combobox(
            demo, textvariable=self.demo_user_var, state="readonly"
        )
        self.demo_user_combo.grid(row=2, column=1, sticky="ew")
        ttk.Button(
            demo, text="Refresh", style="Secondary.TButton", command=self._refresh_users
        ).grid(row=2, column=2, padx=8)

        ttk.Button(
            demo, text="Choose still image", style="Secondary.TButton", command=self._browse
        ).grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.demo_path_label = ttk.Label(
            demo, text="No image selected", style="CardMuted.TLabel", wraplength=700
        )
        self.demo_path_label.grid(row=3, column=1, columnspan=2, sticky="w", pady=(8, 0))

        self.demo_button = ttk.Button(
            demo,
            text="Run static-image verification",
            style="Primary.TButton",
            command=self._run_demo,
        )
        self.demo_button.grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.demo_badge = Badge(demo, "NOT RUN", "muted")
        self.demo_badge.grid(row=4, column=1, sticky="w", pady=(10, 0))
        self.demo_score_label = ttk.Label(
            demo, text="Similarity: -", style="CardValueSmall.TLabel"
        )
        self.demo_score_label.grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.demo_message_label = ttk.Label(
            demo,
            text="",
            style="CardMuted.TLabel",
            wraplength=1050,
            justify="left",
        )
        self.demo_message_label.grid(row=6, column=0, columnspan=4, sticky="w", pady=(6, 0))

        not_protected = Card(body)
        not_protected.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(
            not_protected,
            text="What this project does not protect against",
            style="CardTitle.TLabel",
        ).pack(anchor="w")
        bullet_text(
            not_protected,
            [
                "Presentation attacks, masks, replay and deepfakes beyond what the optional MiniFASNetV2 demo can detect.",
                "An attacker with read access to both the database and the key file.",
                "An attacker with administrator/root access or debugger access.",
                "Template inversion or reconstruction attacks against face embeddings.",
                "Biased or uncalibrated matching performance across demographics.",
                "Coercion, insider misuse or a user enrolling someone else's face.",
                "Network attacks: there is no network service, so there is no network attack surface in this demo.",
            ],
            wraplength=1100,
        ).pack(anchor="w", pady=(8, 0))

        self._refresh_users()

    def on_show(self) -> None:
        self._refresh_users()

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
        self.demo_user_combo.configure(values=values)
        if self.demo_user_var.get() not in values:
            self.demo_user_combo.current(0)

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a still image for the demonstration",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._demo_path = path
        self.demo_path_label.configure(text=str(Path(path)))

    def _run_demo(self) -> None:
        user_id = self._user_map.get(self.demo_user_var.get())
        if user_id is None:
            messagebox.showwarning("No user", "Select an enrolled user first.")
            return
        if not self._demo_path:
            messagebox.showwarning("No image", "Choose a still image first.")
            return
        self.demo_button.configure(state="disabled")
        self.demo_badge.set("CHECKING", "warning")
        self.demo_score_label.configure(text="Similarity: ...")
        self.demo_message_label.configure(text="Running 1:1 verification on the still image...")

        self.run_background(
            lambda: self.ctx.verification.verify_image(user_id, self._demo_path, 0),
            self._demo_result,
            self._demo_failed,
        )

    def _demo_result(self, result: VerificationResult) -> None:
        self.demo_button.configure(state="normal")
        if not result.success:
            self.demo_badge.set("NOT VERIFIED", "danger")
            self.demo_score_label.configure(text="Similarity: -")
            self.demo_message_label.configure(text=result.message)
            return
        assert result.similarity is not None
        self.demo_badge.set(
            "VERIFIED (STATIC IMAGE)" if result.verified else "NOT VERIFIED",
            "warning" if result.verified else "danger",
        )
        self.demo_score_label.configure(
            text=f"Similarity: {result.similarity:.4f}  (threshold {result.threshold:.3f})"
        )
        if result.verified:
            self.demo_message_label.configure(
                text=(
                    "The still image was accepted. This demonstrates that the 1:1 "
                    "verification service alone has no liveness check: matching a "
                    "face is not proof that a live person is present. Use Live Camera "
                    "with Anti-Spoofing Protection ON to exercise the optional PAD."
                ),
                foreground=COLORS["warning"],
            )
        else:
            self.demo_message_label.configure(
                text=(
                    "This still image was rejected, but rejection does not prove "
                    "liveness either. This demo bypasses the optional Live Camera "
                    "PAD check."
                ),
                foreground=COLORS["muted"],
            )

    def _demo_failed(self, exc: Exception) -> None:
        self.demo_button.configure(state="normal")
        self.demo_badge.set("ERROR", "danger")
        self.demo_message_label.configure(text=str(exc))
        messagebox.showerror("Demonstration failed", str(exc))
