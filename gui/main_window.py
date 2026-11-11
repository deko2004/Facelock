"""Main application window with the navigation sidebar and page router."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from config import settings
from core.app_context import AppContext
from gui.about_page import AboutPage
from gui.compare_images import CompareImagesPage
from gui.dashboard import DashboardPage
from gui.database_page import DatabasePage
from gui.enrollment_page import EnrollmentPage
from gui.evaluation_page import EvaluationPage
from gui.live_camera import LiveCameraPage
from gui.security_page import SecurityPage
from gui.styles import COLORS, font, setup_styles
from gui.verification_page import VerificationPage

PAGE_CLASSES = {
    "dashboard": DashboardPage,
    "compare": CompareImagesPage,
    "live": LiveCameraPage,
    "enrollment": EnrollmentPage,
    "verification": VerificationPage,
    "database": DatabasePage,
    "evaluation": EvaluationPage,
    "security": SecurityPage,
    "about": AboutPage,
}

NAV_SECTIONS = [
    (
        "OVERVIEW",
        [("dashboard", "Dashboard")],
    ),
    (
        "BIOMETRIC WORKFLOWS",
        [
            ("compare", "Compare Two Images"),
            ("live", "Live Camera"),
            ("enrollment", "Register User"),
            ("verification", "Verify Identity"),
        ],
    ),
    (
        "DATA & ANALYSIS",
        [
            ("database", "User Database"),
            ("evaluation", "Evaluation & Metrics"),
        ],
    ),
    (
        "ACADEMIC",
        [
            ("security", "Security & Privacy"),
            ("about", "About the Project"),
        ],
    ),
]


class MainWindow(tk.Tk):
    def __init__(self, context: AppContext) -> None:
        super().__init__()
        self.ctx = context
        self.title(settings.APP_NAME)
        self.geometry("1320x860")
        self.minsize(1120, 720)
        self.configure(bg=COLORS["bg"])
        setup_styles(self)

        self.pages: dict[str, ttk.Frame] = {}
        self.nav_buttons: dict[str, ttk.Button] = {}
        self.current_page_name: str | None = None
        self.current_page: ttk.Frame | None = None

        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.show_page("dashboard")

    def _build_layout(self) -> None:
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        sidebar = ttk.Frame(self, style="Sidebar.TFrame", width=252)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(1, weight=1)

        brand = ttk.Frame(sidebar, style="Sidebar.TFrame", padding=(20, 22, 20, 12))
        brand.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            brand,
            text="BIOMETRIC\nAUTHENTICATION\nSYSTEM",
            style="SidebarTitle.TLabel",
            justify="left",
        ).pack(anchor="w")
        ttk.Label(
            brand,
            text="Face Recognition and Verification",
            style="SidebarSubtitle.TLabel",
            wraplength=205,
        ).pack(anchor="w", pady=(6, 0))

        nav = ttk.Frame(sidebar, style="Sidebar.TFrame", padding=(12, 4))
        nav.grid(row=1, column=0, sticky="nsew")
        for section, items in NAV_SECTIONS:
            ttk.Label(
                nav, text=section, style="SidebarSection.TLabel"
            ).pack(anchor="w", padx=8, pady=(14, 5))
            for name, label in items:
                button = ttk.Button(
                    nav,
                    text=label,
                    style="Nav.TButton",
                    command=lambda page=name: self.show_page(page),
                )
                button.pack(fill="x", pady=1)
                self.nav_buttons[name] = button

        footer = ttk.Frame(sidebar, style="Sidebar.TFrame", padding=(18, 10, 18, 18))
        footer.grid(row=2, column=0, sticky="ew")
        tk.Label(
            footer,
            text="LOCAL PROCESSING ONLY",
            bg=COLORS["success_dark"],
            fg=COLORS["success"],
            font=font(8, "bold"),
            padx=8,
            pady=4,
        ).pack(anchor="w")
        ttk.Label(
            footer,
            text=f"v{settings.APP_VERSION}  |  YuNet + SFace",
            style="SidebarSubtitle.TLabel",
        ).pack(anchor="w", pady=(8, 0))
        ttk.Label(
            footer,
            text="Academic demonstration",
            style="SidebarSubtitle.TLabel",
        ).pack(anchor="w")

        content = ttk.Frame(self, style="App.TFrame")
        content.grid(row=0, column=1, sticky="nsew")
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)

        self.page_container = ttk.Frame(content, style="App.TFrame")
        self.page_container.grid(row=0, column=0, sticky="nsew")
        self.page_container.grid_rowconfigure(0, weight=1)
        self.page_container.grid_columnconfigure(0, weight=1)

        status = tk.Frame(content, bg=COLORS["card_alt"], height=30)
        status.grid(row=1, column=0, sticky="ew")
        status.grid_propagate(False)
        self.model_status = tk.Label(
            status,
            text="Loading models...",
            bg=COLORS["card_alt"],
            fg=COLORS["muted"],
            font=font(8),
            padx=16,
        )
        self.model_status.pack(side="left", pady=6)
        self.threshold_status = tk.Label(
            status,
            text="",
            bg=COLORS["card_alt"],
            fg=COLORS["muted"],
            font=font(8),
            padx=16,
        )
        self.threshold_status.pack(side="right", pady=6)
        self._update_status()

    def show_page(self, name: str) -> None:
        if name == self.current_page_name:
            return
        if self.current_page is not None:
            try:
                self.current_page.on_hide()
            except Exception:
                pass
        if name not in self.pages:
            page_class = PAGE_CLASSES[name]
            page = page_class(self.page_container, self)
            page.grid(row=0, column=0, sticky="nsew")
            self.pages[name] = page
        page = self.pages[name]
        page.tkraise()
        self.current_page_name = name
        self.current_page = page
        for page_name, button in self.nav_buttons.items():
            button.configure(
                style="NavActive.TButton" if page_name == name else "Nav.TButton"
            )
        try:
            page.on_show()
        except Exception:
            pass
        self._update_status()

    def set_threshold(self, value: float) -> float:
        threshold = self.ctx.set_threshold(value)
        self._update_status()
        return threshold

    def _update_status(self) -> None:
        if not hasattr(self, "model_status"):
            return
        self.model_status.configure(
            text=(
                f"{self.ctx.engine.detector_label} + "
                f"{self.ctx.engine.recognizer_label} loaded"
            )
        )
        self.threshold_status.configure(
            text=(
                f"Decision threshold {self.ctx.threshold:.3f}  |  "
                "Local-only processing"
            )
        )

    def _on_close(self) -> None:
        for page in self.pages.values():
            try:
                page.on_hide()
            except Exception:
                pass
        try:
            self.ctx.shutdown()
        finally:
            self.destroy()
