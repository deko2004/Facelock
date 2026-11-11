"""Dark navy/blue visual theme for the Tkinter interface."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

COLORS = {
    "bg": "#08111f",
    "sidebar": "#0b1729",
    "sidebar_active": "#173a6b",
    "card": "#111f38",
    "card_alt": "#162745",
    "panel": "#060d18",
    "border": "#20375a",
    "accent": "#2f7df6",
    "accent_hover": "#4b91ff",
    "accent_dark": "#1d4f9e",
    "text": "#eaf1ff",
    "muted": "#9fb0cc",
    "subtle": "#6f84a6",
    "success": "#20c997",
    "success_dark": "#0d4f3b",
    "warning": "#ffd166",
    "warning_dark": "#5a4510",
    "danger": "#ff6b6b",
    "danger_dark": "#5a1f28",
    "entry": "#0c1729",
    "white": "#ffffff",
}

FONT_FAMILY = "Ubuntu"
MONO_FAMILY = "Ubuntu Mono"


def font(size: int, weight: str = "normal") -> tuple[str, int, str]:
    return (FONT_FAMILY, size, weight)


def mono_font(size: int, weight: str = "normal") -> tuple[str, int, str]:
    return (MONO_FAMILY, size, weight)


def setup_styles(root: tk.Misc) -> ttk.Style:
    """Configure ttk styles once per application."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    root.configure(bg=COLORS["bg"])

    style.configure("App.TFrame", background=COLORS["bg"])
    style.configure("Sidebar.TFrame", background=COLORS["sidebar"])
    style.configure("Card.TFrame", background=COLORS["card"])
    style.configure("CardAlt.TFrame", background=COLORS["card_alt"])
    style.configure("Panel.TFrame", background=COLORS["panel"])

    style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"])
    style.configure(
        "PageTitle.TLabel",
        background=COLORS["bg"],
        foreground=COLORS["text"],
        font=font(22, "bold"),
    )
    style.configure(
        "PageSubtitle.TLabel",
        background=COLORS["bg"],
        foreground=COLORS["muted"],
        font=font(10),
    )
    style.configure(
        "Sidebar.TLabel",
        background=COLORS["sidebar"],
        foreground=COLORS["text"],
    )
    style.configure(
        "SidebarTitle.TLabel",
        background=COLORS["sidebar"],
        foreground=COLORS["white"],
        font=font(15, "bold"),
    )
    style.configure(
        "SidebarSubtitle.TLabel",
        background=COLORS["sidebar"],
        foreground=COLORS["muted"],
        font=font(9),
    )
    style.configure(
        "SidebarSection.TLabel",
        background=COLORS["sidebar"],
        foreground=COLORS["subtle"],
        font=font(8, "bold"),
    )
    style.configure(
        "Card.TLabel",
        background=COLORS["card"],
        foreground=COLORS["text"],
    )
    style.configure(
        "CardTitle.TLabel",
        background=COLORS["card"],
        foreground=COLORS["text"],
        font=font(13, "bold"),
    )
    style.configure(
        "CardMuted.TLabel",
        background=COLORS["card"],
        foreground=COLORS["muted"],
        font=font(9),
    )
    style.configure(
        "CardValue.TLabel",
        background=COLORS["card"],
        foreground=COLORS["accent"],
        font=font(24, "bold"),
    )
    style.configure(
        "CardValueSmall.TLabel",
        background=COLORS["card"],
        foreground=COLORS["text"],
        font=font(15, "bold"),
    )
    style.configure(
        "Mono.TLabel",
        background=COLORS["card"],
        foreground=COLORS["muted"],
        font=mono_font(9),
    )

    style.configure(
        "Primary.TButton",
        background=COLORS["accent"],
        foreground=COLORS["white"],
        borderwidth=0,
        focusthickness=0,
        padding=(14, 8),
        font=font(10, "bold"),
    )
    style.map(
        "Primary.TButton",
        background=[
            ("active", COLORS["accent_hover"]),
            ("pressed", COLORS["accent_dark"]),
            ("disabled", COLORS["card_alt"]),
        ],
        foreground=[("disabled", COLORS["subtle"])],
    )
    style.configure(
        "Secondary.TButton",
        background=COLORS["card_alt"],
        foreground=COLORS["text"],
        borderwidth=0,
        focusthickness=0,
        padding=(12, 7),
        font=font(10),
    )
    style.map(
        "Secondary.TButton",
        background=[
            ("active", COLORS["border"]),
            ("pressed", COLORS["accent_dark"]),
            ("disabled", COLORS["card"]),
        ],
        foreground=[("disabled", COLORS["subtle"])],
    )
    style.configure(
        "Danger.TButton",
        background=COLORS["danger_dark"],
        foreground=COLORS["white"],
        borderwidth=0,
        focusthickness=0,
        padding=(12, 7),
        font=font(10, "bold"),
    )
    style.map(
        "Danger.TButton",
        background=[("active", COLORS["danger"]), ("pressed", "#8b2f3a")],
    )
    style.configure(
        "Ghost.TButton",
        background=COLORS["bg"],
        foreground=COLORS["muted"],
        borderwidth=0,
        focusthickness=0,
        padding=(8, 6),
        font=font(9),
    )
    style.map(
        "Ghost.TButton",
        background=[("active", COLORS["card"])],
        foreground=[("active", COLORS["text"])],
    )
    style.configure(
        "Nav.TButton",
        background=COLORS["sidebar"],
        foreground=COLORS["muted"],
        borderwidth=0,
        focusthickness=0,
        padding=(16, 11),
        anchor="w",
        font=font(10),
    )
    style.map(
        "Nav.TButton",
        background=[("active", COLORS["card"])],
        foreground=[("active", COLORS["text"])],
    )
    style.configure(
        "NavActive.TButton",
        background=COLORS["sidebar_active"],
        foreground=COLORS["white"],
        borderwidth=0,
        focusthickness=0,
        padding=(16, 11),
        anchor="w",
        font=font(10, "bold"),
    )
    style.map(
        "NavActive.TButton",
        background=[("active", COLORS["sidebar_active"])],
        foreground=[("active", COLORS["white"])],
    )

    style.configure(
        "TEntry",
        fieldbackground=COLORS["entry"],
        foreground=COLORS["text"],
        insertcolor=COLORS["text"],
        bordercolor=COLORS["border"],
        lightcolor=COLORS["border"],
        darkcolor=COLORS["border"],
        padding=7,
    )
    style.map(
        "TEntry",
        bordercolor=[("focus", COLORS["accent"])],
        lightcolor=[("focus", COLORS["accent"])],
    )
    style.configure(
        "TCombobox",
        fieldbackground=COLORS["entry"],
        background=COLORS["card_alt"],
        foreground=COLORS["text"],
        arrowcolor=COLORS["muted"],
        bordercolor=COLORS["border"],
        lightcolor=COLORS["border"],
        darkcolor=COLORS["border"],
        padding=6,
    )
    style.map(
        "TCombobox",
        fieldbackground=[
            ("readonly", COLORS["entry"]),
            ("disabled", COLORS["card"]),
        ],
        foreground=[("readonly", COLORS["text"]), ("disabled", COLORS["subtle"])],
        selectbackground=[("readonly", COLORS["entry"])],
        selectforeground=[("readonly", COLORS["text"])],
    )
    style.configure(
        "TCheckbutton",
        background=COLORS["card"],
        foreground=COLORS["text"],
        font=font(9),
        focuscolor=COLORS["card"],
    )
    style.map(
        "TCheckbutton",
        background=[("active", COLORS["card"])],
        indicatorcolor=[("selected", COLORS["accent"]), ("!selected", COLORS["entry"])],
    )
    style.configure(
        "TRadiobutton",
        background=COLORS["card"],
        foreground=COLORS["text"],
        font=font(9),
    )
    style.map(
        "TRadiobutton",
        background=[("active", COLORS["card"])],
        indicatorcolor=[("selected", COLORS["accent"]), ("!selected", COLORS["entry"])],
    )
    style.configure(
        "Treeview",
        background=COLORS["card"],
        fieldbackground=COLORS["card"],
        foreground=COLORS["text"],
        rowheight=28,
        borderwidth=0,
        font=font(9),
    )
    style.configure(
        "Treeview.Heading",
        background=COLORS["card_alt"],
        foreground=COLORS["muted"],
        relief="flat",
        font=font(9, "bold"),
        padding=(8, 6),
    )
    style.map(
        "Treeview",
        background=[("selected", COLORS["accent_dark"])],
        foreground=[("selected", COLORS["white"])],
    )
    style.map(
        "Treeview.Heading",
        background=[("active", COLORS["border"])],
    )
    style.configure(
        "Vertical.TScrollbar",
        background=COLORS["card_alt"],
        troughcolor=COLORS["bg"],
        bordercolor=COLORS["bg"],
        arrowcolor=COLORS["muted"],
    )
    style.configure(
        "Horizontal.TScrollbar",
        background=COLORS["card_alt"],
        troughcolor=COLORS["bg"],
        bordercolor=COLORS["bg"],
        arrowcolor=COLORS["muted"],
    )
    style.configure("TSeparator", background=COLORS["border"])
    style.configure(
        "TLabelframe",
        background=COLORS["card"],
        foreground=COLORS["muted"],
        bordercolor=COLORS["border"],
    )
    style.configure(
        "TLabelframe.Label",
        background=COLORS["card"],
        foreground=COLORS["muted"],
        font=font(9, "bold"),
    )
    return style
