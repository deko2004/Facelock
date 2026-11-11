"""Reusable Tkinter widgets for the dark navy interface."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageTk

from gui.styles import COLORS, font


class BasePage(ttk.Frame):
    """Base class for pages with background-work helpers and lifecycle hooks."""

    def __init__(self, parent: tk.Misc, controller) -> None:
        super().__init__(parent, style="App.TFrame", padding=24)
        self.controller = controller
        self.ctx = controller.ctx
        self._result_queue: queue.Queue = queue.Queue()
        self._poll_job: str | None = None
        self.after(50, self._poll_results)

    def on_show(self) -> None:
        """Called when the page becomes visible."""

    def on_hide(self) -> None:
        """Called when the page is hidden or the application closes."""

    def run_background(
        self,
        work: Callable,
        on_success: Callable,
        on_error: Callable | None = None,
    ) -> None:
        """Run *work* in a daemon thread and deliver the result on the Tk thread.

        Tkinter is not thread-safe, so the worker only puts its result on a
        queue.  The main thread polls that queue and invokes the callbacks.
        """

        def runner() -> None:
            try:
                result = work()
            except Exception as exc:  # delivered to the GUI thread
                result = exc
            self._result_queue.put((on_success, on_error, result))

        threading.Thread(target=runner, daemon=True).start()

    def _poll_results(self) -> None:
        self._poll_job = None
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return

        while True:
            try:
                on_success, on_error, result = self._result_queue.get_nowait()
            except queue.Empty:
                break
            try:
                if isinstance(result, Exception):
                    if on_error is not None:
                        on_error(result)
                    else:
                        messagebox.showerror("Operation failed", str(result))
                else:
                    on_success(result)
            except Exception as exc:
                messagebox.showerror("Operation failed", str(exc))

        try:
            self._poll_job = self.after(50, self._poll_results)
        except tk.TclError:
            self._poll_job = None


class PageHeader(ttk.Frame):
    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        subtitle: str = "",
        right_widget: tk.Widget | None = None,
    ) -> None:
        super().__init__(parent, style="App.TFrame")
        self.grid_columnconfigure(0, weight=1)
        title_label = ttk.Label(self, text=title, style="PageTitle.TLabel")
        title_label.grid(row=0, column=0, sticky="w")
        if subtitle:
            ttk.Label(
                self, text=subtitle, style="PageSubtitle.TLabel", wraplength=900
            ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        if right_widget is not None:
            right_widget.grid(row=0, column=1, rowspan=2, sticky="e", padx=(16, 0))


class Card(ttk.Frame):
    def __init__(self, parent: tk.Misc, padding: int = 18, style: str = "Card.TFrame"):
        super().__init__(parent, style=style, padding=padding)


class StatCard(Card):
    def __init__(
        self, parent: tk.Misc, title: str, value: str = "-", caption: str = ""
    ) -> None:
        super().__init__(parent, padding=16)
        self.columnconfigure(0, weight=1)
        ttk.Label(self, text=title.upper(), style="CardMuted.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.value_label = ttk.Label(self, text=value, style="CardValue.TLabel")
        self.value_label.grid(row=1, column=0, sticky="w", pady=(6, 2))
        self.caption_label = ttk.Label(
            self, text=caption, style="CardMuted.TLabel", wraplength=220
        )
        self.caption_label.grid(row=2, column=0, sticky="w")

    def set_value(self, value: str, colour: str | None = None) -> None:
        self.value_label.configure(text=value)
        if colour:
            self.value_label.configure(foreground=colour)


class Badge(tk.Label):
    """Small coloured status pill."""

    _KINDS = {
        "success": (COLORS["success_dark"], COLORS["success"]),
        "danger": (COLORS["danger_dark"], COLORS["danger"]),
        "warning": (COLORS["warning_dark"], COLORS["warning"]),
        "info": (COLORS["accent_dark"], COLORS["white"]),
        "muted": (COLORS["card_alt"], COLORS["muted"]),
    }

    def __init__(self, parent: tk.Misc, text: str = "", kind: str = "muted") -> None:
        super().__init__(
            parent,
            text=text,
            font=font(9, "bold"),
            padx=10,
            pady=4,
            borderwidth=0,
        )
        self.set(text, kind)

    def set(self, text: str, kind: str = "muted") -> None:
        background, foreground = self._KINDS.get(kind, self._KINDS["muted"])
        self.configure(text=text, bg=background, fg=foreground)


class ImagePanel(ttk.Frame):
    """An image display with a fixed pixel height and aspect-preserving scaling."""

    def __init__(
        self,
        parent: tk.Misc,
        height: int = 260,
        max_size: tuple[int, int] = (420, 260),
        placeholder: str = "No image selected",
    ) -> None:
        super().__init__(parent, style="Panel.TFrame")
        self._max_size = max_size
        self._photo: ImageTk.PhotoImage | None = None
        self._container = tk.Frame(self, bg=COLORS["panel"], height=height)
        self._container.pack(fill="both", expand=True)
        self._container.pack_propagate(False)
        self._label = tk.Label(
            self._container,
            bg=COLORS["panel"],
            fg=COLORS["subtle"],
            font=font(10),
            justify="center",
            text=placeholder,
        )
        self._label.pack(fill="both", expand=True)

    def show_bgr(
        self,
        image: np.ndarray,
        max_size: tuple[int, int] | None = None,
    ) -> None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.show_pil(Image.fromarray(rgb), max_size=max_size)

    def show_pil(
        self,
        image: Image.Image,
        max_size: tuple[int, int] | None = None,
    ) -> None:
        target = max_size or self._max_size
        copy = image.copy()
        copy.thumbnail(target, Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(copy)
        self._label.configure(image=self._photo, text="")

    def clear(self, message: str = "No image selected") -> None:
        self._photo = None
        self._label.configure(image="", text=message)

    def set_message(self, message: str) -> None:
        self._photo = None
        self._label.configure(image="", text=message)


class ScrollableFrame(ttk.Frame):
    """A vertically scrollable container for long documentation-style pages."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, style="App.TFrame")
        self.canvas = tk.Canvas(
            self, bg=COLORS["bg"], highlightthickness=0, borderwidth=0
        )
        self.scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.inner = ttk.Frame(self.canvas, style="App.TFrame")
        self._window = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw"
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.inner.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)
        self.inner.bind("<Leave>", self._unbind_wheel)

    def _on_inner_configure(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        self.canvas.itemconfigure(self._window, width=event.width)

    def _bind_wheel(self, _event=None) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", self._on_wheel)
        self.canvas.bind_all("<Button-5>", self._on_wheel)

    def _unbind_wheel(self, _event=None) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_wheel(self, event) -> None:
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(delta, "units")


def section_label(parent: tk.Misc, text: str, style: str = "CardTitle.TLabel") -> ttk.Label:
    return ttk.Label(parent, text=text, style=style)


def bullet_text(parent: tk.Misc, lines: list[str], wraplength: int = 800) -> ttk.Label:
    body = "\n".join(f"- {line}" for line in lines)
    return ttk.Label(
        parent,
        text=body,
        style="Card.TLabel",
        justify="left",
        wraplength=wraplength,
    )
