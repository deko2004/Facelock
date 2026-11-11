from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from core.app_context import AppContext
from gui.main_window import PAGE_CLASSES, MainWindow


@pytest.mark.skipif(
    not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
    reason="Tkinter requires a graphical display",
)
def test_all_pages_construct_and_show(tmp_path: Path) -> None:
    context = AppContext.create(
        db_path=tmp_path / "gui.db",
        key_path=tmp_path / "gui.key",
        settings_path=tmp_path / "gui_settings.json",
        threshold=0.363,
    )
    app = MainWindow(context)
    app.withdraw()
    try:
        app.update()
        for name in PAGE_CLASSES:
            app.show_page(name)
            app.update()
            page = app.pages[name]
            page.on_show()
            app.update()
            assert page.winfo_exists()
    finally:
        app._on_close()


@pytest.mark.skipif(
    not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
    reason="Tkinter requires a graphical display",
)
def test_background_results_are_delivered_on_main_thread(tmp_path: Path) -> None:
    context = AppContext.create(
        db_path=tmp_path / "bg.db",
        key_path=tmp_path / "bg.key",
        settings_path=tmp_path / "bg_settings.json",
        threshold=0.363,
    )
    app = MainWindow(context)
    app.withdraw()
    try:
        app.update()
        page = app.pages["dashboard"]
        received: list[object] = []
        page.run_background(lambda: 42, received.append)
        deadline = time.time() + 5
        while not received and time.time() < deadline:
            app.update()
            time.sleep(0.02)
        assert received == [42]
    finally:
        app._on_close()
