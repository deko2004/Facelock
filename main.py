"""Application entry point.

Run with:
    ./run.sh
or:
    venv/bin/python main.py
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

from config import settings  # noqa: E402
from core.app_context import AppContext  # noqa: E402
from gui.main_window import MainWindow  # noqa: E402


def _show_fatal_error(message: str) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Cannot start application", message)
        root.destroy()
    except Exception:
        print(f"ERROR: {message}", file=sys.stderr)


def _check_environment() -> int:
    try:
        context = AppContext.create()
    except Exception as exc:
        print(f"Startup check failed: {exc}", file=sys.stderr)
        return 1
    print(f"{settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Detector: {context.engine.detector_label}")
    print(f"Recognizer: {context.engine.recognizer_label}")
    print(f"Enrolled users: {context.database.count_users()}")
    print(f"Threshold: {context.threshold:.3f}")
    print("Models loaded successfully.")
    context.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=settings.APP_NAME)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify models and database, print status, then exit",
    )
    parser.add_argument(
        "--version", action="store_true", help="print the application version"
    )
    args = parser.parse_args(argv)

    if args.version:
        print(settings.APP_VERSION)
        return 0
    if args.check:
        return _check_environment()

    settings.ensure_directories()
    try:
        context = AppContext.create()
    except Exception as exc:
        _show_fatal_error(
            f"{exc}\n\nCheck that both ONNX models are present in the models/ "
            "folder and that the virtual environment is working."
        )
        return 1

    try:
        app = MainWindow(context)
        app.mainloop()
    except KeyboardInterrupt:
        context.shutdown()
        return 130
    except Exception as exc:
        context.shutdown()
        _show_fatal_error(f"Unexpected application error:\n{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
