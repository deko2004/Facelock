"""Application entry point.

Run with:
    ./run.sh
or:
    venv/bin/python main.py
"""

from __future__ import annotations

import argparse
import importlib.util
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
    print("Core models loaded successfully.")
    print(f"Anti-spoofing model: {settings.PAD_MODEL}")
    if not settings.PAD_MODEL.is_file():
        print("Anti-spoofing: optional model not found; OFF mode still works.")
    elif importlib.util.find_spec("onnxruntime") is None:
        print(
            "Anti-spoofing: onnxruntime is not installed; OFF mode still works. "
            "Install with: pip install -r requirements.txt"
        )
    else:
        try:
            context.anti_spoofing.enable()
            detector = context.anti_spoofing.ensure_loaded()
            width, height = detector.input_size
            print(
                f"Anti-spoofing: {settings.PAD_MODEL_LABEL} loaded "
                f"({width}x{height}, 3 classes)."
            )
        except Exception as exc:
            print(f"Anti-spoofing: model found but could not load: {exc}")
        finally:
            context.anti_spoofing.disable()
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
            f"{exc}\n\nCheck that the YuNet and SFace ONNX models are present in "
            "models/ and that the virtual environment is working. The optional "
            "anti-spoofing model is only needed when PAD is enabled."
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
