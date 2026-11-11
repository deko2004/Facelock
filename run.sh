#!/usr/bin/env bash
# Reliable launcher for the Biometric Authentication System.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="$ROOT_DIR/venv"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

echo "BIOMETRIC AUTHENTICATION SYSTEM"
echo "Project: $ROOT_DIR"

if [ ! -x "$PYTHON" ]; then
    echo "Virtual environment not found. Creating venv/ ..."
    python3 -m venv "$VENV_DIR"
    "$PIP" install --upgrade pip
    "$PIP" install -r requirements.txt
fi

if ! "$PYTHON" -c "import tkinter" >/dev/null 2>&1; then
    echo "ERROR: Tkinter is not available in this Python installation."
    echo "Install it with: sudo apt install python3-tk"
    exit 1
fi

if ! "$PYTHON" - <<'PY'
import importlib.util
import sys

required = {
    "cv2": "opencv-contrib-python",
    "numpy": "numpy",
    "PIL": "Pillow",
    "cryptography": "cryptography",
}
missing = [package for module, package in required.items()
           if importlib.util.find_spec(module) is None]
if missing:
    print("Missing Python packages: " + ", ".join(missing))
    sys.exit(1)
PY
then
    echo "Installing project dependencies..."
    "$PIP" install -r requirements.txt
fi

if ! "$PYTHON" - <<'PY'
import importlib.util
import sys

required = ["cv2", "numpy", "PIL", "cryptography"]
missing = [module for module in required if importlib.util.find_spec(module) is None]
if missing:
    print("Still missing: " + ", ".join(missing))
    sys.exit(1)
PY
then
    echo "ERROR: dependencies are still missing."
    echo "Run: $PIP install -r requirements.txt"
    exit 1
fi

YUNET="$ROOT_DIR/models/face_detection_yunet_2023mar.onnx"
SFACE="$ROOT_DIR/models/face_recognition_sface_2021dec.onnx"
if [ ! -f "$YUNET" ] || [ ! -f "$SFACE" ]; then
    echo "ERROR: ONNX models are missing."
    echo "Expected:"
    echo "  $YUNET"
    echo "  $SFACE"
    echo
    echo "Download from OpenCV Zoo:"
    echo "  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    echo "  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    exit 1
fi

PAD_MODEL="$ROOT_DIR/models/anti_spoofing/MiniFASNetV2.onnx"
if [ ! -f "$PAD_MODEL" ]; then
    echo "WARNING: optional anti-spoofing model is missing."
    echo "  Expected: $PAD_MODEL"
    echo "  Download: https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx"
    echo "  The app will start, but Anti-Spoofing Protection will be unavailable."
fi

if ! "$PYTHON" -c "import onnxruntime" >/dev/null 2>&1; then
    echo "WARNING: onnxruntime is not installed."
    echo "  Anti-Spoofing Protection will be unavailable."
    echo "  Install with: $PIP install -r requirements.txt"
fi

mkdir -p "$ROOT_DIR/database" "$ROOT_DIR/results" "$ROOT_DIR/results/evaluation"

if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    echo "WARNING: no graphical display detected (DISPLAY/WAYLAND_DISPLAY are empty)."
    echo "The Tkinter interface needs a desktop session. Use '--check' for a headless model check."
fi

export OPENCV_LOG_LEVEL="${OPENCV_LOG_LEVEL:-ERROR}"
exec "$PYTHON" "$ROOT_DIR/main.py" "$@"
