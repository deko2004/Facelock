# Setup

## Tested environment

- Pop!_OS 22.04 (Linux)
- Python 3.12.3
- OpenCV 5.0.0.93 (`opencv-contrib-python`)
- NumPy 2.5.4
- Pillow 13.0.0
- Tkinter 8.6
- cryptography 50.0.2
- onnxruntime 1.31.1
- pytest 9.2.0
- Webcam `/dev/video0`
- X11 display `:1` (the GUI also works under a normal Wayland session)

OpenCV prints this warning when loading the models with its new graph engine:

```text
Targets are not supported by the new graph engine for now
```

It is non-fatal and does not prevent detection or matching.

## System packages

On Debian/Ubuntu/Pop!_OS:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-tk \
  libgl1 libglib2.0-0 v4l-utils
```

`python3-tk` is required for the interface. The OpenGL/GLib libraries are needed
by the OpenCV wheel.

## Python environment

From the project directory:

```bash
cd /home/deku/FaceRecognitionProject
python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
```

The project already includes a working `venv/` in the current development
checkout. `./run.sh` creates one automatically if it is missing.

## Models

The required YuNet and SFace models live in `models/`:

```text
models/face_detection_yunet_2023mar.onnx
models/face_recognition_sface_2021dec.onnx
```

Download from OpenCV Zoo if they are missing:

```bash
mkdir -p models
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -L -o models/face_recognition_sface_2021dec.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

The application checks that both required files exist and fails with a clear
message if they do not.

### Optional anti-spoofing model

The Live Camera page can use MiniFASNetV2 for presentation-attack detection:

```text
models/anti_spoofing/MiniFASNetV2.onnx
```

If the file is missing, the application still starts and works with
Anti-Spoofing Protection OFF. To restore it:

```bash
mkdir -p models/anti_spoofing
curl -L -o models/anti_spoofing/MiniFASNetV2.onnx \
  https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx
sha256sum models/anti_spoofing/MiniFASNetV2.onnx
```

Expected SHA-256:
`b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907`.

The model requires `onnxruntime`, which is listed in `requirements.txt`. The
exact preprocessing contract and Apache-2.0 attribution are in
`models/anti_spoofing/README.md`.

## Camera permissions

Check that your user can read and write the device:

```bash
ls -l /dev/video*
id
```

If needed, add your user to the `video` group and log out/in:

```bash
sudo usermod -aG video "$USER"
```

Test the device:

```bash
v4l2-ctl --list-devices
```

In this environment `/dev/video0` works. `/dev/video1` exists but is not a usable
capture device. The GUI lists the devices it finds and attempts to open the
selected one when the user presses Start.

## Verify the installation

```bash
./run.sh --check
```

Expected output includes:

```text
BIOMETRIC AUTHENTICATION SYSTEM v1.0.0
Detector: YuNet 2023mar
Recognizer: SFace 2021dec (128-D)
Enrolled users: 0
Threshold: 0.363
Core models loaded successfully.
Anti-spoofing model: /home/deku/FaceRecognitionProject/models/anti_spoofing/MiniFASNetV2.onnx
Anti-spoofing: MiniFASNetV2 (ONNX) loaded (80x80, 3 classes).
```

If the PAD model or `onnxruntime` is missing, `--check` reports that the optional
component is unavailable instead of failing the core check.

The enrolled-user count is `0` only on a fresh database. An existing local
database may contain your own test enrolments.

## Launch

```bash
./run.sh
```

If you are in an SSH session without a display, the GUI cannot open. Use a local
desktop session or forward X11:

```bash
ssh -X user@host
cd /home/deku/FaceRecognitionProject
./run.sh
```

For a headless model/database check only:

```bash
./run.sh --check
```

## Run the tests

```bash
venv/bin/python -m pytest
```

No physical webcam is required. The GUI smoke test is skipped automatically when
neither `DISPLAY` nor `WAYLAND_DISPLAY` is set.

## Common problems

### `ModuleNotFoundError: No module named 'cv2'`

Use the project interpreter, not the system Python:

```bash
venv/bin/python main.py
```

Or let the launcher repair missing dependencies:

```bash
./run.sh --check
```

### `ModuleNotFoundError: No module named 'onnxruntime'`

The core application can run without `onnxruntime`, but Anti-Spoofing Protection
cannot. Install the project requirements:

```bash
venv/bin/pip install -r requirements.txt
```

If installation is not possible, leave the toggle OFF; the core YuNet/SFace
workflows still work.

### Anti-Spoofing Protection shows a model error

Confirm the optional model exists and matches the expected checksum:

```bash
ls -l models/anti_spoofing/MiniFASNetV2.onnx
sha256sum models/anti_spoofing/MiniFASNetV2.onnx
```

Re-download it from the URL in the Models section if the file is missing or
corrupt. The application intentionally blocks authentication while the toggle is
ON and the model is unavailable; turn the toggle OFF to return to the original
behaviour.

### `Could not open camera 0`

- Confirm the device exists with `ls -l /dev/video*`.
- Close other applications that may be using the camera.
- Check the `video` group membership.
- Try selecting a different camera in the GUI.
- In a virtual machine, pass the USB webcam through to the guest.

### The GUI does not appear

- Confirm `echo "$DISPLAY"` is non-empty.
- Confirm `python3 -c "import tkinter; tkinter.Tk()"` opens a window.
- If using SSH, enable X11 forwarding with `ssh -X` or use a local session.

### The threshold seems wrong

The default `0.363` is OpenCV's SFace cosine reference for LFW, not a calibrated
value for this application. Change it on the Security page or with
`--threshold` in the evaluation script, and validate it on your own data.

### Model key or database errors

The encryption key is `database/templates.key` and the database is
`database/biometric.db`. If the key is lost, existing templates cannot be
decrypted. Do not delete or replace the key unless you intend to re-enrol every
user.
