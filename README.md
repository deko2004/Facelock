# Biometric Authentication System

A local-only desktop demonstration of face recognition and 1:1 identity
verification, built for a Master's-level Cybersecurity project. It uses OpenCV
YuNet for face detection and SFace for 128-dimensional face embeddings, with a
Tkinter interface, encrypted template storage and an honest evaluation pipeline.

**Arabic overview:** [README.ar.md](README.ar.md)

> **Academic demonstration only.** This is not a production identity system. The
> optional MiniFASNetV2 PAD check is a demonstration, not a guarantee against all
> presentation attacks. It has no certified bias evaluation, no hardware-backed
> key storage and no operational threshold calibration. Do not use it to protect
> real accounts, buildings or sensitive data.

## Features

- **Compare images**: detect faces in two photographs and compute a cosine
  similarity score with a threshold decision.
- **Live camera identification and verification**: label detected faces with the
  closest enrolled name, and optionally run 1:1 verification against one user.
- **Optional anti-spoofing protection**: toggle MiniFASNetV2 PAD on the Live
  Camera page. When enabled, the largest face must pass a temporal liveness check
  before SFace identity verification runs.
- **Enrolment**: create a local user from an uploaded image or webcam frame.
  Consent is required and the original photograph is not stored.
- **Database**: list, search and delete enrolled users; inspect non-biometric
  audit events.
- **Evaluation**: compute genuine/impostor pairs, FAR/FRR, accuracy and a
  threshold sweep from `dataset/<identity>/<images>`, then export CSV and a PNG
  chart.
- **Security page**: privacy/threat-model explanation plus a controlled
  static-photo spoofing demonstration.
- **About page**: objectives, technology, model notes, Facelock analysis and
  academic disclaimer.
- **Local-only design**: no network calls, no cloud API and no telemetry.

## Quick start

Tested on Pop!_OS 22.04 with Python 3.12, OpenCV 5.0.0, onnxruntime 1.31.1,
Tkinter 8.6 and a `/dev/video0` webcam.

```bash
cd /home/deku/FaceRecognitionProject
./run.sh --check     # verify models, database and threshold
./run.sh             # launch the GUI
```

`run.sh` creates `venv/` if needed, installs `requirements.txt` when packages are
missing, checks the ONNX models and starts the application. It does not modify
the models or your enrolled data.

### Models

The application expects these files:

```text
models/face_detection_yunet_2023mar.onnx
models/face_recognition_sface_2021dec.onnx
```

They are from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo):

```bash
mkdir -p models
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -L -o models/face_recognition_sface_2021dec.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

The optional anti-spoofing model is included at:

```text
models/anti_spoofing/MiniFASNetV2.onnx
```

If it is missing, the application still starts and works with Anti-Spoofing
Protection OFF. To restore it:

```bash
mkdir -p models/anti_spoofing
curl -L -o models/anti_spoofing/MiniFASNetV2.onnx \
  https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx
```

SHA-256:
`b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907`.
See `models/anti_spoofing/README.md` for provenance, the exact preprocessing
contract and Apache-2.0 attribution.

## Main workflows

1. Open **Enrolment**, enter a user ID and display name, tick consent, then load
   an image or capture a frame.
2. Open **Verification**, select the enrolled user and compare an image or a live
   camera frame.
3. On **Live Camera**, tick **Anti-Spoofing Protection** to require a live face
   before SFace identity verification. The toggle takes effect without restarting
   the camera. If the model or `onnxruntime` is unavailable, the toggle shows an
   explicit error and authentication is blocked while it remains on.
4. Open **Compare Images** for a direct two-photo comparison that does not require
   enrolment.
5. Use **Database** to review or delete enrolled users. Deletion removes the
   encrypted template and writes an audit event.
6. Use **Evaluation** with a folder arranged as
   `dataset/<identity>/<image files>`.

The default threshold is `0.363`, OpenCV's published SFace cosine reference for
its LFW calibration. It is a starting point, **not** a calibrated threshold for
this application. The GUI can change it and persists the value locally.

## Evaluation smoke test

The repository includes a small public sample from the
`ageitgey/face_recognition` examples under `images/eval_demo/`. It contains four
identities and five usable images, so it is only a pipeline smoke test.

```bash
venv/bin/python scripts/run_evaluation.py
```

Observed result at threshold `0.363`:

```text
Identities: 4  usable images: 5
Genuine pairs: 1  impostor pairs: 9
Confusion matrix: TP=1 FN=0 FP=0 TN=9
FAR: 0.00%  FRR: 0.00%
Accuracy: 100.00%
WARNING: This is a very small sample. Treat the metrics as a pipeline smoke
test, not as a biometric performance claim.
```

Artifacts are written to `results/evaluation/`:

- `pairs.csv`: one row per comparison
- `summary.csv`: metrics and warnings
- `far_frr_chart.png`: FAR/FRR against threshold

The tiny sample cannot establish real-world accuracy. See
[docs/evaluation.md](docs/evaluation.md) for the methodology and limitations.

## Security and privacy

- Embeddings are encrypted at rest with Fernet (AES-128-CBC + HMAC-SHA256).
- The key is stored separately in `database/templates.key` with `0600`
  permissions; the database uses `0600` and its directory `0700`.
- Enrolment photographs are never written to the database. Only the encrypted
  template and basic metadata are stored.
- User IDs are validated to a safe ASCII subset.
- Audit events record workflow outcomes and scores, not raw embeddings or images.
- No network communication is performed.
- Encryption at rest does **not** protect against malware, a compromised user
  account, an attacker who can read the whole project directory, or a malicious
  process running as the same user.

The full threat model is in
[docs/security_privacy.md](docs/security_privacy.md).

## Testing

```bash
venv/bin/python -m pytest
```

Current result: **105 passed**. The suite covers model loading and detection,
quality checks, matching math, 1:N identification labels, encryption, database
behaviour, enrolment, verification, evaluation, camera lifecycle with mocked
captures, settings, PAD preprocessing/class mapping, temporal PAD decisions,
PAD-blocked camera results and a GUI page-construction smoke test. The PAD
effectiveness tests use the official Silent-Face genuine/attack samples and a
synthetic screen-replay sample. The GUI smoke test is skipped automatically when
no graphical display is available; no physical webcam is required for the test
suite.

## Project layout

```text
config/       Settings, paths and threshold persistence
core/         Detection, embeddings, matching, storage, enrolment, verification,
              evaluation, camera lifecycle and the composition root
gui/          Tkinter pages, styling, widgets and the camera worker thread
models/       YuNet, SFace and optional MiniFASNetV2 ONNX files
images/       Sample images and the small public evaluation demo
database/     Local SQLite database and encryption key (git-ignored)
results/      Generated evaluation artifacts
scripts/      Command-line evaluation runner
tests/        pytest suite
docs/         Setup, usage, architecture, security, evaluation and tool selection
```

## Documentation

- [Arabic overview](README.ar.md)
- [Arabic quick guide](docs/ar/README.md)
- [Setup](docs/setup.md)
- [Usage](docs/usage.md)
- [Architecture](docs/architecture.md)
- [Security and privacy](docs/security_privacy.md)
- [Evaluation](docs/evaluation.md)
- [Tool selection and Facelock analysis](docs/tool_selection.md)
- [Testing](docs/testing.md)

## Limitations

- Anti-spoofing is optional and is not a guarantee against all presentation
  attacks. The default PAD thresholds are not calibrated for every camera,
  printer or display; determined attackers may still succeed.
- No demographic bias evaluation; the sample set is far too small and not
  representative.
- The threshold is not calibrated on a validation set.
- Templates are protected at rest but decrypted in process memory during
  matching.
- The database stores display names in plaintext; only the biometric templates
  are encrypted.
- Local file permissions and the operating-system account are part of the
  security boundary.
- The camera path assumes a Linux/V4L2 webcam. Other platforms may need code
  changes.
