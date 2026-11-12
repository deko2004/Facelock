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
- **Responsive camera pipeline**: a dedicated capture thread feeds the newest
  frame to a processing thread, with configurable detection/recognition cadence
  and optional performance diagnostics.
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

## Performance tuning

The live camera runs capture and inference on separate threads and keeps only the
newest frame, so a slow inference pass cannot build an unbounded backlog. YuNet
runs every second processed frame by default, and SFace embeddings are reused only
while the same face region is still present. PAD scheduling and temporal evidence
are unchanged.

Defaults are tuned for CPU-only laptops and can be overridden before launch:

| Variable | Default | Purpose |
|---|---|---|
| `FACE_CAMERA_BUFFERSIZE` | `2` | V4L2 capture buffer; `1` reduced measured camera FPS on the development machine |
| `FACE_DETECTION_INTERVAL` | `2` | Run YuNet every N processed frames |
| `FACE_RECOGNITION_INTERVAL` | `2` | Reuse an SFace embedding for at most N frames |
| `FACE_FEATURE_CACHE_IOU` | `0.5` | Minimum IoU for reusing a cached embedding |
| `FACE_DETECTION_MAX_WIDTH` | `0` | Optional YuNet downscale width; `0` keeps native resolution |
| `FACE_OPENCV_THREADS` | `min(2, CPUs)` | OpenCV DNN thread count |
| `FACE_PAD_THREADS` | `1` | ONNX Runtime threads for PAD |
| `FACE_PREVIEW_REFRESH_MS` | `33` | GUI preview poll interval |
| `FACE_DIAGNOSTICS` | `0` | Set to `1` to show live timing/CPU/RAM diagnostics |

`FACE_RECOGNITION_INTERVAL` is clamped to at least `FACE_DETECTION_INTERVAL`, so
SFace does not align a current frame with stale landmarks. Keep the detection
interval low for authentication; raising it trades alignment freshness for CPU.

Set `FACE_DETECTION_MAX_WIDTH=320` only if detection latency matters more than
small-face accuracy. Measure before changing defaults:

```bash
venv/bin/python scripts/benchmark_performance.py --camera --seconds 5
venv/bin/python scripts/benchmark_performance.py --camera --pad --json
```

On the development machine, the refactor kept actual-camera throughput at about
15 results/s while roughly halving process CPU time versus the original
single-threaded worker. Exact numbers vary with CPU load and camera drivers.

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

Current result: **117 passed**. The suite covers model loading and detection,
quality checks, matching math, 1:N identification labels, encryption, database
behaviour, enrolment, verification, evaluation, camera lifecycle with mocked
captures, latest-frame buffering, performance diagnostics, worker scheduling and
caching, settings, PAD preprocessing/class mapping, temporal PAD decisions,
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
scripts/      Command-line evaluation and performance benchmark runners
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
