# Architecture

## Overview

The application is a layered desktop program:

```text
Tkinter GUI
    |
    v
Application services (AppContext)
    |
    +--> FaceEngine        YuNet detection + SFace embeddings
    +--> FaceMatcher       cosine similarity + threshold decision
    +--> AntiSpoofingService  optional MiniFASNetV2 PAD runtime
    +--> EnrollmentService
    +--> VerificationService
    +--> EvaluationService
    +--> FaceDatabase      SQLite + encrypted templates
    +--> TemplateCipher    Fernet key management
    +--> CameraManager     OpenCV VideoCapture lifecycle
```

`config/settings.py` supplies paths, model names, quality limits and threshold
persistence. `core/app_context.py` is the composition root: it constructs the
engine, database, cipher, matcher and services once and shares them with the GUI.

## Data flow

### Enrolment

```text
image/frame
  -> load_image / frame validation
  -> YuNet detect
  -> quality assessment
  -> SFace alignCrop + feature
  -> 128-D float32 embedding
  -> Fernet encrypt
  -> SQLite users table
```

The original image is not persisted. The database stores the encrypted template,
embedding dimension, model label, threshold, display name and timestamp.

### 1:1 verification

```text
probe image/frame
  -> YuNet detect
  -> SFace feature
  -> decrypt enrolled template
  -> cosine similarity
  -> similarity >= threshold ?
  -> VERIFIED / NOT VERIFIED
  -> audit event (score, threshold, source; no image or embedding)
```

The score is a geometric similarity, not a probability. A low-quality probe is
still scored, but the result message includes the quality warning.

### Live camera with optional PAD

```text
frame
  -> YuNet detect
  -> largest/primary face
  -> if PAD OFF: original SFace path
  -> if PAD ON:
       crop with scale 2.7 -> resize 80x80 BGR float32 (no /255)
       -> MiniFASNetV2 ONNX (every N frames)
       -> 3-class softmax: index 1 = live
       -> temporal live evidence (3 consecutive frames by default)
       -> LIVE ? SFace identity verification
       -> SPOOF / ERROR / UNCERTAIN / NO FACE ? block authentication
```

Only the primary face is eligible for PAD and identity verification. Other
detected faces are marked as PAD-primary-only and receive no identity label.
Face changes are detected with bounding-box IoU and reset the temporal evidence.
The runtime is disabled by default, so OFF mode adds no PAD inference and keeps
the original YuNet + SFace behaviour.

### Evaluation

```text
dataset/<identity>/<images>
  -> largest face per usable image
  -> SFace feature per image
  -> genuine pairs: same identity, different images
  -> impostor pairs: different identities
  -> confusion matrix at threshold
  -> FAR/FRR sweep 0.00..1.00
  -> pairs.csv, summary.csv, far_frr_chart.png
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `config/settings.py` | Paths, model labels, quality limits, threshold clamp/load/save |
| `core/face_engine.py` | Load YuNet/SFace, detect faces, align and extract embeddings, cosine/L2 helpers |
| `core/image_processor.py` | Safe image loading, quality checks, annotation, comparison overlay |
| `core/matcher.py` | `MatchResult`, `decide`, `FaceMatcher` |
| `core/anti_spoofing.py` | MiniFASNetV2 loading/inference, raw-BGR preprocessing, PAD tracker and runtime |
| `core/security.py` | User input validation, private permissions, Fernet encryption |
| `core/database.py` | SQLite schema, encrypted templates, users, audit log |
| `core/enrollment.py` | Consent, validation, quality gate, template creation |
| `core/verification.py` | 1:1 probe comparison against one user |
| `core/evaluation.py` | Dataset scan, pair generation, metrics, CSV and chart |
| `core/camera_manager.py` | Device discovery and `VideoCapture` lifecycle |
| `core/app_context.py` | Composition root, shared threshold updates, optional PAD service |
| `gui/main_window.py` | Sidebar, page router, status bar, shutdown |
| `gui/camera_worker.py` | Background camera/detection worker and result queue |
| `gui/*_page.py` | Page-specific UI and event handlers |
| `scripts/run_evaluation.py` | Headless evaluation runner |

## Database schema

```sql
CREATE TABLE users (
    user_id       TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL,
    template      BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL,
    model         TEXT NOT NULL,
    threshold     REAL NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    user_id     TEXT,
    detail      TEXT NOT NULL DEFAULT ''
);
```

Each operation uses a short-lived SQLite connection, which avoids sharing a
connection between the Tkinter thread and camera/evaluation workers. The database
and key files are created with `0600` permissions and their directory with `0700`
where the filesystem supports it.

## Threading

- Tkinter widgets are touched only on the main thread.
- `LiveCameraWorker` runs in a daemon thread, reads frames, runs YuNet/SFace,
  optionally runs MiniFASNetV2 on the primary face, compares eligible faces with
  the enrolled templates for name labels, and publishes the newest
  `LiveFrameResult` through a bounded queue.
- PAD inference is cached and repeated every `PAD_FRAME_INTERVAL` frames; the
  temporal tracker requires `PAD_REQUIRED_LIVE_FRAMES` consecutive live results
  before SFace is allowed to run.
- The GUI polls the queue with `after(...)`, so a slow inference pass never blocks
  the event loop.
- `FaceEngine` and `AntiSpoofingDetector` protect model calls with locks, allowing
  the live worker and a user-triggered operation to coexist safely.
- `CameraManager` uses an `RLock` for open/read/release and is idempotent on
  release.

## Error handling

- Missing core models raise `FaceEngineError` with an actionable path message.
- Missing PAD model, missing `onnxruntime` or PAD inference failure raises
  `AntiSpoofingError`; while PAD is ON this blocks authentication rather than
  silently falling back to unprotected matching.
- Invalid input raises `ValidationError`.
- Database conflicts raise `DuplicateUserError`; missing users raise
  `UserNotFoundError`.
- Encryption failures raise `SecurityError`.
- Camera failures raise `CameraError`; the worker publishes the error instead of
  dying silently.
- GUI actions catch expected domain errors and show a message rather than
  crashing the event loop.
- Audit logging is best-effort and must not break a biometric workflow.

## OpenCV 5 compatibility

OpenCV 5 renamed the SFace distance constants. `core/face_engine.py` resolves
both spellings:

- cosine: `FaceRecognizerSF_FR_COSINE` (5.x) or
  `FaceRecognizerSF_DIS_TYPE_COSINE` (4.x)
- L2: `FaceRecognizerSF_FR_NORM_L2` (5.x) or `FaceRecognizerSF_DIS_TYPE_L2`
  (4.x)

The application computes cosine similarity explicitly so that the meaning is
always "higher is more similar" and never a distance.

## Extension points

- Replace the demonstration PAD model with a certified PAD component and
  calibrate it on representative attack data.
- Add a second template per user and a fusion rule.
- Replace SQLite with an encrypted database or OS keyring.
- Add a proper validation dataset and calibrate the threshold.
- Add per-user thresholds or score normalisation.
- Add a headless REST API only if the threat model is extended accordingly.
