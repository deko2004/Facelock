# Testing

## Run the suite

```bash
cd /home/deku/FaceRecognitionProject
venv/bin/python -m pytest
```

Observed result:

```text
143 passed (about 9s on the development machine)
```

The suite does not require a physical webcam. It uses temporary databases, keys
and images, so it does not modify the real `database/` or `results/` folders.

## What is covered

| Area | Tests |
|---|---|
| Models | Model loading, detection, no-face handling, multiple faces, embedding shape, self-similarity, different-person scores, missing-model errors |
| Matching | Threshold boundary, known cosine values, shape mismatch, zero vectors, result text, threshold clamping, 1:N best-match identification |
| Image processing | Valid/missing/unsupported/corrupt images, quality pass/reject, annotation does not mutate the input, custom face labels, save |
| Security | User ID and display-name validation, Fernet round-trip, wrong-key failure, key permissions, safe unlink |
| Database | Add/get/list/count/delete, duplicate rejection, missing user, invalid ID, non-finite embedding, encryption at rest, file permissions, corrupted token, audit log |
| Enrolment | Consent, duplicate, invalid ID, no face, missing file, frame enrolment, success path |
| Verification | Same person, different person, unknown user, no face, deletion, shared threshold update |
| Evaluation | Confusion matrix, FAR/FRR/accuracy/precision/recall, missing/empty dataset, demo dataset, no-genuine-pair case, CSV and chart export |
| PAD | ONNX input/output contract, raw `0..255` preprocessing, official genuine/fake samples, synthetic screen replay, temporal evidence/reset, service load/error paths, worker blocking and SFace-after-live |
| Frame source | Recorded-video decoding, looping, restart, progress, invalid/missing files, metadata mapping |
| Source integrity | Recorded/physical/virtual/unknown classification, fail-closed policy, session binding, source-change invalidation, audit events |
| Active liveness | Baseline after issuance, temporal movement, wrong-direction rejection, timeout, no-face, multiple-face, low confidence, face change and reset |
| Video injection | Recorded frames through the real worker; unprotected acceptance; source-integrity rejection before identity; liveness blocking/pending/pass paths |
| Camera | Device enumeration shape, read-before-open error, open failure, mocked open/read/release, buffer-size property, deferred release during a blocking read, probe success/failure, worker name labels |
| Performance | Latest-frame replacement/drop counting, diagnostics summary/CPU-RSS sampling, detection/recognition cadence, feature-cache reuse/invalidation, lifecycle stop/join |
| Settings | Threshold clamp, save/load, invalid/missing settings fallback |
| GUI | Every page constructs, shows and hides under a real Tk display; background results are delivered on the main thread |

## GUI test

`tests/test_gui_smoke.py` creates the main window with a temporary context,
visits every page in `PAGE_CLASSES`, calls `on_show()` and closes the application.
It is skipped automatically when neither `DISPLAY` nor `WAYLAND_DISPLAY` is set.

Run it explicitly:

```bash
DISPLAY=:1 venv/bin/python -m pytest tests/test_gui_smoke.py -v
```

## Evaluation test

`tests/test_evaluation.py` runs the real YuNet/SFace pipeline on
`images/eval_demo/` and asserts the observed smoke-test shape:

- 5 usable images
- 4 identities
- 1 genuine pair
- 9 impostor pairs
- FAR and FRR available and zero at threshold `0.363`
- 101-point threshold sweep
- CSV and PNG chart exports

This is a regression test for the pipeline, not a performance benchmark.

## Performance test

`tests/test_frame_buffer.py` and `tests/test_diagnostics.py` cover the latest-frame
buffer and diagnostics primitives. `tests/test_camera_worker.py` covers detection
and recognition cadence, feature-cache reuse/invalidation, PAD scheduling and
worker shutdown. `tests/test_camera_manager.py` covers capture buffer settings.

Use the benchmark runner for local measurements:

```bash
venv/bin/python scripts/benchmark_performance.py
venv/bin/python scripts/benchmark_performance.py --camera --seconds 5
venv/bin/python scripts/benchmark_performance.py --camera --pad --json
```

The benchmark is not part of the test suite because webcam availability, CPU load
and drivers vary. On the development machine the refactor kept actual-camera
throughput near 15 results/s while roughly halving process CPU time; preview
conversion also dropped from roughly 11 ms to a few milliseconds on the fast
path. Treat those numbers as environment-specific, not guarantees.

## PAD test

`tests/test_anti_spoofing.py` loads the real `MiniFASNetV2.onnx` model. It checks:

- the ONNX input shape `[N, 3, 80, 80]`, output shape `[N, 3]` and class mapping;
- preprocessing keeps BGR values in `0..255` and does not divide by 255;
- official Silent-Face genuine and fake samples are classified as expected;
- a synthetic screen-replay image is blocked by the temporal tracker;
- consecutive live results are required before SFace runs;
- spoof/error/no-face paths block SFace and cannot return `verified=True`;
- missing model and inference-error paths raise explicit errors.

The sample assets are in `tests/assets/anti_spoofing/` and are attributed to the
upstream Apache-2.0 project. This is an effectiveness smoke test on a few images,
not a PAD benchmark or a calibration dataset.

## Video injection tests

`tests/test_video_injection_integration.py` generates a temporary AVI and feeds it
through `LiveCameraWorker` with fake engines and real frame-source, source
integrity and active-liveness services. It verifies:

- a recorded replay can be accepted when protections are off;
- source integrity rejects the recorded source before identity verification;
- multiple faces fail active liveness before identity;
- a pending liveness challenge keeps identity blocked;
- a satisfied challenge allows the identity path to run.

No physical webcam is required. The tests do not prove real-world PAD or liveness
effectiveness; they prove that the gate order and fail-closed decisions are wired
into the actual worker.

## Manual checks

The automated suite cannot replace a manual GUI/camera check:

1. Run `./run.sh --check`.
2. Launch `./run.sh` on a desktop session.
3. Open each page.
4. Enrol a test user from an image.
5. Verify the same image and a different person's image.
6. Start the live camera, confirm preview/FPS, then stop. Optionally relaunch
   with `FACE_DIAGNOSTICS=1` and confirm the diagnostics row updates.
7. Enable **Anti-Spoofing Protection**, confirm the model loads, present a live
   face and confirm SFace runs only after live evidence accumulates.
8. Present a printed photo or a face on a phone/tablet screen and confirm the
   result is `AUTH BLOCKED` or `SPOOF DETECTED` rather than verified.
9. Toggle PAD OFF and confirm the original behaviour returns without restarting
   the camera.
10. Open **Video Injection Lab**, choose a recorded video of the enrolled test
    user and run Original / Vulnerable, Full Protection and Liveness Simulation.
    Confirm that Full Protection reports `Source Integrity` and does not run
    identity, while Liveness Simulation exercises the active-liveness layer.
11. Delete the test user and confirm the count decreases.
12. Run the evaluation page on `images/eval_demo`.

## Known test limitations

- The camera tests use a fake `cv2.VideoCapture`; they do not validate a real
  device driver. Recorded-video tests use generated AVI files and fake engines.
- Source-integrity tests exercise metadata policy, not kernel/driver or
  virtual-camera bypasses. Active-liveness tests use synthetic landmarks, not a
  calibrated head-pose dataset.
- The GUI test constructs pages but does not automate every button path.
- PAD tests use a few official samples and a synthetic screen replay; they are
  not a representative PAD benchmark, and thresholds are not calibrated.
- There is no demographic fairness test; the sample is too small.
- The evaluation result is a pipeline smoke test, not a biometric benchmark.
- Security tests cover the implemented controls, not a full penetration test.

## Interpreting failures

- Missing core models: place YuNet and SFace ONNX files in `models/`.
- PAD tests failing because `MiniFASNetV2.onnx` is missing: restore the optional
  model and checksum described in [setup.md](setup.md).
- PAD tests failing because `onnxruntime` is missing: `venv/bin/pip install -r
  requirements.txt`.
- Tkinter display error: run in a desktop session or forward X11.
- `cryptography` missing: `venv/bin/pip install -r requirements.txt`.
- Evaluation test failure after changing thresholds: the expected FAR/FRR values
  are tied to the bundled sample and threshold `0.363`; update the test only if
  the experimental setup intentionally changes.
