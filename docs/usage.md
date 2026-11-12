# Usage

Launch the application:

```bash
cd /home/deku/FaceRecognitionProject
./run.sh
```

The left sidebar contains the workflows grouped as Overview, Biometric Workflows,
Data & Analysis, and Academic. The bottom status bar shows the loaded models,
the current threshold and that processing is local-only.

## Dashboard

The dashboard gives a quick summary: enrolled users, threshold, model names and
recent audit events. Use it as the starting point after launch.

## Compare Two Images

This workflow compares two photographs without enrolling anyone.

1. Open **Compare Two Images**.
2. Use **Browse...** for image A and image B.
3. Each image is loaded, checked for a detectable face and previewed.
4. Adjust the decision threshold if needed.
5. Press **Compare faces**.
6. Read the cosine similarity, threshold and MATCH / NO MATCH decision.
7. Optionally press **Save annotated images** to write annotated copies under
   `results/`.

Only the largest detected face in each image is used. If an image contains no
face, the comparison stops with a clear message. The score is a cosine
similarity, not a probability that the two people are the same.

Supported input formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`,
`.tiff`.

## Register User (Enrolment)

1. Open **Register User**.
2. Enter a user ID. It must start with a letter or number and may contain
   letters, numbers, dot, dash or underscore, up to 64 characters.
3. Enter a display name (up to 80 characters).
4. Tick the consent checkbox. Enrolment is refused without consent.
5. Choose **Browse image** or **Capture from webcam**.
6. Wait for detection and the quality report.
7. Press **Create encrypted template**.

The quality check rejects faces smaller than 80 px, very blurry faces and faces
that are too dark or too bright. If several faces are present, the first
detected face is used; use an image with one clear face for predictable results.

The original photograph is not stored. Only the encrypted SFace template and
basic metadata (user ID, display name, dimension, model, threshold, timestamp)
are written to SQLite. Duplicate user IDs are rejected.

## Verify Identity (1:1)

1. Open **Verify Identity**.
2. Select an enrolled user and press **Refresh** if the list is stale.
3. Either:
   - **Browse image** and press **Verify selected image**, or
   - press **Start verification** for the webcam and **Stop** when finished.
4. Read the similarity, threshold and decision.

A successful comparison means the probe's SFace embedding has cosine similarity
greater than or equal to the configured threshold. It does **not** prove that the
person is physically present or that the image is live.

If the probe contains no face, the result is NOT VERIFIED. Low-quality probes
still produce a score, but the message includes the quality warning. The event is
recorded in the audit log without the image or embedding.

## Live Camera

The **Live Camera** page provides a live preview with face boxes, FPS, name
labels for enrolled faces, an optional 1:1 verification against a selected user,
and an optional anti-spoofing check.

1. Choose a camera. The list comes from `/dev/video*`; the device is opened only
   when you press **Start camera**.
2. Press **Start camera**. Every detected face is compared with all enrolled
   templates; if the best cosine similarity passes the threshold, the display
   name is drawn above the box. Otherwise the box says `Unknown`.
3. Optionally choose an enrolled user to run an additional 1:1 verification on
   the largest face. The side panel shows that score and decision.
4. Optionally tick **Anti-Spoofing Protection**. The toggle takes effect without
   restarting the camera. The first time it is enabled, MiniFASNetV2 loads in the
   background; Start is disabled until loading finishes.
5. Press **Stop** before leaving the page or closing the application.

### Anti-spoofing behaviour

When the toggle is ON:

- Only the largest/primary face is evaluated and eligible for SFace identity
  verification. Other faces are marked `PAD: primary only` and are not labelled
  with identities.
- A face must be classified as live for several consecutive PAD frames before
  SFace runs. The default is 3 live frames at a live threshold of `0.80`; PAD
  inference runs every second frame.
- A spoof result, low-confidence result, no face, model error or face change
  blocks authentication. SFace is not run for a spoofed face.
- A face change is detected with bounding-box IoU and resets the temporal
  evidence.
- The side panel shows `Anti-spoofing`, live confidence and the temporal evidence
  in the hint line. The main status badge shows states such as `PAD CHECK`,
  `AUTH BLOCKED`, `HOLD STILL`, `LIVE FACE`, `VERIFIED` or `NOT VERIFIED` as
  appropriate.
- If the model or `onnxruntime` is unavailable, the toggle reports an explicit
  error and Start stays disabled while protection is ON. The application never
  silently falls back to unprotected authentication.

When the toggle is OFF, the page uses the original YuNet + SFace behaviour with
no PAD inference. A static photograph or screen replay can still produce a match.

PAD is a demonstration, not a guarantee. The default thresholds are not calibrated
for every camera, printer, display or lighting condition. Use the Security page
and manual printed-photo/screen-replay checks to understand the residual risk.

Camera work runs in a dedicated capture thread plus a processing thread so the
interface stays responsive. The capture thread keeps only the newest frame; the
processing thread runs YuNet every second frame and reuses an SFace embedding for
at most two frames while the same face box is present. The worker is stopped and
the camera released when the page is hidden or the application closes.

Set `FACE_DIAGNOSTICS=1` before launch to show capture/process FPS, YuNet/SFace/PAD
timings, frame latency, CPU, RAM and dropped frames. Other defaults can be tuned
with the `FACE_*` variables listed in the README performance section. The
recognition interval is clamped to at least the detection interval so SFace does
not reuse stale landmarks. If preview latency matters more than small-face
accuracy, `FACE_DETECTION_MAX_WIDTH=320` is an optional trade-off; leave it at `0`
for native-resolution detection.

The name label is a 1:N identification demonstration, not proof of identity. It
uses the same uncalibrated threshold as verification, and false matches become
more likely as the number of enrolled users grows. Restart the camera after
adding a new user so the new template is loaded.

## Video Injection Attack Lab

The **Academic -> Video Injection Lab** page replays a local recorded video
through the real YuNet/SFace/PAD authentication pipeline. It is not a separate
recognition implementation.

1. Enrol the person shown in the video.
2. Choose a supported video file and select the enrolled target user.
3. Choose a mode:
   - **Original / Vulnerable**: source integrity, active liveness and PAD are
     off. A matching replay may be accepted.
   - **Full Protection**: all three protections are on. A recorded source is
     rejected before identity verification.
   - **Liveness Simulation**: source integrity and PAD are off; active liveness
     is on. This is a labelled simulation because a recorded source is not
     trusted.
4. Press **Run attack**. Use Pause, Resume, Replay or Stop as needed.
5. Read the pipeline panel: source kind, source-integrity status, liveness
   challenge/progress, PAD status, whether identity verification ran, final
   decision and the layer that detected the attack.
6. Run the same video in the other modes and compare the results table.

Full Protection normally blocks at **Source Integrity**; active liveness is not
reached in that mode. Use Liveness Simulation to exercise the liveness layer. A
recording can satisfy a random head-movement challenge if it happens to perform
the requested movement, so a pass is not proof of a live person.

The lab is a demonstration. Application-level source validation is not
cryptographic camera authentication, and OS-level frame injection, malicious
drivers or virtual cameras can bypass it. See
[video_injection_lab.md](video_injection_lab.md) for the full design and
limitations.

## User Database

The **User Database** page lists enrolled users and lets you inspect metadata.
Select a row to see the user ID, display name, embedding dimension, model,
threshold, creation time and encrypted template size.

**Delete selected user** permanently removes the encrypted template and writes a
deletion audit event. There is no recycle bin; re-enrolment is required to restore
the user.

The database never contains enrolment photographs. Display names are stored in
plaintext; only the biometric template is encrypted.

## Evaluation & Metrics

The dataset must be arranged as:

```text
dataset/
  identity_a/
    image_1.jpg
    image_2.jpg
  identity_b/
    image_1.jpg
  ...
```

1. Open **Evaluation & Metrics**.
2. Choose the dataset folder (the bundled demo is `images/eval_demo`).
3. Set the threshold.
4. Press **Run evaluation**.

The service uses the largest face in each usable image. It builds:

- **Genuine pairs**: two different images of the same identity.
- **Impostor pairs**: images from different identities.

It reports the confusion matrix (TP, FN, FP, TN), FAR, FRR, accuracy, precision
and recall, plus a FAR/FRR sweep from 0.00 to 1.00. CSV files and a PNG chart are
written to `results/evaluation/` (the GUI uses `evaluation_pairs.csv`,
`evaluation_summary.csv` and `evaluation_chart.png`; the CLI uses `pairs.csv`,
`summary.csv` and `far_frr_chart.png`).

The bundled demo is deliberately tiny: 4 identities, 5 usable images, 1 genuine
pair and 9 impostor pairs. Treat the result as a pipeline smoke test, not a
performance claim. See [evaluation.md](evaluation.md).

## Security & Privacy

This page explains the threat model, encryption, data minimisation and the
difference between recognition and liveness. It also contains a controlled
static-photo spoofing demonstration: the verification service used by the demo
has no PAD check, so a still photograph can be accepted. This is an educational
warning, not a claim that the system is secure against spoofing. Use the Live
Camera toggle to exercise the optional MiniFASNetV2 PAD path.

Use the threshold control here to change the shared decision threshold. The value
is persisted locally in `database/app_settings.json`.

## About the Project

The About page summarises the objectives, technology stack, model details,
Facelock analysis and academic disclaimer. It also lists the main references.

## Practical guidance

- Use good, front-facing lighting for both enrolment and verification.
- Keep the face reasonably large in the frame.
- Use one clear face per enrolment image.
- Do not reuse a user ID for a different person.
- Treat the threshold as a demonstration setting and validate it on your own
  data before drawing conclusions.
- Treat PAD as a demonstration, not a certified liveness guarantee. Validate the
  thresholds on your own camera and attack samples.
- Do not use this system as the only control for a real account or physical
  access point.
