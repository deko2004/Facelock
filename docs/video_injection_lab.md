# Video Injection Attack Lab

The **Video Injection Attack Lab** is an academic demonstration of a recorded
video replay attack. A local video file is decoded by the real camera worker and
passes through the same YuNet detection, SFace embedding, MiniFASNetV2 PAD and
decision pipeline used by the Live Camera page.

The lab is intentionally not a separate recognition implementation. It changes
the *frame source* and the enabled protection gates, not the biometric model,
threshold, database or matching code.

> **Scope warning.** This is a demonstration, not a certified anti-spoofing
> product. Application-level source validation cannot provide cryptographic proof
> that frames came from a real camera. An attacker who controls the operating
> system, camera driver, virtual-camera stack or process memory can bypass these
> checks. Active liveness here is a landmark-based head-movement challenge, not
> certified liveness detection.

## What is real and what is simulated

Real:

- the recorded file is decoded with OpenCV by `RecordedVideoSource`;
- the worker uses the normal capture/processing threads and latest-frame buffer;
- YuNet detects faces in the decoded frames;
- SFace extracts embeddings and compares them with the encrypted enrolled
  template;
- MiniFASNetV2 PAD runs through its normal runtime and temporal tracker;
- the source-integrity and active-liveness services run in the authentication
  path and can block identity verification;
- the result records which layer blocked the attempt.

Simulated:

- **Liveness Simulation** mode deliberately turns source integrity and PAD off
  so that a recorded file can reach the active-liveness layer. The page labels
  this mode as a simulation and never reports the source as trusted.
- The active-liveness challenge is a software challenge-response check. It does
  not prove that a physical person is present.

## Architecture

```text
RecordedVideoSource
  -> LiveCameraWorker capture thread
  -> LatestFrameBuffer
  -> LiveCameraWorker processing thread
       -> YuNet detect
       -> source integrity (optional, fail-closed)
       -> active liveness (optional, fail-closed)
       -> MiniFASNetV2 PAD (optional, fail-closed)
       -> SFace identity check
       -> LiveFrameResult
  -> Video Injection Lab page
```

The same worker is used by the Live Camera page. Without an explicit
`frame_source`, the worker wraps the existing `CameraManager` in
`PhysicalCameraSource`, so existing camera behaviour and tests are unchanged.

### Source integrity

`core/source_integrity.py` classifies the frame origin and binds it to one
authentication session.

- Recorded files are classified as `RECORDED_VIDEO` and rejected by the default
  policy.
- Linux camera metadata is read from `/sys/class/video4linux/videoN` and the
  resolved device path. Virtual-camera markers and unsupported/unknown topology
  are rejected or treated as uncertain.
- A device name such as `/dev/video0` is not trusted by itself.
- The session stores the source identity. If the identity changes during the
  session, the session is invalidated and authentication is blocked.
- Missing metadata, a missing service, or an unbound session fails closed when
  source integrity is enabled.
- Audit events record session start/end, verification and rejection. They do not
  contain frames or embeddings.

This is a metadata and policy check. It cannot prove camera authenticity against
an attacker with OS or driver control.

### Active liveness

`core/active_liveness.py` implements a randomized head-movement challenge using
YuNet's five landmarks:

- `TURN_HEAD_LEFT`
- `TURN_HEAD_RIGHT`
- `LOOK_UP`
- `LOOK_DOWN`

The service records a baseline only after the challenge is issued, then requires
several consecutive frames of movement in the requested direction. The default
is 3 frames within 10 seconds. It rejects missing faces, multiple faces, low
confidence, invalid landmarks, face changes and expired challenges. The identity
check does not run while the challenge is pending or failed.

Normalized yaw and pitch are derived from the nose position relative to the eye
midpoint and inter-eye distance. These thresholds are heuristics, not calibrated
3-D head-pose measurements.

### Passive PAD

When enabled, the existing MiniFASNetV2 path is reused without changes:

- primary face only;
- BGR crop scale `2.7`, `80x80`, raw `0..255` values;
- class index `1` is live;
- 3 consecutive live frames by default;
- spoof, error, uncertain or no-face results block SFace identity verification.

PAD remains optional and is not a liveness guarantee.

## Modes

| Mode | Source integrity | Active liveness | Passive PAD | Purpose |
|---|---|---|---|---|
| Original / Vulnerable | Off | Off | Off | Show that recognition alone can accept a replayed video. |
| Full Protection | On | On | On | Show the layered defense. A recorded file is rejected at the source gate. |
| Liveness Simulation | Off | On | Off | Exercise active liveness independently, clearly labelled as untrusted/simulation. |
| Custom | User choice | User choice | User choice | Explore combinations. |

In **Full Protection**, the recorded source is rejected before active liveness or
identity verification is reached. This is expected. Use **Liveness Simulation**
to demonstrate the active-liveness layer separately.

## Running the demonstration

1. Enrol the person shown in the video on the **Register User** page.
2. Prepare a short video of that person (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`,
   `.m4v`, `.mpg` or `.mpeg`).
3. Open **Academic -> Video Injection Lab**.
4. Choose the video and select the enrolled target user.
5. Run the same video in these modes:
   - **Original / Vulnerable**: PAD off, source integrity off, active liveness
     off. The replay may be accepted, demonstrating the vulnerability.
   - **Full Protection**: source integrity on, active liveness on, PAD on. The
     replay should be blocked by **Source Integrity**.
   - **Liveness Simulation**: source integrity off, active liveness on, PAD off.
     The replay should be blocked by **Active Liveness** unless the recording
     happens to satisfy the random challenge.
6. Read the pipeline panel and the comparison table:
   - source kind and integrity status;
   - active-liveness challenge, progress and status;
   - passive PAD status;
   - whether identity verification executed;
   - final decision;
   - which layer detected the attack;
   - processing time.

The page never fabricates a successful or failed decision. It displays the
actual worker result.

### Expected comparison

| Mode | Typical result on a recorded replay | Detected by |
|---|---|---|
| Original / Vulnerable | `ACCEPTED (attack succeeded)` if the face matches | `None` |
| Full Protection | `BLOCKED` | `Source Integrity` |
| Liveness Simulation | `BLOCKED` unless the replay satisfies the random movement | `Active Liveness` |

A recorded video of the enrolled person can legitimately pass active liveness if
it happens to perform the requested movement. That is a limitation of
challenge-response without depth or certified liveness, not a code failure.

## Result fields

`LiveFrameResult` now carries non-biometric experiment metadata:

- `source_kind`, `source_label`, `source_identity`;
- `source_integrity_status`, `source_integrity_reason`, `source_changed`;
- `liveness_status`, `liveness_challenge`, `liveness_progress`;
- `passive_pad_ran`;
- `identity_executed`;
- `final_decision`, `decision_reason`, `layer_detected`;
- `processing_ms`.

The final decision is `ACCEPTED`, `REJECTED` or `NOT RUN`. A pending liveness
challenge is not treated as accepted and does not execute identity verification.

## Configuration

The lab uses the existing PAD settings and these additional optional variables:

| Variable | Default | Purpose |
|---|---|---|
| `FACE_LIVENESS_TIMEOUT` | `10.0` | Seconds allowed for the challenge |
| `FACE_LIVENESS_REQUIRED_FRAMES` | `3` | Consecutive movement frames required |
| `FACE_LIVENESS_YAW_THRESHOLD` | `0.18` | Normalized horizontal movement threshold |
| `FACE_LIVENESS_PITCH_THRESHOLD` | `0.12` | Normalized vertical movement threshold |
| `FACE_LIVENESS_MIN_CONFIDENCE` | `0.80` | Minimum YuNet detection confidence |
| `FACE_LIVENESS_FACE_IOU_THRESHOLD` | `0.35` | Minimum face-box IoU before treating the face as changed |

These defaults are starting points, not calibrated operating points. Test them on
your own camera, lighting, display and attack samples.

## Tests

New coverage includes:

- `tests/test_frame_source.py`: recorded-video decoding, looping, restart,
  progress, invalid files and metadata mapping.
- `tests/test_source_integrity.py`: classification, fail-closed policy, session
  binding, source-change invalidation and audit events.
- `tests/test_active_liveness.py`: baseline-after-issuance, temporal movement,
  wrong-direction rejection, timeout, no-face, multiple-face, low-confidence,
  face-change and reset behaviour.
- `tests/test_video_injection_integration.py`: recorded frames through the real
  worker; unprotected acceptance; source-integrity rejection before identity;
  liveness blocking; pending liveness; liveness pass followed by identity.

The suite uses generated AVI files and fake engines. It does not require a
physical webcam.

## Limitations

- Application-level source validation is not cryptographic camera authenticity.
- A kernel driver, virtual camera, OS-level frame injection or process-memory
  attacker can bypass the checks.
- A recorded video can satisfy a head-movement challenge if it contains the
  requested movement. Randomization makes this less likely but does not eliminate
  it.
- The active-liveness thresholds are not calibrated across cameras, glasses,
  lighting, head coverings, facial hair or mobility limitations.
- MiniFASNetV2 is a demonstration PAD classifier, not a certified PAD system.
- The lab does not evaluate deepfakes, masks, 3-D presentations or adversarial
  perturbations.
- The comparison table records the latest representative result per mode, not a
  statistical trial. Use repeated trials for a real evaluation.
- Do not use this lab as the only control for a real account or physical access
  point.
