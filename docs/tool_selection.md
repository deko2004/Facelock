# Tool selection and Facelock analysis

## Requirement

The project needed a face recognition component that could:

1. Compare two still images, not only open a webcam.
2. Support 1:1 verification against multiple locally enrolled users.
3. Expose a configurable similarity threshold and an interpretable score.
4. Run fully offline on Linux with Python 3.12.
5. Integrate with an existing OpenCV + Tkinter desktop application.
6. Keep biometric templates local and encrypted.

## Candidates considered

| Option | Image-to-image | Multi-user templates | Threshold control | Offline | Fit for this project |
|---|---|---|---|---|---|
| OpenCV YuNet + SFace | Yes | Yes, app-managed | Yes | Yes | **Selected** |
| Facelock PyPI package | No documented API for this | No app-managed templates | Not exposed | Yes | Not suitable |
| `face_recognition` (dlib) | Yes | App-managed | Distance threshold | Yes | Viable, but heavier build |
| DeepFace | Yes | App-managed | Model-specific | Yes | Powerful but heavy |
| Cloud face API | Yes | Vendor-managed | Limited | No | Rejected on privacy |

## Why OpenCV YuNet + SFace was selected

- The project already had `opencv-contrib-python` and working ONNX models.
- YuNet is a small, fast face detector with a published OpenCV Zoo model.
- SFace produces a 128-dimensional embedding and OpenCV exposes alignment
  (`alignCrop`) and feature extraction (`feature`).
- The same pipeline works for uploaded images and webcam frames.
- Cosine similarity is easy to explain and to compare against a threshold.
- No extra runtime, model server or cloud dependency is required.
- The application controls the database, threshold, quality checks and audit
  events rather than delegating them to a black-box package.

The selected models are:

```text
models/face_detection_yunet_2023mar.onnx
models/face_recognition_sface_2021dec.onnx
```

They come from the OpenCV Zoo. The code resolves the OpenCV 4.x and 5.x SFace
constant names, because OpenCV 5 renamed them.

## Facelock analysis

### What the published package is

The proposal mentioned **Facelock**. At the time of investigation, the PyPI
package metadata showed:

- package name: `facelock`
- version: `0.1.2`
- release date: April 2025
- license: MIT
- requires Python: `>=3.8`

The package is a small wrapper around InsightFace. Its published usage is a
webcam-based authentication check that returns a boolean, rather than an
image-to-image comparison API with per-user templates and a configurable score.
The implementation uses `opencv-python-headless`, `onnxruntime` and InsightFace
model assets.

### Why it was not used

1. **Wrong abstraction level.** The project needs two-image comparison and 1:1
   verification against a selected enrolled user. Facelock's documented workflow
   is a webcam authentication check, not a general comparison engine.
2. **No multi-template management.** There is no application-facing API for
   enrolling several users, listing them, deleting them or choosing one for a
   verification attempt. Those are core requirements here.
3. **No exposed threshold or score.** The boolean result is not enough for an
   academic discussion of FAR/FRR, score distributions or threshold trade-offs.
4. **Dependency conflict.** Facelock depends on `opencv-python-headless`, while
   this desktop project uses `opencv-contrib-python` with Tkinter and webcam
   support. Installing both OpenCV distributions in one environment is a known
   source of conflicts.
5. **Model download/import behaviour.** The InsightFace-based path expects or
   downloads model assets, which is unnecessary when the project already has
   pinned ONNX models and needs reproducible offline behaviour.
6. **No GUI or evaluation pipeline.** The project would still need to build the
   entire interface, storage, encryption, quality checks and evaluation layer.

### What would change the decision

If the requirement were only "open a webcam and return a boolean for one known
face", Facelock could be a reasonable quick prototype. It is not a good fit for a
multi-user, image-and-webcam, threshold-driven academic system with an evaluation
component.

### Evidence

- PyPI: <https://pypi.org/project/facelock/>
- Repository: <https://github.com/Ranuja01/facelock>
- OpenCV Zoo YuNet:
  <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>
- OpenCV Zoo SFace:
  <https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface>

Package metadata and repository contents can change. Re-check the sources before
citing this analysis in a report.

## Other options

### `face_recognition` (dlib)

Pros: simple API, well-known 128-D embeddings, image-to-image comparison.

Cons: requires a dlib build toolchain, adds a large native dependency, has no
built-in GUI or encrypted template store, and its default model is not the same
as the project's OpenCV models. The existing OpenCV stack made SFace the lower
friction choice.

### DeepFace

Pros: many models, high-level verification API, built-in evaluation helpers.

Cons: heavy TensorFlow/Keras dependency, large download and memory footprint,
more moving parts, and less control over the exact preprocessing and embedding
path. Overkill for a small offline desktop demonstration.

### Cloud face APIs

Pros: managed accuracy and liveness options.

Cons: sends biometric data to a third party, requires network and credentials,
has cost and terms-of-service implications, and conflicts with the project's
local-only privacy goal. Rejected.

## Conclusion

OpenCV YuNet + SFace is the best fit for the stated requirements: offline,
image-and-webcam capable, multi-user, threshold-driven, explainable and compatible
with the existing environment. Facelock is documented here as a considered
alternative, not as a hidden dependency. No Facelock code is used in this
repository.
