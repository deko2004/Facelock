# MiniFASNetV2 anti-spoofing model

This folder contains the optional presentation-attack detection (PAD) model used
by the Live Camera page.

## Files

- `MiniFASNetV2.onnx` - MiniFASNetV2 export used by `onnxruntime`.
- `LICENSE` - Apache License 2.0 from the upstream Silent-Face project.

## Provenance

- Model file: <https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx>
- Original architecture/training code: <https://github.com/minivision-ai/Silent-Face-Anti-Spoofing>
- SHA-256: `b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907`
- Size: `1,743,581` bytes

Both upstream projects are distributed under the Apache License 2.0. Keep the
license and this attribution with the model when redistributing it.

## Runtime contract

- Input: one BGR face crop, `float32`, shape `[1, 3, 80, 80]`.
- Crop: face box expanded by `2.7`, clipped to the frame, then resized to
  `80x80`.
- Pixel range: **`0..255`**. The upstream `ToTensor` implementation deliberately
  does not divide by 255. Do not add `/255` normalization.
- Output: three logits. Class index `1` is live/real; indices `0` and `2` are
  presentation attacks.

The application applies softmax to the logits and then temporal smoothing. The
model is a demonstration component, not a guarantee against every attack.
