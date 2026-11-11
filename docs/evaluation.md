# Evaluation

## Purpose

The evaluation module checks that the full detection -> embedding -> matching
pipeline works and reports standard 1:1 comparison metrics on a labelled folder.
It does not claim that the bundled sample is a biometric benchmark.

## Dataset format

```text
dataset/
  identity_a/
    image_1.jpg
    image_2.jpg
  identity_b/
    image_1.jpg
  identity_c/
    image_1.png
```

Each immediate subfolder is one identity. Supported image extensions are `.jpg`,
`.jpeg`, `.png`, `.bmp`, `.webp`, `.tif` and `.tiff`.

The evaluator:

1. Scans each identity folder.
2. Loads each image and runs YuNet.
3. Uses the **largest** face if multiple faces are detected.
4. Extracts an SFace embedding.
5. Skips unreadable images and images with no face, recording the reason.
6. Builds genuine and impostor pairs.

## Pair definitions

- **Genuine pair**: two different images of the same identity.
- **Impostor pair**: one image from identity A and one from identity B, where
  A and B are different identities.

The evaluator does not compare an image with itself. For `n` images of one
identity, it creates `n * (n - 1) / 2` genuine pairs. For identities A and B with
`n_a` and `n_b` images, it creates `n_a * n_b` impostor pairs.

Limits are applied to keep a run bounded: 500 genuine pairs, 500 impostor pairs
and 30 images per identity by default. When a limit is hit, a warning is added to
the report.

## Metrics

For a threshold `t`, a pair is accepted when `similarity >= t`.

| Symbol | Meaning |
|---|---|
| TP | Genuine pair accepted |
| FN | Genuine pair rejected |
| FP | Impostor pair accepted |
| TN | Impostor pair rejected |

```text
FAR = FP / (FP + TN)      # impostor acceptance rate
FRR = FN / (FN + TP)      # genuine rejection rate
Accuracy = (TP + TN) / (TP + FN + FP + TN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
```

If a denominator is zero, the corresponding metric is reported as `N/A`, not as
zero. The report also includes a threshold sweep from `0.00` to `1.00` in steps of
`0.01` so the FAR/FRR trade-off can be inspected.

## Running the evaluation

GUI:

1. Open **Evaluation & Metrics**.
2. Select the dataset folder.
3. Set the threshold.
4. Press **Run evaluation**.

Command line:

```bash
cd /home/deku/FaceRecognitionProject
venv/bin/python scripts/run_evaluation.py
```

Use a custom dataset and threshold:

```bash
venv/bin/python scripts/run_evaluation.py /path/to/dataset --threshold 0.40
```

The default output folder is `results/evaluation/`:

```text
results/evaluation/pairs.csv
results/evaluation/summary.csv
results/evaluation/far_frr_chart.png
```

## Bundled smoke-test dataset

The repository includes a small public sample from the
`ageitgey/face_recognition` examples:

```text
images/eval_demo/
  biden/biden_01.jpg
  lacamoire/lacamoire_01.png
  miranda/miranda_01.png
  obama/obama_01.jpg
  obama/obama_02.jpg
```

It has 4 identities and 5 usable images. That yields:

- 1 genuine pair: the two Obama images.
- 9 impostor pairs: every cross-identity combination.

This is far too small to estimate real-world FAR/FRR. It is included only so the
pipeline and exports can be verified reproducibly. The demo images are public
examples used for testing; check their provenance and licensing before
redistributing them.

## Observed result

Command:

```bash
venv/bin/python scripts/run_evaluation.py
```

Observed at threshold `0.363`:

```text
Identities: 4  usable images: 5
Genuine pairs: 1  impostor pairs: 9
Confusion matrix: TP=1 FN=0 FP=0 TN=9
FAR: 0.00%  FRR: 0.00%
Accuracy: 100.00%
WARNING: This is a very small sample. Treat the metrics as a pipeline smoke
test, not as a biometric performance claim.
```

Pair-level scores:

| Pair | Type | Cosine similarity | Accepted at 0.363 |
|---|---|---:|---|
| Obama 01 vs Obama 02 | Genuine | 0.74384229 | Yes |
| Biden vs Lacamoire | Impostor | -0.03334977 | No |
| Biden vs Miranda | Impostor | -0.02256019 | No |
| Biden vs Obama 01 | Impostor | 0.09934830 | No |
| Biden vs Obama 02 | Impostor | 0.23496394 | No |
| Lacamoire vs Miranda | Impostor | 0.20767295 | No |
| Lacamoire vs Obama 01 | Impostor | 0.12915534 | No |
| Lacamoire vs Obama 02 | Impostor | 0.20247913 | No |
| Miranda vs Obama 01 | Impostor | 0.15897670 | No |
| Miranda vs Obama 02 | Impostor | 0.23693982 | No |

The highest impostor score in this sample is `0.2369`, below the selected
threshold. The genuine score is `0.7438`. This is a sanity check on the
pipeline, not evidence of general accuracy.

## Interpretation rules

- Report the sample size and identity count with every metric.
- Do not describe a 100% result on 10 comparisons as a real-world accuracy.
- Do not compare FAR/FRR across datasets without controlling for image quality,
  demographics, pose, illumination and the pair-generation protocol.
- The threshold is a cosine-similarity cut-off, not a probability.
- The evaluation does not include presentation attacks, so it says nothing about
  spoofing resistance.
- The evaluation does not measure demographic fairness.

## Reproducibility notes

- Model files are fixed OpenCV Zoo releases: YuNet 2023mar and SFace 2021dec.
- Images are decoded with OpenCV and resized only if larger than 2400 px.
- The largest detected face is used when multiple faces are present.
- Scores are computed with an explicit cosine formula on `float32` embeddings.
- The generated CSV includes absolute file paths, so it is environment-specific.
- The threshold sweep and pair list are deterministic for the same models and
  images.

## Future validation

A credible study would need, at minimum:

1. A large, consented, representative dataset with multiple images per identity.
2. A fixed train/validation/test split and a documented threshold calibration.
3. Evaluation across demographic groups, pose, illumination, occlusion, age and
   image quality.
4. Presentation-attack datasets and a certified PAD component.
5. Confidence intervals or bootstrap analysis, not single-point estimates.
6. A pre-registered protocol and an independent replication.

Until then, treat the bundled result as a smoke test.
