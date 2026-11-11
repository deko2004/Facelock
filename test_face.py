
import cv2
import os

MODEL = "models/face_detection_yunet_2023mar.onnx"

# Load YuNet detector
detector = cv2.FaceDetectorYN.create(
    MODEL,
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)

# Select an image
image_path = input("Enter image path: ").strip()

if not os.path.isfile(image_path):
    print("ERROR: Image not found!")
    raise SystemExit

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Cannot read image!")
    raise SystemExit

height, width = image.shape[:2]

# Update detector input size
detector.setInputSize((width, height))

# Detect faces
_, faces = detector.detect(image)

if faces is None:
    print("No face detected.")
else:
    print(f"Detected {len(faces)} face(s).")

    for face in faces:
        x, y, w, h = map(int, face[:4])

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            "Face Detected",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    os.makedirs("results", exist_ok=True)
    output = "results/face_detection_result.jpg"
    cv2.imwrite(output, image)

    print("Result saved to:", output)
