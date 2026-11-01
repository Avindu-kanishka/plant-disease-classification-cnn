
import csv
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =====================================================
# 1. PATHS
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "backup_before_deduplicated_training" / "plantdoc_aware_mobilenetv2.keras"

TEST_DIR = Path(
    r"D:\projects\Plant disease data set\PlantDoc-Dataset-master\test"
)

OUTPUT_DIR = PROJECT_DIR / "reports" / "misclassified_images"
CSV_PATH = PROJECT_DIR / "reports" / "plantdoc_misclassified_images.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. CLASS NAMES AND MAPPING
# =====================================================

CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

CLASS_MAPPING = {
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
}


# =====================================================
# 3. LOAD MODEL
# =====================================================

print("\nLoading model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# =====================================================
# 4. LOAD TEST IMAGES
# =====================================================

images = []
true_labels = []
image_paths = []

print("\nLoading PlantDoc test images...")

for folder_name, class_name in CLASS_MAPPING.items():

    folder_path = TEST_DIR / folder_name

    if not folder_path.exists():
        print(f"Missing folder: {folder_path}")
        continue

    for image_path in sorted(folder_path.iterdir()):

        if image_path.suffix.lower() not in [
            ".jpg", ".jpeg", ".png", ".webp", ".jfif"
        ]:
            continue

        # Exclude known conflicting image
        if image_path.name.lower() in {"tomato_v8.jpg", "tomato-mosaic-1wdr2jx-300x225.jpg"}:
            continue

        try:
            image = tf.keras.utils.load_img(
                image_path,
                target_size=(160, 160)
            )

            image = tf.keras.utils.img_to_array(image)
            image = preprocess_input(image)

            images.append(image)
            true_labels.append(CLASS_NAMES.index(class_name))
            image_paths.append(image_path)

        except Exception as error:
            print(f"Skipping {image_path}: {error}")


if not images:
    raise ValueError("No test images found.")

X_test = np.array(images, dtype=np.float32)
y_true = np.array(true_labels)

print("Total test images:", len(X_test))


# =====================================================
# 5. GENERATE PREDICTIONS
# =====================================================

print("\nGenerating predictions...")

predictions = model.predict(
    X_test,
    batch_size=16,
    verbose=1
)

y_pred = np.argmax(predictions, axis=1)
confidence_scores = np.max(predictions, axis=1)


# =====================================================
# 6. EXPORT MISCLASSIFIED IMAGES
# =====================================================

print("\nExporting misclassified images...")

records = []
misclassified_count = 0

for i in range(len(y_true)):

    if y_true[i] == y_pred[i]:
        continue

    misclassified_count += 1

    actual_class = CLASS_NAMES[y_true[i]]
    predicted_class = CLASS_NAMES[y_pred[i]]

    confidence = float(confidence_scores[i])

    original_path = image_paths[i]

    actual_folder = actual_class.replace("Tomato___", "")
    predicted_folder = predicted_class.replace("Tomato___", "")

    output_folder = OUTPUT_DIR / actual_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    new_filename = (
        f"{misclassified_count:03d}"
        f"__PRED-{predicted_folder}"
        f"__CONF-{confidence:.3f}"
        f"__{original_path.name}"
    )

    destination = output_folder / new_filename

    shutil.copy2(original_path, destination)

    records.append({
        "image_number": misclassified_count,
        "image_name": original_path.name,
        "actual_class": actual_class,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "original_path": str(original_path),
        "copied_path": str(destination),
    })

    print(
        f"{misclassified_count}. "
        f"Actual: {actual_class} | "
        f"Predicted: {predicted_class} | "
        f"Confidence: {confidence:.4f}"
    )


# =====================================================
# 7. SAVE CSV REPORT
# =====================================================

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "image_number",
        "image_name",
        "actual_class",
        "predicted_class",
        "confidence",
        "original_path",
        "copied_path",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(records)


# =====================================================
# 8. FINAL SUMMARY
# =====================================================

accuracy = np.mean(y_true == y_pred) * 100

print("\n========================================")
print("MISCLASSIFIED IMAGE INSPECTION COMPLETED")
print("========================================")

print("Total test images:", len(y_true))
print("Correct predictions:", int(np.sum(y_true == y_pred)))
print("Misclassified images:", misclassified_count)
print(f"Test accuracy: {accuracy:.2f}%")

print("\nImages saved to:", OUTPUT_DIR)
print("CSV report saved to:", CSV_PATH)
