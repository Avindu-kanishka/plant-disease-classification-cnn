
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# =====================================================
# 1. PATH CONFIGURATION
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR / "models" / "backup_before_deduplicated_training"
    / "plantdoc_aware_mobilenetv2.keras"
)
TEST_DIR = Path(
    r"D:\projects\Plant disease data set\PlantDoc-Dataset-master\test"
)

REPORT_DIR = PROJECT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. CLASS NAMES
# =====================================================

CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
]
# =====================================================
# 3. PLANTDOC CLASS MAPPING
# =====================================================

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
# 4. CHECK PATHS
# =====================================================

print("\nChecking project paths...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"PlantDoc test directory not found: {TEST_DIR}"
    )

print("Model path:", MODEL_PATH)
print("Test directory:", TEST_DIR)


# =====================================================
# 5. LOAD MODEL
# =====================================================

print("\nLoading PlantDoc-aware model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# =====================================================
# 6. LOAD PLANTDOC TEST IMAGES
# =====================================================

images = []
true_labels = []
image_paths = []

excluded_images = []

print("\nLoading PlantDoc test images...")

for folder_name, class_name in CLASS_MAPPING.items():

    folder_path = TEST_DIR / folder_name

    if not folder_path.exists():
        print(f"Warning: Folder not found: {folder_path}")
        continue

    for image_path in sorted(folder_path.iterdir()):

        if image_path.suffix.lower() not in [
            ".jpg", ".jpeg", ".png", ".webp", ".jfif"
        ]:
            continue

        # Exclude the known conflicting duplicate image
        if image_path.name.lower() in {"tomato_v8.jpg", "tomato-mosaic-1wdr2jx-300x225.jpg", "%2320+bacterial+spot+and+speck.jpg", "page_2.jpg", "tylcv-seminar-1-638.jpg"}:
            print(f"Excluding known conflicting image: {image_path}")
            excluded_images.append(str(image_path))
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
            image_paths.append(str(image_path))

        except Exception as e:
            print(f"Skipping {image_path}: {e}")


if len(images) == 0:
    raise ValueError(
        "No test images were loaded. Check dataset folders."
    )


X_test = np.array(images, dtype=np.float32)
y_true = np.array(true_labels)

print("\nTotal test images:", len(X_test))
print("Excluded images:", len(excluded_images))


# =====================================================
# 7. MODEL PREDICTIONS
# =====================================================

print("\nGenerating predictions...")

predictions = model.predict(
    X_test,
    batch_size=16,
    verbose=1
)

y_pred = np.argmax(predictions, axis=1)


# =====================================================
# 8. EVALUATION METRICS
# =====================================================

accuracy = accuracy_score(y_true, y_pred)

print("\n========== PLANTDOC-AWARE TEST RESULTS ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

report_text = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    zero_division=0
)

print("\nClassification Report:\n")
print(report_text)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)


# =====================================================
# 9. SAVE CLASSIFICATION REPORT
# =====================================================

report_path = (
    REPORT_DIR / "plantdoc_original_secondary56_classification_report.csv"
)

pd.DataFrame(report).transpose().to_csv(report_path)

print("\nClassification report saved:", report_path)


# =====================================================
# 10. CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES)))
)

plt.figure(figsize=(12, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.title(
    "Original MobileNetV2 - Secondary 56 Image Sensitivity Analysis"
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

cm_path = (
    FIGURE_DIR / "plantdoc_original_secondary56_confusion_matrix.png"
)

plt.savefig(cm_path, dpi=300)
plt.close()

print("Confusion matrix saved:", cm_path)


# =====================================================
# 11. SAVE EVALUATION RESULTS
# =====================================================

results = {
    "model": "PlantDoc-Aware MobileNetV2",
    "dataset": "PlantDoc Test - Secondary 56-image sensitivity analysis",
    "total_test_images": int(len(X_test)),
    "excluded_images": excluded_images,
    "accuracy": float(accuracy),
    "accuracy_percentage": float(accuracy * 100),
    "macro_precision": float(
        report["macro avg"]["precision"]
    ),
    "macro_recall": float(
        report["macro avg"]["recall"]
    ),
    "macro_f1_score": float(
        report["macro avg"]["f1-score"]
    ),
    "weighted_f1_score": float(
        report["weighted avg"]["f1-score"]
    ),
}

results_path = (
    REPORT_DIR / "plantdoc_original_secondary56_evaluation_results.json"
)

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)


# =====================================================
# 12. FINAL SUMMARY
# =====================================================

print("\n========================================")
print("PLANTDOC-AWARE EVALUATION COMPLETED!")
print("========================================")

print(f"Total test images : {len(X_test)}")
print(f"Accuracy          : {accuracy * 100:.2f}%")
print(f"Macro F1-score    : {results['macro_f1_score']:.4f}")
print(f"Weighted F1-score : {results['weighted_f1_score']:.4f}")

print("\nSaved files:")
print("Classification report:", report_path)
print("Confusion matrix:", cm_path)
print("Evaluation results:", results_path)
