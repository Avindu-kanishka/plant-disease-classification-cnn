
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =====================================================
# 1. PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR / "models" / "plantdoc_targeted_mobilenetv2.keras"
)

TEST_DIR = Path(
    r"D:\projects\Plant disease data set\PlantDoc-Dataset-master\test"
)

REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. CLASS DEFINITIONS
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
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
}


# =====================================================
# 3. CHECK PATHS
# =====================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test dataset not found: {TEST_DIR}"
    )


# =====================================================
# 4. LOAD MODEL
# =====================================================

print("\nLoading PlantDoc Targeted MobileNetV2 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# =====================================================
# 5. LOAD TEST IMAGES
# =====================================================

images = []
true_labels = []
image_paths = []

print("\nLoading PlantDoc test images...")

for folder_name, class_name in CLASS_MAPPING.items():

    folder_path = TEST_DIR / folder_name

    if not folder_path.exists():
        print(f"Warning: Folder not found: {folder_path}")
        continue

    for image_path in folder_path.iterdir():

        if image_path.suffix.lower() not in [
            ".jpg", ".jpeg", ".png", ".webp", ".jfif"
        ]:
            continue

        try:

            image = tf.keras.utils.load_img(
                image_path,
                target_size=(160, 160)
            )

            image = tf.keras.utils.img_to_array(image)

            image = preprocess_input(image)

            images.append(image)

            true_labels.append(
                CLASS_NAMES.index(class_name)
            )

            image_paths.append(str(image_path))

        except Exception as e:

            print(f"Skipping {image_path}: {e}")


if len(images) == 0:
    raise ValueError(
        "No test images were loaded. Check dataset folders."
    )


X_test = np.array(images)
y_true = np.array(true_labels)

print("\nTotal test images:", len(X_test))


# =====================================================
# 6. MODEL PREDICTIONS
# =====================================================

print("\nGenerating predictions...")

predictions = model.predict(
    X_test,
    verbose=1
)

y_pred = np.argmax(predictions, axis=1)


# =====================================================
# 7. EVALUATION METRICS
# =====================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n========== PLANTDOC TARGETED TEST RESULTS ==========")

print(f"Accuracy: {accuracy * 100:.2f}%")

print(f"Correct predictions: {np.sum(y_true == y_pred)}")

print(f"Incorrect predictions: {np.sum(y_true != y_pred)}")


report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# =====================================================
# 8. SAVE CLASSIFICATION REPORT
# =====================================================

report_path = (
    REPORT_DIR / "plantdoc_targeted_classification_report.csv"
)

pd.DataFrame(report).transpose().to_csv(report_path)

print("\nClassification report saved:", report_path)


# =====================================================
# 9. CONFUSION MATRIX
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
    "PlantDoc Targeted MobileNetV2 - PlantDoc Test Confusion Matrix"
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

cm_path = (
    FIGURE_DIR / "plantdoc_targeted_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300
)

plt.close()

print("Confusion matrix saved:", cm_path)


# =====================================================
# 10. SAVE EVALUATION RESULTS
# =====================================================

results = {

    "model": "PlantDoc Targeted MobileNetV2",

    "experiment": "Experiment 03 - Targeted Class Improvement",

    "dataset": "PlantDoc Test",

    "total_test_images": int(len(X_test)),

    "correct_predictions": int(np.sum(y_true == y_pred)),

    "incorrect_predictions": int(np.sum(y_true != y_pred)),

    "accuracy": float(accuracy),

    "accuracy_percentage": float(accuracy * 100),

    "classification_report": report

}


results_path = (
    REPORT_DIR / "plantdoc_targeted_evaluation_results.json"
)

with open(results_path, "w") as f:

    json.dump(
        results,
        f,
        indent=4
    )


# =====================================================
# 11. FINAL SUMMARY
# =====================================================

print("\n========================================")

print("PLANTDOC TARGETED EVALUATION COMPLETED!")

print("========================================")

print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("Total test images:", len(X_test))

print("Classification report:", report_path)

print("Confusion matrix:", cm_path)

print("Evaluation results:", results_path)

print("========================================")
