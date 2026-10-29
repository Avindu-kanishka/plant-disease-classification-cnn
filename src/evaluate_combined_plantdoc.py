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


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "plant_disease_mobilenetv2_7class.keras"

TEST_DIR = Path(
    r"D:\projects\Plant disease data set\PlantDoc-Dataset-master\test"
)

REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# Seven model classes
CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]


# PlantDoc folder mapping
CLASS_MAPPING = {
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
}


# Load model
print("Loading combined 7-class model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# Load PlantDoc test images
images = []
true_labels = []
image_paths = []

for folder_name, class_name in CLASS_MAPPING.items():

    folder_path = TEST_DIR / folder_name

    if not folder_path.exists():
        print(f"Warning: Folder not found: {folder_path}")
        continue

    for image_path in folder_path.iterdir():

        if image_path.suffix.lower() not in [
            ".jpg", ".jpeg", ".png", ".webp"
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
            true_labels.append(CLASS_NAMES.index(class_name))
            image_paths.append(str(image_path))

        except Exception as e:
            print(f"Skipping {image_path}: {e}")


X_test = np.array(images)
y_true = np.array(true_labels)

print(f"\nTotal test images: {len(X_test)}")


# Predict
print("\nPredicting...")

predictions = model.predict(X_test, verbose=1)

y_pred = np.argmax(predictions, axis=1)


# Evaluation
accuracy = accuracy_score(y_true, y_pred)

print("\n========== PLANTDOC TEST RESULTS ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

print("\nClassification Report:")

print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    zero_division=0
))


# Save classification report
pd.DataFrame(report).transpose().to_csv(
    REPORT_DIR / "combined_plantdoc_classification_report.csv"
)


# Confusion matrix
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

plt.title("Combined 7-Class Model - PlantDoc Test Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "combined_plantdoc_confusion_matrix.png",
    dpi=300
)

plt.close()


# Save evaluation results
results = {
    "model": "Combined MobileNetV2 7-Class",
    "dataset": "PlantDoc Test",
    "total_test_images": int(len(X_test)),
    "accuracy": float(accuracy),
    "accuracy_percentage": float(accuracy * 100)
}

with open(
    REPORT_DIR / "combined_plantdoc_evaluation_results.json",
    "w"
) as f:
    json.dump(results, f, indent=4)


print("\nEvaluation completed!")
print("Reports saved in the reports folder.")