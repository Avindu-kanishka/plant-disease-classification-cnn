from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "plant_disease_mobilenetv2_finetuned.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "reports" / "class_names.json"

PLANTDOC_TEST = Path(
    r"D:\projects\Plant disease data set\PlantDoc-Dataset-master\test"
)

IMAGE_SIZE = (160, 160)

CLASS_MAPPING = {
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
}


def main():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        model_classes = json.load(f)

    if isinstance(model_classes, dict):
        model_classes = model_classes["classes"]

    model = tf.keras.models.load_model(MODEL_PATH)

    y_true = []
    y_pred = []
    skipped = 0

    image_extensions = {".jpg", ".jpeg", ".png", ".jfif", ".webp"}

    for folder_name, model_class in CLASS_MAPPING.items():

        folder = PLANTDOC_TEST / folder_name

        if not folder.is_dir():
            print(f"Missing folder: {folder_name}")
            continue

        true_index = model_classes.index(model_class)

        for image_path in sorted(folder.rglob("*")):

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in image_extensions:
                continue

            try:
                image = tf.keras.utils.load_img(
                    image_path,
                    target_size=IMAGE_SIZE
                )

                image_array = tf.keras.utils.img_to_array(image)
                image_array = np.expand_dims(image_array, axis=0)

                prediction = model.predict(image_array, verbose=0)[0]
                predicted_index = int(np.argmax(prediction))

                y_true.append(true_index)
                y_pred.append(predicted_index)

            except Exception as error:
                skipped += 1
                print(f"Skipped {image_path.name}: {error}")

    if not y_true:
        raise RuntimeError("No images were evaluated.")

    labels = sorted(set(y_true))
    evaluated_classes = [model_classes[i] for i in labels]

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    accuracy = accuracy_score(y_true, y_pred)

    print("\nPlantDoc Evaluation")
    print("-------------------")
    print(f"Images evaluated: {len(y_true)}")
    print(f"Images skipped: {skipped}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=evaluated_classes,
        zero_division=0
    ))

    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=evaluated_classes,
        yticklabels=evaluated_classes
    )

    plt.title("PlantDoc Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path = figures_dir / "plantdoc_confusion_matrix_finetuned.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\nConfusion matrix saved: {output_path}")


if __name__ == "__main__":
    main()

