from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "plant_disease_mobilenetv2.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "reports" / "class_names.json"
PLANTDOC_TEST = Path(
    r"D:\projects\Plant disease data set\PlantDoc-Dataset-master\test"
)
IMAGE_SIZE = (160, 160)

# PlantDoc folder name -> existing model class name
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

    # Support either a list of class names or a dictionary containing them.
    if isinstance(model_classes, dict):
        model_classes = model_classes.get("classes", model_classes)

    if not isinstance(model_classes, list):
        raise ValueError("Unexpected format in reports/class_names.json")

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

        if model_class not in model_classes:
            print(f"Model class not found: {model_class}")
            skipped += 1
            continue

        true_index = model_classes.index(model_class)

        for image_path in sorted(folder.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in image_extensions:
                continue

            try:
                image = tf.keras.utils.load_img(
                    image_path, target_size=IMAGE_SIZE
                )
                image_array = tf.keras.utils.img_to_array(image)
                image_array = np.expand_dims(image_array, axis=0)

                probabilities = model.predict(image_array, verbose=0)[0]
                predicted_index = int(np.argmax(probabilities))

                y_true.append(true_index)
                y_pred.append(predicted_index)
            except Exception as exc:
                skipped += 1
                print(f"Skipped {image_path.name}: {exc}")

    if not y_true:
        raise RuntimeError(
            "No images evaluated. Check PlantDoc folder names and class_names.json."
        )

    target_names = [model_classes[i] for i in sorted(set(y_true))]
    labels = sorted(set(y_true))

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(model_classes))),
        target_names=model_classes,
        output_dict=True,
        zero_division=0,
    )

    # Save the full report and the summary.
    output_path = PROJECT_ROOT / "reports" / "plantdoc_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "dataset": "PlantDoc test split",
        "evaluated_images": len(y_true),
        "skipped_images": skipped,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": report,
        "note": (
            "Exploratory cross-dataset evaluation. PlantDoc labels may not "
            "be perfectly equivalent to PlantVillage labels; results are not "
            "directly comparable to the original PlantVillage test accuracy."
        ),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nPlantDoc evaluation complete")
    print(f"Images evaluated: {len(y_true)}")
    print(f"Images skipped: {skipped}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=[model_classes[i] for i in labels],
        zero_division=0,
    ))
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
