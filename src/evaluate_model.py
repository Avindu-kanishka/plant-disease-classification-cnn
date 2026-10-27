from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "split" / "test"
MODEL_PATH = PROJECT_ROOT / "models" / "plant_disease_mobilenetv2.keras"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORT_DIR / "figures"

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(160, 160),
        batch_size=32,
        shuffle=False,
    )

    class_names = test_ds.class_names
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])

    print("Predicting test images...")
    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    ))

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        REPORT_DIR / "classification_report.csv"
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
    )

    fig, ax = plt.subplots(figsize=(14, 12))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    display.plot(
        ax=ax,
        xticks_rotation=75,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    ax.set_title("Tomato Disease Classification - Confusion Matrix")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    results = {
        "test_images": int(len(y_true)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "classes": class_names,
    }

    with open(REPORT_DIR / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nEvaluation complete.")
    print(json.dumps(results, indent=2))
    print("Saved: reports/classification_report.csv")
    print("Saved: reports/figures/confusion_matrix.png")

if __name__ == "__main__":
    main()
