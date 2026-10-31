
from pathlib import Path
import math
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_DIR / "reports" / "misclassified_images"
OUTPUT_DIR = PROJECT_DIR / "reports" / "misclassified_contact_sheets"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".jfif"}

for class_folder in sorted(INPUT_DIR.iterdir()):

    if not class_folder.is_dir():
        continue

    image_paths = sorted(
        p for p in class_folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        continue

    columns = 3
    rows = math.ceil(len(image_paths) / columns)

    fig, axes = plt.subplots(
        rows, columns,
        figsize=(15, rows * 4.5)
    )

    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, image_path in zip(axes, image_paths):

        try:
            image = Image.open(image_path).convert("RGB")
            ax.imshow(image)
            ax.axis("off")

            # Filename format:
            # 001__PRED-Late_blight__CONF-0.678__image.jpg
            parts = image_path.name.split("__")

            predicted = parts[1].replace("PRED-", "")
            confidence = parts[2].replace("CONF-", "")

            ax.set_title(
                f"Predicted: {predicted}\nConfidence: {confidence}",
                fontsize=9
            )

        except Exception as error:
            ax.set_title(f"Error: {error}")
            ax.axis("off")

    for ax in axes[len(image_paths):]:
        ax.axis("off")

    fig.suptitle(
        f"Actual Class: {class_folder.name}",
        fontsize=15,
        fontweight="bold"
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{class_folder.name}.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved:", output_path)

print("\nContact sheet generation completed!")
print("Output folder:", OUTPUT_DIR)
