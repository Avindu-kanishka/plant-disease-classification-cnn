
from pathlib import Path
import math
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_DIR / "reports" / "misclassified_images"
OUTPUT_DIR = PROJECT_DIR / "reports" / "misclassified_contact_sheets_59"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".jfif"}

if not INPUT_DIR.exists():
    raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

saved_count = 0

for class_folder in sorted(INPUT_DIR.iterdir()):

    if not class_folder.is_dir():
        continue

    image_paths = sorted(
        p for p in class_folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"Skipped (no images): {class_folder.name}")
        continue

    columns = 3
    rows = math.ceil(len(image_paths) / columns)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, rows * 4.5),
        squeeze=False
    )

    axes = axes.flatten()

    for ax, image_path in zip(axes, image_paths):
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")

            ax.imshow(image)
            ax.axis("off")

            parts = image_path.name.split("__")

            predicted = "Unknown"
            confidence = "Unknown"

            if len(parts) >= 3:
                predicted = parts[1].replace("PRED-", "")
                confidence = parts[2].replace("CONF-", "")

            ax.set_title(
                f"Predicted: {predicted}\nConfidence: {confidence}",
                fontsize=9
            )

        except Exception as error:
            ax.set_title(f"Image error: {error}", fontsize=8)
            ax.axis("off")

    for ax in axes[len(image_paths):]:
        ax.axis("off")

    fig.suptitle(
        f"Actual Class: {class_folder.name}",
        fontsize=15,
        fontweight="bold"
    )

    fig.tight_layout()

    safe_name = class_folder.name.replace("_", "-")
    output_path = OUTPUT_DIR / f"{safe_name}_contact_sheet.png"

    try:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        saved_count += 1
        print(f"Saved: {output_path}")
    finally:
        plt.close(fig)

print("\n========================================")
print("CONTACT SHEET GENERATION COMPLETED")
print("========================================")
print(f"Contact sheets saved: {saved_count}")
print(f"Output folder: {OUTPUT_DIR}")
