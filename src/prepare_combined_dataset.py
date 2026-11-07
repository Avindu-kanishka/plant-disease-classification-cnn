
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

PLANTDOC_DIR = Path(
    r"D:\projects\Plant disease data set\PlantDoc-Dataset-master"
)

EXTRACTED_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "tensorflow_datasets"
    / "downloads"
    / "extracted"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "plantdoc_more_data"

# PlantVillage extracted dataset
PLANTVILLAGE_DIR_NAME = (
    "Plant_leave_diseases_dataset_without_augmentation"
)

# Seven disease classes and their PlantDoc folder names
CLASS_MAPPING = {
    "Tomato___Bacterial_spot": "Tomato leaf bacterial spot",
    "Tomato___Early_blight": "Tomato Early blight leaf",
    "Tomato___Late_blight": "Tomato leaf late blight",
    "Tomato___Leaf_Mold": "Tomato mold leaf",
    "Tomato___Septoria_leaf_spot": "Tomato Septoria leaf spot",
    "Tomato___Tomato_mosaic_virus": "Tomato leaf mosaic virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato leaf yellow virus",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_images(folder):
    if not folder.exists():
        return []

    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Locate extracted PlantVillage dataset
    extracted_folders = list(EXTRACTED_DIR.iterdir())

    dataset_candidates = [
        folder / PLANTVILLAGE_DIR_NAME
        for folder in extracted_folders
        if folder.is_dir()
        and (folder / PLANTVILLAGE_DIR_NAME).is_dir()
    ]

    if not dataset_candidates:
        raise FileNotFoundError(
            "PlantVillage extracted dataset was not found."
        )

    plantvillage_dir = dataset_candidates[0]

    combined_records = []
    plantdoc_test_records = []

    print("\nCollecting PlantVillage images...")

    # Collect PlantVillage images
    for class_name in CLASS_MAPPING:
        class_dir = plantvillage_dir / class_name
        images = get_images(class_dir)

        print(f"{class_name}: {len(images)} images")

        for image_path in images:
            combined_records.append({
                "image_path": str(image_path.resolve()),
                "label": class_name,
                "source": "PlantVillage",
            })

    print("\nCollecting PlantDoc training images...")

    # Collect PlantDoc training images only
    for pv_class, plantdoc_folder in CLASS_MAPPING.items():
        class_dir = PLANTDOC_DIR / "train" / plantdoc_folder
        images = get_images(class_dir)

        print(f"{plantdoc_folder}: {len(images)} images")

        for image_path in images:
            combined_records.append({
                "image_path": str(image_path.resolve()),
                "label": pv_class,
                "source": "PlantDoc",
            })

    combined_df = pd.DataFrame(combined_records)

    if combined_df.empty:
        raise ValueError("No training images were found.")

    # Check missing classes
    found_classes = set(combined_df["label"])
    missing_classes = set(CLASS_MAPPING) - found_classes

    if missing_classes:
        raise ValueError(f"Missing classes: {missing_classes}")

    # Stratified 80/20 train-validation split
    train_df, val_df = train_test_split(
        combined_df,
        test_size=0.20,
        random_state=42,
        stratify=combined_df["label"],
    )

    # Collect original PlantDoc test images separately
    print("\nCollecting PlantDoc test images...")

    for pv_class, plantdoc_folder in CLASS_MAPPING.items():
        class_dir = PLANTDOC_DIR / "test" / plantdoc_folder
        images = get_images(class_dir)

        print(f"{plantdoc_folder}: {len(images)} test images")

        for image_path in images:
            plantdoc_test_records.append({
                "image_path": str(image_path.resolve()),
                "label": pv_class,
                "source": "PlantDoc",
            })

    test_df = pd.DataFrame(plantdoc_test_records)

    # Save CSV manifests
    train_df.to_csv(OUTPUT_DIR / "combined_train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "combined_val.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "plantdoc_test.csv", index=False)

    # Save class counts
    counts = combined_df.groupby(
        ["label", "source"]
    ).size().unstack(fill_value=0)

    counts.to_csv(OUTPUT_DIR / "class_counts.csv")

    print("\n========== DATASET SUMMARY ==========")
    print(f"Total combined images: {len(combined_df)}")
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print(f"Separate PlantDoc test images: {len(test_df)}")

    print("\nClass counts by source:")
    print(counts)

    print(f"\nCSV files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
