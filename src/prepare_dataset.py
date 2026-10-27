from pathlib import Path
import random
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "processed" / "tomato"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "split"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def main():
    random.seed(SEED)

    classes = sorted(p for p in SOURCE_DIR.iterdir() if p.is_dir())
    if not classes:
        raise RuntimeError(f"No class folders found in {SOURCE_DIR}")

    total_images = 0

    for class_dir in classes:
        images = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        random.shuffle(images)

        n = len(images)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)

        splits = {
            "train": images[:train_end],
            "validation": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split_name, split_images in splits.items():
            target_dir = OUTPUT_DIR / split_name / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)

            for image_path in split_images:
                target_path = target_dir / image_path.name
                if not target_path.exists():
                    shutil.copy2(image_path, target_path)

        counts = {name: len(files) for name, files in splits.items()}
        total_images += n
        print(
            f"{class_dir.name}: "
            f"train={counts['train']}, "
            f"validation={counts['validation']}, "
            f"test={counts['test']}"
        )

    print(f"\nClasses: {len(classes)}")
    print(f"Total source images: {total_images}")
    print(f"Split directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
