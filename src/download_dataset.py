from pathlib import Path
import tensorflow_datasets as tfds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "tensorflow_datasets"

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Dataset cache:", DATA_DIR)
    print("Loading PlantVillage dataset...")

    (dataset,), info = tfds.load(
        "plant_village",
        split=["train"],
        data_dir=str(DATA_DIR),
        with_info=True,
        as_supervised=True,
    )

    print("Dataset version:", info.version)
    print("Total classes:", info.features["label"].num_classes)
    print("Dataset loaded successfully.")

if __name__ == "__main__":
    main()
