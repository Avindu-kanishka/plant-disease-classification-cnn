import pandas as pd
import hashlib
from pathlib import Path

suspects = [
    "%2320+Bacterial+Spot+and+Speck.jpg",
    "page_2.jpg",
    "tylcv-seminar-1-638.jpg"
]

report = pd.read_csv("reports/plantdoc_misclassified_images.csv")

train = pd.read_csv("data/processed/plantdoc_more_data/combined_train_deduplicated.csv")
val = pd.read_csv("data/processed/plantdoc_more_data/combined_val_final.csv")

def sha(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except Exception:
        return None

for name in suspects:
    row = report[report["image_name"] == name].iloc[0]
    target_hash = sha(row["original_path"])

    print("\nIMAGE:", name)
    print("Actual class:", row["actual_class"])

    matches = []

    for split, df in [("TRAIN", train), ("VALIDATION", val)]:
        for _, r in df.iterrows():
            if sha(r["image_path"]) == target_hash:
                matches.append((split, r["label"]))

    print("Exact duplicate matches:", matches if matches else "None")
