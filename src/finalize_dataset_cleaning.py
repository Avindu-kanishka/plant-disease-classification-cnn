import pandas as pd
from pathlib import Path

base = Path("data/processed/plantdoc_more_data")

train = pd.read_csv(base / "combined_train_clean.csv")
val = pd.read_csv(base / "combined_val_clean.csv")

remove_paths = {
    r"Tomato___Late_blight\image (1188).jpg",
    r"Tomato___Late_blight\image (264).jpg",
    r"Tomato___Late_blight\image (882).jpg",
    r"Tomato___Late_blight\image (275).JPG",
    r"Tomato___Late_blight\image (109).JPG",
}

def norm(p):
    return str(p).replace("/", "\\").lower()

mask = train["image_path"].map(
    lambda p: any(norm(p).endswith(norm(x)) for x in remove_paths)
)

if int(mask.sum()) != 5:
    raise ValueError(f"Expected 5 duplicates, found {int(mask.sum())}")

removed = train.loc[mask].copy()
train_final = train.loc[~mask].copy()

train_final.to_csv(base / "combined_train_final.csv", index=False)
val.to_csv(base / "combined_val_final.csv", index=False)

removed.to_csv(
    "reports/plantvillage_duplicate_exclusions.csv", index=False
)

print("FINAL CLEANING COMPLETED")
print("Training before:", len(train))
print("Training after:", len(train_final))
print("Training removed:", int(mask.sum()))
print("Validation:", len(val))
print("Validation images retained:", len(val))
print("Saved: combined_train_final.csv")
print("Saved: combined_val_final.csv")
print("Saved: reports/plantvillage_duplicate_exclusions.csv")
