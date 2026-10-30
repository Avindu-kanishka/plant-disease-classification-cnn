import pandas as pd
from pathlib import Path

BASE = Path("data/processed/plantdoc_more_data")
REPORTS = Path("reports")

train = pd.read_csv(BASE / "combined_train.csv")
val = pd.read_csv(BASE / "combined_val.csv")

def norm(p):
    return str(p).replace("/", "\\").strip().lower()

remove_train = {
r"D:\projects\Plant disease data set\PlantDoc-Dataset-master\train\Tomato Early blight leaf\early-blight-closeup-sm.jpg",
r"D:\projects\Plant disease data set\PlantDoc-Dataset-master\train\Tomato leaf yellow virus\tomato-yellow-leaf-curl-disease.jpg",
r"D:\projects\Plant disease data set\PlantDoc-Dataset-master\train\Tomato mold leaf\tylc.jpg",
r"D:\projects\Plant disease data set\PlantDoc-Dataset-master\train\Tomato Septoria leaf spot\early-blight-septoria-ls-fig-3.jpg",
r"D:\projects\Plant disease data set\PlantDoc-Dataset-master\train\Tomato Septoria leaf spot\tomato_V8.jpg",
}

remove_val = {
r"D:\projects\Plant disease data set\PlantDoc-Dataset-master\train\Tomato Early blight leaf\earlyblightpotato.jpg",
r"D:\projects\Plant disease data set\PlantDoc-Dataset-master\train\Tomato leaf late blight\IMG_5360.jpg",
r"D:\projects\Plant disease data set\PlantDoc-Dataset-master\train\Tomato leaf yellow virus\tylcv-seminar-1-638.jpg",
r"D:\projects\Plant disease data set\PlantDoc-Dataset-master\train\Tomato Septoria leaf spot\xgray-leaf-spot.jpg.pagespeed.ic.ltF8KfKtQ1.jpg",
}

remove_train = {norm(p) for p in remove_train}
remove_val = {norm(p) for p in remove_val}

train_mask = train.image_path.map(norm).isin(remove_train)
val_mask = val.image_path.map(norm).isin(remove_val)

if train_mask.sum() != 5:
    raise ValueError(f"Expected 5 training exclusions, found {train_mask.sum()}")

if val_mask.sum() != 4:
    raise ValueError(f"Expected 4 validation exclusions, found {val_mask.sum()}")

train_clean = train.loc[~train_mask].copy()
val_clean = val.loc[~val_mask].copy()

train_out = BASE / "combined_train_clean.csv"
val_out = BASE / "combined_val_clean.csv"

train_clean.to_csv(train_out, index=False)
val_clean.to_csv(val_out, index=False)

excluded = []
for split, original, mask in [
    ("train", train, train_mask),
    ("validation", val, val_mask)
]:
    for _, row in original.loc[mask].iterrows():
        excluded.append({
            "split": split,
            "image_path": row.image_path,
            "label": row.label,
            "source": row.source,
            "reason": "Exact duplicate, cross-species ambiguity, or split leakage"
        })

pd.DataFrame(excluded).to_csv(
    REPORTS / "plantdoc_cleaning_exclusions.csv", index=False
)

test_report = pd.DataFrame([{
    "image_path": r"D:\projects\Plant disease data set\PlantDoc-Dataset-master\test\Tomato leaf bacterial spot\tomato_V8.jpg",
    "label": "Tomato leaf bacterial spot",
    "action": "EXCLUDE",
    "reason": "Exact duplicate of training image labelled Septoria"
}])

test_report.to_csv(
    REPORTS / "plantdoc_test_exclusion_report.csv", index=False
)

print("DATASET CLEANING COMPLETED")
print("-" * 40)
print("Original train:", len(train))
print("Clean train:", len(train_clean))
print("Removed train:", int(train_mask.sum()))
print()
print("Original validation:", len(val))
print("Clean validation:", len(val_clean))
print("Removed validation:", int(val_mask.sum()))
print()
print("Test exclusions:", len(test_report))
print("Created:", train_out)
print("Created:", val_out)
print("Created: reports/plantdoc_cleaning_exclusions.csv")
print("Created: reports/plantdoc_test_exclusion_report.csv")
