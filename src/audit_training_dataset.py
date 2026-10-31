
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image


# =====================================================
# 1. PATHS
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "processed" / "plantdoc_more_data"

TRAIN_CSV = DATA_DIR / "combined_train_final.csv"
VAL_CSV = DATA_DIR / "combined_val_final.csv"

REPORT_DIR = PROJECT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. LOAD DATASETS
# =====================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

train_df["split"] = "train"
val_df["split"] = "validation"

df = pd.concat([train_df, val_df], ignore_index=True)

required_columns = {"image_path", "label", "source"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Missing columns: {required_columns - set(df.columns)}"
    )

print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Total images:", len(df))


# =====================================================
# 3. FILE AND IMAGE QUALITY CHECK
# =====================================================

print("\nChecking image files...")

records = []

for index, row in df.iterrows():

    image_path = Path(str(row["image_path"]))

    record = {
        "image_path": str(image_path),
        "label": row["label"],
        "source": row["source"],
        "split": row["split"],
        "exists": image_path.exists(),
        "readable": False,
        "width": None,
        "height": None,
        "format": None,
        "file_size_bytes": None,
        "sha256": None,
        "error": None,
    }

    if image_path.exists():

        try:
            record["file_size_bytes"] = image_path.stat().st_size

            with Image.open(image_path) as img:
                img.verify()

            with Image.open(image_path) as img:
                record["width"], record["height"] = img.size
                record["format"] = img.format

            record["readable"] = True

            # Exact file-content duplicate detection
            sha256 = hashlib.sha256()

            with open(image_path, "rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    sha256.update(chunk)

            record["sha256"] = sha256.hexdigest()

        except Exception as error:
            record["error"] = str(error)

    else:
        record["error"] = "File not found"

    records.append(record)

    if (index + 1) % 500 == 0:
        print(f"Checked {index + 1}/{len(df)} images")


audit_df = pd.DataFrame(records)


# =====================================================
# 4. MISSING AND UNREADABLE IMAGES
# =====================================================

missing_df = audit_df[~audit_df["exists"]]

unreadable_df = audit_df[
    audit_df["exists"] & ~audit_df["readable"]
]

missing_df.to_csv(
    REPORT_DIR / "training_missing_images.csv",
    index=False
)

unreadable_df.to_csv(
    REPORT_DIR / "training_unreadable_images.csv",
    index=False
)


# =====================================================
# 5. EXACT DUPLICATE DETECTION
# =====================================================

print("\nChecking exact duplicate images...")

valid_df = audit_df[
    audit_df["readable"] & audit_df["sha256"].notna()
].copy()

duplicate_df = valid_df[
    valid_df.duplicated("sha256", keep=False)
].sort_values("sha256")

duplicate_df.to_csv(
    REPORT_DIR / "training_exact_duplicates.csv",
    index=False
)

# Duplicates appearing across train and validation
cross_split_duplicates = []

for sha, group in duplicate_df.groupby("sha256"):

    splits = set(group["split"])

    if "train" in splits and "validation" in splits:

        for _, row in group.iterrows():
            cross_split_duplicates.append(row.to_dict())

cross_split_df = pd.DataFrame(cross_split_duplicates)

cross_split_df.to_csv(
    REPORT_DIR / "training_validation_duplicates.csv",
    index=False
)


# =====================================================
# 6. CLASS AND SOURCE COUNTS
# =====================================================

class_counts = (
    df.groupby(["split", "label"])
    .size()
    .reset_index(name="image_count")
)

source_counts = (
    df.groupby(["split", "source"])
    .size()
    .reset_index(name="image_count")
)

class_counts.to_csv(
    REPORT_DIR / "training_class_distribution.csv",
    index=False
)

source_counts.to_csv(
    REPORT_DIR / "training_source_distribution.csv",
    index=False
)


# =====================================================
# 7. IMAGE DIMENSIONS AND FORMATS
# =====================================================

dimension_counts = (
    valid_df.groupby(["width", "height"])
    .size()
    .reset_index(name="image_count")
    .sort_values("image_count", ascending=False)
)

format_counts = (
    valid_df.groupby("format")
    .size()
    .reset_index(name="image_count")
)

dimension_counts.to_csv(
    REPORT_DIR / "training_image_dimensions.csv",
    index=False
)

format_counts.to_csv(
    REPORT_DIR / "training_image_formats.csv",
    index=False
)


# =====================================================
# 8. FINAL SUMMARY
# =====================================================

cross_split_group_count = (
    cross_split_df["sha256"].nunique()
    if not cross_split_df.empty else 0
)

summary = {
    "training_images": int(len(train_df)),
    "validation_images": int(len(val_df)),
    "total_images": int(len(df)),
    "missing_images": int(len(missing_df)),
    "unreadable_images": int(len(unreadable_df)),
    "exact_duplicate_rows": int(len(duplicate_df)),
    "cross_split_duplicate_groups": int(cross_split_group_count),
    "readable_images": int(audit_df["readable"].sum()),
}

summary_df = pd.DataFrame(
    list(summary.items()),
    columns=["metric", "value"]
)

summary_df.to_csv(
    REPORT_DIR / "training_dataset_audit_summary.csv",
    index=False
)


# =====================================================
# 9. PRINT RESULTS
# =====================================================

print("\n========================================")
print("TRAINING DATASET AUDIT COMPLETED")
print("========================================")

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nClass distribution:")
print(class_counts.to_string(index=False))

print("\nSource distribution:")
print(source_counts.to_string(index=False))

print("\nReports saved in:", REPORT_DIR)
