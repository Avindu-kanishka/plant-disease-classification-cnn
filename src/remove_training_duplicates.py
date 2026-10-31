
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = PROJECT_DIR / "reports"
DATA_DIR = PROJECT_DIR / "data" / "processed" / "plantdoc_more_data"

TRAIN_CSV = DATA_DIR / "combined_train_final.csv"
DUPLICATE_CSV = REPORT_DIR / "training_exact_duplicates.csv"

OUTPUT_CSV = DATA_DIR / "combined_train_deduplicated.csv"
EXCLUSION_REPORT = REPORT_DIR / "training_duplicate_exclusions.csv"

# Load files
train_df = pd.read_csv(TRAIN_CSV)
duplicates_df = pd.read_csv(DUPLICATE_CSV)

# Identify duplicate hashes
duplicate_hashes = duplicates_df["sha256"].dropna().unique()

# Keep the first occurrence of each exact duplicate group
hash_to_remove = set()
excluded_rows = []

for sha in duplicate_hashes:

    group = duplicates_df[
        duplicates_df["sha256"] == sha
    ]

    paths = set(group["image_path"])

    # Find matching rows in the training CSV
    matching_indices = train_df[
        train_df["image_path"].isin(paths)
    ].index.tolist()

    if len(matching_indices) > 1:

        # Keep first row, remove remaining duplicates
        for index in matching_indices[1:]:
            excluded_rows.append(train_df.loc[index].to_dict())
            hash_to_remove.add(train_df.loc[index, "image_path"])

# Create deduplicated dataset
clean_df = train_df[
    ~train_df["image_path"].isin(hash_to_remove)
].copy()

# Save new dataset and exclusion report
clean_df.to_csv(OUTPUT_CSV, index=False)

pd.DataFrame(excluded_rows).to_csv(
    EXCLUSION_REPORT,
    index=False
)

print("\n========== DUPLICATE CLEANING ==========")
print("Original training images:", len(train_df))
print("Removed duplicate entries:", len(excluded_rows))
print("Final training images:", len(clean_df))

print("\nNew training CSV:", OUTPUT_CSV)
print("Exclusion report:", EXCLUSION_REPORT)

print("\nOriginal CSV remains unchanged.")
print("Original image files remain unchanged.")
