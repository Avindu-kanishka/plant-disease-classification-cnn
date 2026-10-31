
import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = Path(
    r"D:\projects\Plant disease data set\PlantDoc-Dataset-master\test"
)

OUTPUT_DIR = PROJECT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "plantdoc_test_quality_audit.csv"

CLASS_MAPPING = {
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".jfif"
}

rows = []

for folder_name, class_name in CLASS_MAPPING.items():

    folder = TEST_DIR / folder_name

    if not folder.exists():
        print("Missing folder:", folder)
        continue

    for image_path in sorted(folder.iterdir()):

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if image_path.name.lower() == "tomato_v8.jpg":
            continue

        rows.append({
            "image_name": image_path.name,
            "image_path": str(image_path),
            "original_class": class_name,
            "audit_status": "",
            "verified_class": "",
            "plant_species_correct": "",
            "disease_label_correct": "",
            "image_quality": "",
            "issue_type": "",
            "reviewer_notes": "",
        })

fieldnames = [
    "image_name",
    "image_path",
    "original_class",
    "audit_status",
    "verified_class",
    "plant_species_correct",
    "disease_label_correct",
    "image_quality",
    "issue_type",
    "reviewer_notes",
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as file:

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\n====================================")
print("PLANTDOC QUALITY AUDIT TEMPLATE")
print("====================================")
print("Total test images:", len(rows))
print("Audit CSV:", OUTPUT_CSV)
print("\nAudit template created successfully!")
