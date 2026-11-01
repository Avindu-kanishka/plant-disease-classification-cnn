from pathlib import Path
import pandas as pd
import re

# 1. Correct the quality review report
records = [
    ["%2320+Bacterial+Spot+and+Speck.jpg", "Tomato___Bacterial_spot", "Exclude candidate", "Educational slide with text and multiple images"],
    ["page_2.jpg", "Tomato___Tomato_mosaic_virus", "Exclude candidate", "Mixed tomato and cucumber educational composite"],
    ["tylcv-seminar-1-638.jpg", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Exclude candidate", "Research presentation slide"],
    ["07.17.18-Common_Tomato_Diseases_Canker-258x300.jpg", "Tomato___Bacterial_spot", "Investigate", "Filename refers to canker; dataset label is bacterial spot"],
    ["tmv-tomato-34.jpg", "Tomato___Tomato_mosaic_virus", "Investigate", "Natural photograph; disease identity not independently verified"],
    ["fungus-univ-of-minnesoeta.jpg", "Tomato___Leaf_Mold", "Investigate", "Natural photograph; disease identity not independently verified"],
    ["early-blight-septoria-ls-fig-3.jpg", "Tomato___Septoria_leaf_spot", "Keep provisionally", "Natural leaf photograph; label appears plausible"]
]

columns = ["image_name", "actual_class", "decision", "reason"]
review = pd.DataFrame(records, columns=columns)
review.to_csv("reports/plantdoc_final_quality_review.csv", index=False)

print("Corrected quality review:", len(review), "actual images")

# 2. Create a separate secondary evaluation script
source = Path("src/evaluate_plantdoc_original.py")
destination = Path("src/evaluate_plantdoc_original_secondary56.py")

code = source.read_text(encoding="utf-8")

old_exclusion = '{"tomato_v8.jpg", "tomato-mosaic-1wdr2jx-300x225.jpg"}'
new_exclusion = (
    '{"tomato_v8.jpg", "tomato-mosaic-1wdr2jx-300x225.jpg", '
    '"%2320+bacterial+spot+and+speck.jpg", "page_2.jpg", '
    '"tylcv-seminar-1-638.jpg"}'
)

if old_exclusion not in code:
    raise RuntimeError("Original exclusion condition not found. No script created.")

code = code.replace(old_exclusion, new_exclusion)

replacements = {
    "plantdoc_aware_classification_report.csv":
        "plantdoc_original_secondary56_classification_report.csv",
    "plantdoc_aware_evaluation_results.json":
        "plantdoc_original_secondary56_evaluation_results.json",
    "plantdoc_aware_confusion_matrix.png":
        "plantdoc_original_secondary56_confusion_matrix.png",
    "PlantDoc-Aware MobileNetV2 - PlantDoc Test Confusion Matrix":
        "Original MobileNetV2 - Secondary 56 Image Sensitivity Analysis",
    '"dataset": "PlantDoc Test"':
        '"dataset": "PlantDoc Test - Secondary 56-image sensitivity analysis"'
}

for old, new in replacements.items():
    code = code.replace(old, new)

destination.write_text(code, encoding="utf-8")

print("Secondary script created:", destination)
print("Original evaluation script unchanged.")
