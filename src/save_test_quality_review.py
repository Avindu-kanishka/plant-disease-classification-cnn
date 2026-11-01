import pandas as pd
from pathlib import Path

data = [
    {
        "image_name": "%2320+Bacterial+Spot+and+Speck.jpg",
        "actual_class": "Tomato___Bacterial_spot",
        "decision": "Exclude candidate",
        "reason": "Educational slide with text and multiple images",
        "exact_duplicate_in_train_val": False
    },
    {
        "image_name": "page_2.jpg",
        "actual_class": "Tomato___Tomato_mosaic_virus",
        "decision": "Exclude candidate",
        "reason": "Mixed tomato and cucumber educational composite",
        "exact_duplicate_in_train_val": False
    },
    {
        "image_name": "tylcv-seminar-1-638.jpg",
        "actual_class": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "decision": "Exclude candidate",
        "reason": "Research presentation slide rather than a single leaf image",
        "exact_duplicate_in_train_val": False
    }
]

df = pd.DataFrame(data)
path = Path("reports/plantdoc_test_quality_review.csv")
df.to_csv(path, index=False)

print(df.to_string(index=False))
print("\nSaved:", path.resolve())
