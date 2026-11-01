from PIL import Image
from pathlib import Path
import pandas as pd

df = pd.read_csv("reports/plantdoc_misclassified_images.csv")

names = [
    "07.17.18-Common_Tomato_Diseases_Canker-258x300.jpg",
    "tmv-tomato-34.jpg",
    "fungus-univ-of-minnesoeta.jpg"
]

for name in names:
    row = df[df["image_name"] == name].iloc[0]
    path = Path(row["original_path"])

    with Image.open(path) as im:
        print("\nIMAGE:", name)
        print("Dimensions:", im.size)
        print("Format:", im.format)
        print("Mode:", im.mode)
        print("Actual label:", row["actual_class"])
        print("Prediction:", row["predicted_class"])
        print("Confidence:", row["confidence"])
