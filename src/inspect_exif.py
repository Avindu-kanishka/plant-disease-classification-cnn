from PIL import Image, ExifTags
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

    print("\n" + "=" * 55)
    print("IMAGE:", name)
    print("SOURCE PATH:", path)
    print("DATASET CLASS:", row["actual_class"])

    with Image.open(path) as img:
        exif = img.getexif()
        print("EXIF DATA:")

        if exif:
            for key, value in exif.items():
                tag = ExifTags.TAGS.get(key, key)
                print(f"  {tag}: {value}")
        else:
            print("  No EXIF metadata available")
