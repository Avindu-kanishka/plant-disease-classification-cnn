import pandas as pd
from PIL import Image, ImageDraw
from pathlib import Path
import math

csv_path = Path("reports/plantdoc_misclassified_images.csv")
output = Path("reports/priority_suspect_images.png")

df = pd.read_csv(csv_path)

suspects = [
    "%2320+Bacterial+Spot+and+Speck.jpg",
    "07.17.18-Common_Tomato_Diseases_Canker-258x300.jpg",
    "fungus-univ-of-minnesoeta.jpg",
    "early-blight-septoria-ls-fig-3.jpg",
    "page_2.jpg",
    "tmv-tomato-34.jpg",
    "tylcv-seminar-1-638.jpg"
]

selected = df[df["image_name"].isin(suspects)]

if len(selected) != len(suspects):
    print("WARNING: Some requested images were not found.")

cols = 2
cell_w, cell_h = 650, 430
rows = math.ceil(len(selected) / cols)

sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
draw = ImageDraw.Draw(sheet)

for i, (_, row) in enumerate(selected.iterrows()):
    x = (i % cols) * cell_w
    y = (i // cols) * cell_h

    path = Path(row["original_path"])

    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((cell_w - 30, 330))

        sheet.paste(img, (x + 10, y + 10))

        info = (
            f'{row["image_name"]}\n'
            f'Actual: {row["actual_class"]}\n'
            f'Predicted: {row["predicted_class"]} | '
            f'Confidence: {row["confidence"]}'
        )
        draw.multiline_text(
            (x + 10, y + 345),
            info,
            fill="black",
            spacing=5
        )

    except Exception as e:
        draw.text((x + 10, y + 10), f"ERROR: {path}\n{e}", fill="red")

sheet.save(output)
print("Images inspected:", len(selected))
print("Saved:", output.resolve())
