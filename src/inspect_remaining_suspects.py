import pandas as pd
from PIL import Image, ImageDraw
from pathlib import Path

df = pd.read_csv("reports/plantdoc_misclassified_images.csv")

suspects = [
    "07.17.18-Common_Tomato_Diseases_Canker-258x300.jpg",
    "tmv-tomato-34.jpg",
    "fungus-univ-of-minnesoeta.jpg"
]

selected = df[df["image_name"].isin(suspects)]

W, H = 1200, 450
sheet = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(sheet)

for i, (_, row) in enumerate(selected.iterrows()):
    x = i * 400
    path = Path(row["original_path"])

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((380, 340))
        sheet.paste(im, (x + 10, 10))

    info = (
        f'{row["image_name"]}\n'
        f'Actual: {row["actual_class"]}\n'
        f'Predicted: {row["predicted_class"]}\n'
        f'Confidence: {row["confidence"]}'
    )
    draw.multiline_text((x + 10, 355), info, fill="black", spacing=4)

output = Path("reports/remaining_suspect_images.png")
sheet.save(output)

print("Images inspected:", len(selected))
print("Saved:", output.resolve())
