import pandas as pd
from pathlib import Path

records = [
    ["%2320+Bacterial+Spot+and+Speck.jpg","Tomato___Bacterial_spot","Exclude candidate","Educational slide containing text and multiple images"],
    ["page_2.jpg","Tomato___Tomato_mosaic_virus","Exclude candidate","Mixed tomato and cucumber educational composite"],
    ["tylcv-seminar-1-638.jpg","Tomato___Tomato_Yellow_Leaf_Curl_Virus","Exclude candidate","Research presentation slide, not a single leaf photograph"],
    ["07.17.18-Common_Tomato_Diseases_Canker-258x300.jpg","Tomato___Bacterial_spot","Investigate","Filename indicates canker; dataset label is bacterial spot; source unverified"],
    ["tmv-tomato-34.jpg","Tomato___Tomato_mosaic_virus","Keep provisionally","Natural high-resolution photograph; disease identity not independently verified"],
    ["fungus-univ-of-minnesoeta.jpg","Tomato___Leaf_Mold","Keep provisionally","Natural photograph; disease identity not independently verified"],
    ["early-blight-septoria-ls-fig-3.jpg","Tomato___Septoria_leaf_spot","Keep provisionally","Natural leaf photograph; Septoria label appears plausible"],
    ["# additional review pending","Unknown","Investigate","Further source verification required"],
    ["# additional review pending","Unknown","Investigate","Further source verification required"],
    ["# additional review pending","Unknown","Investigate","Further source verification required"]
]

columns = ["image_name","actual_class","decision","reason"]
df = pd.DataFrame(records, columns=columns)

out = Path("reports/plantdoc_final_quality_review.csv")
df.to_csv(out, index=False)

print(df.groupby("decision").size().to_string())
print("\nSaved:", out.resolve())
