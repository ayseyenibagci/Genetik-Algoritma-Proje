import os
import pandas as pd
ana_klasor = "Research Data Set each week"
veriler = []
for hafta in os.listdir(ana_klasor):
    hafta_yolu = os.path.join(ana_klasor, hafta)
    if os.path.isdir(hafta_yolu):
        for sinif in ["Wheats", "Weeds"]:
            sinif_yolu = os.path.join(hafta_yolu, sinif)
            for dosya in os.listdir(sinif_yolu):
                if dosya.lower().endswith((".jpg", ".jpeg", ".png")):
                    veriler.append({
                        "image_path": os.path.join(sinif_yolu, dosya),
                        "label": sinif,
                        "week": hafta
                    })
df = pd.DataFrame(veriler)
df.to_csv("labels.csv", index=False)
print(df.head())
print(df["label"].value_counts())
print("Toplam görüntü:", len(df))

