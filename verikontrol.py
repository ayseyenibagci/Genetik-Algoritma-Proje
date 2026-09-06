import pandas as pd
import pandas as pd
from PIL import Image


df = pd.read_csv("labels.csv")
print("Satır-sütun:", df.shape)
print("\nSütunlar:")
print(df.columns)
print("\nHaftalar:")
print(df["week"].value_counts())
print("\nLabel dağılımı:")
print(df["label"].value_counts())

for i in range(5):
    img_path = df.iloc[i]["image_path"]

    img = Image.open(img_path)

    print(
        img_path,
        "->",
        img.size,
        img.mode
    )


genislikler = []
yukseklikler = []

for path in df["image_path"]:
    img = Image.open(path)

    genislikler.append(img.size[0])
    yukseklikler.append(img.size[1])

print("Minimum genişlik :", min(genislikler))
print("Maksimum genişlik:", max(genislikler))

print("Minimum yükseklik :", min(yukseklikler))
print("Maksimum yükseklik:", max(yukseklikler))




