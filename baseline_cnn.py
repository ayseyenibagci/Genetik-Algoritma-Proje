import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

print("TensorFlow sürümü:", tf.__version__)
print("GPU listesi:", tf.config.list_physical_devices("GPU"))

df = pd.read_csv("labels.csv")
print("\nCSV okundu.")
print("Veri boyutu:", df.shape)
print(df["label"].value_counts())

IMG_SIZE = 224
X = []
y = []
print("\nResimler yükleniyor.")

for index, row in df.iterrows():
    if index % 100 == 0:
        print(index, "resim işlendi:", row["image_path"])
    try:
        img = Image.open(row["image_path"])
        img = img.convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img)
        X.append(img)
        y.append(row["label"])
    except Exception as e:
        print("Hatalı resim:", row["image_path"])
        print("Hata:", e)
print("\nToplam yüklenen resim:", len(X))
X = np.array(X, dtype=np.float32) / 255.0

encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("\nSınıflar:", encoder.classes_)
print("X şekli:", X.shape)
print("y şekli:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain:", X_train.shape)
print("Test :", X_test.shape)


model = keras.Sequential([
    keras.layers.Input(shape=(224, 224, 3)),

    keras.layers.Conv2D(32, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(128, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.GlobalAveragePooling2D(),

    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)

print("\nTest Accuracy:", acc)
print("Test Loss:", loss)



