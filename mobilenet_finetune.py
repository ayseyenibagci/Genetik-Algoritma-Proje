import pandas as pd
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras

print("TensorFlow:", tf.__version__)

df = pd.read_csv("labels.csv")
IMG_SIZE = 224
X = []
y = []
print("Resimler yükleniyor.")
for index, row in df.iterrows():

    if index % 200 == 0:
        print(index, "resim işlendi")

    img = Image.open(row["image_path"])

    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(img)

    X.append(img)
    y.append(row["label"])

X = np.array(X, dtype=np.float32)

X = keras.applications.mobilenet_v2.preprocess_input(X)

encoder = LabelEncoder()
y = encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Train:", X_train.shape)
print("Test :", X_test.shape)


base_model = keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)



base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False



model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(
        128,
        activation="relu"
    ),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.summary()



early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1
)


history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.20,
    callbacks=[early_stop, reduce_lr]
)


loss, acc = model.evaluate(
    X_test,
    y_test,
    verbose=0
)
print("MobileNetV2 Fine Tune Accuracy:", acc)
print("MobileNetV2 Fine Tune Loss:", loss)



