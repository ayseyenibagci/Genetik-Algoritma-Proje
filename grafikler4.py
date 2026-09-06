import pandas as pd
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

import tensorflow as tf
from tensorflow import keras

# ==========================
# AYARLAR
# ==========================

IMG_SIZE = 224
SEED = 42

BEST_CHROMOSOME = [64, 0.3, 0.001, 16]
# [dense_neuron, dropout, learning_rate, batch_size]

# ==========================
# VERİYİ YÜKLE
# ==========================

df = pd.read_csv("labels.csv")

X = []
y = []

print("Resimler yükleniyor...")

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

print("Sınıflar:", encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=SEED,
    stratify=y_train
)

# ==========================
# MODEL
# ==========================

def build_model(chromosome):
    dense_neuron, dropout_rate, learning_rate, batch_size = chromosome

    base_model = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(dense_neuron, activation="relu"),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

model = build_model(BEST_CHROMOSOME)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=BEST_CHROMOSOME[3],
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# ==========================
# TAHMİN
# ==========================

y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)

# ==========================
# CLASSIFICATION REPORT
# ==========================

report = classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv("classification_report.csv")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# ==========================
# CONFUSION MATRIX
# ==========================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")

plt.xticks([0, 1], encoder.classes_)
plt.yticks([0, 1], encoder.classes_)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="black",
            fontsize=14,
            fontweight="bold"
        )

plt.colorbar()
plt.tight_layout()
plt.savefig("17_confusion_matrix.png", dpi=300)
plt.show()

# ==========================
# ROC CURVE
# ==========================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="#DC143C", linewidth=2.5, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")

plt.title("ROC Eğrisi", fontsize=14, fontweight="bold")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("18_roc_curve.png", dpi=300)
plt.show()

# ==========================
# METRİK BAR GRAFİĞİ
# ==========================

accuracy = report["accuracy"]
precision = report["weighted avg"]["precision"]
recall = report["weighted avg"]["recall"]
f1 = report["weighted avg"]["f1-score"]

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
values = [accuracy, precision, recall, f1]

colors = ["#6A5ACD", "#20B2AA", "#FFA500", "#FF1493"]

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics, values, color=colors)

plt.ylim(0, 1)
plt.title("MobileNetV2 + GA Performans Metrikleri", fontsize=14, fontweight="bold")
plt.ylabel("Skor")

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("19_performans_metrikleri.png", dpi=300)
plt.show()

print("\nGrafikler oluşturuldu:")
print("17_confusion_matrix.png")
print("18_roc_curve.png")
print("19_performans_metrikleri.png")
print("classification_report.csv")