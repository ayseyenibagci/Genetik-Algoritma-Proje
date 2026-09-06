import os

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import random
import pandas as pd
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = 224
EPOCHS = 5

POPULATION_SIZE = 6
GENERATIONS = 5
MUTATION_RATE = 0.2

DENSE_VALUES = [64, 128, 256, 512]
DROPOUT_VALUES = [0.2, 0.3, 0.4, 0.5]
LEARNING_RATE_VALUES = [0.001, 0.0005, 0.0001]
BATCH_SIZE_VALUES = [16, 32]

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

df = pd.read_csv("labels.csv")

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

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


def create_chromosome():
    return [
        random.choice(DENSE_VALUES),
        random.choice(DROPOUT_VALUES),
        random.choice(LEARNING_RATE_VALUES),
        random.choice(BATCH_SIZE_VALUES)
    ]


def build_model(chromosome):
    dense_neuron = chromosome[0]
    dropout_rate = chromosome[1]
    learning_rate = chromosome[2]

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


def fitness(chromosome):
    print("\nKromozom deneniyor:", chromosome)

    batch_size = chromosome[3]
    model = build_model(chromosome)

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=0
    )

    val_acc = history.history["val_accuracy"][-1]
    print("Fitness / Validation Accuracy:", val_acc)

    keras.backend.clear_session()

    return val_acc


def selection(population, fitness_scores):
    sorted_population = [
        x for _, x in sorted(
            zip(fitness_scores, population),
            key=lambda pair: pair[0],
            reverse=True
        )
    ]
    return sorted_population[:2]


def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 2)

    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    return child1, child2


def mutate(chromosome):
    chromosome = chromosome.copy()

    if random.random() < MUTATION_RATE:
        gene_index = random.randint(0, len(chromosome) - 1)

        if gene_index == 0:
            chromosome[gene_index] = random.choice(DENSE_VALUES)
        elif gene_index == 1:
            chromosome[gene_index] = random.choice(DROPOUT_VALUES)
        elif gene_index == 2:
            chromosome[gene_index] = random.choice(LEARNING_RATE_VALUES)
        elif gene_index == 3:
            chromosome[gene_index] = random.choice(BATCH_SIZE_VALUES)

        print("Mutasyon oldu! Yeni kromozom:", chromosome)

    return chromosome


population = [create_chromosome() for _ in range(POPULATION_SIZE)]

best_chromosome = None
best_fitness = 0

generation_best_scores = []
generation_avg_scores = []

all_results = []

for generation in range(GENERATIONS):
    print(f"{generation + 1}. NESİL")

    fitness_scores = []

    for chromosome_index, chromosome in enumerate(population):
        score = fitness(chromosome)
        fitness_scores.append(score)

        all_results.append({
            "generation": generation + 1,
            "chromosome_index": chromosome_index + 1,
            "dense_neuron": chromosome[0],
            "dropout": chromosome[1],
            "learning_rate": chromosome[2],
            "batch_size": chromosome[3],
            "fitness": score
        })

        if score > best_fitness:
            best_fitness = score
            best_chromosome = chromosome.copy()

    generation_best_scores.append(max(fitness_scores))
    generation_avg_scores.append(sum(fitness_scores) / len(fitness_scores))

    print("\nBu neslin fitness değerleri:", fitness_scores)
    print("Bu neslin en iyi fitness:", max(fitness_scores))
    print("Bu neslin ortalama fitness:", sum(fitness_scores) / len(fitness_scores))
    print("Şu ana kadarki en iyi kromozom:", best_chromosome)
    print("Şu ana kadarki en iyi fitness:", best_fitness)

    parents = selection(population, fitness_scores)

    new_population = parents.copy()

    while len(new_population) < POPULATION_SIZE:
        child1, child2 = crossover(parents[0], parents[1])

        child1 = mutate(child1)
        child2 = mutate(child2)

        new_population.append(child1)

        if len(new_population) < POPULATION_SIZE:
            new_population.append(child2)

    population = new_population

print("MOBILENETV2 + GA TAMAMLANDI")
print("En iyi kromozom:", best_chromosome)
print("En iyi validation accuracy:", best_fitness)

best_model = build_model(best_chromosome)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

best_model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=best_chromosome[3],
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)

print("\nMobileNetV2 + GA Test Accuracy:", test_acc)
print("MobileNetV2 + GA Test Loss:", test_loss)

summary_df = pd.DataFrame({
    "generation": list(range(1, GENERATIONS + 1)),
    "best_fitness": generation_best_scores,
    "avg_fitness": generation_avg_scores
})

all_results_df = pd.DataFrame(all_results)

final_result_df = pd.DataFrame([{
    "best_chromosome": str(best_chromosome),
    "dense_neuron": best_chromosome[0],
    "dropout": best_chromosome[1],
    "learning_rate": best_chromosome[2],
    "batch_size": best_chromosome[3],
    "best_validation_accuracy": best_fitness,
    "test_accuracy": test_acc,
    "test_loss": test_loss
}])

summary_df.to_csv("ga_generation_summary.csv", index=False)
all_results_df.to_csv("ga_all_chromosomes.csv", index=False)
final_result_df.to_csv("ga_final_result.csv", index=False)

print("\nCSV dosyaları kaydedildi:")
print("ga_generation_summary.csv")
print("ga_all_chromosomes.csv")
print("ga_final_result.csv")