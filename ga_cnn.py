import random
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras



IMG_SIZE = 224
EPOCHS = 5

POPULATION_SIZE = 6
GENERATIONS = 5
MUTATION_RATE = 0.2

FILTER_1_VALUES = [16, 32, 64]
FILTER_2_VALUES = [32, 64, 128]
FILTER_3_VALUES = [64, 128]
DENSE_VALUES = [64, 128, 256]
DROPOUT_VALUES = [0.2, 0.3, 0.4, 0.5]
LEARNING_RATE_VALUES = [0.001, 0.0005, 0.0001]
BATCH_SIZE_VALUES = [16, 32]


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

X = np.array(X, dtype=np.float32) / 255.0

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)



def create_chromosome():
    return [
        random.choice(FILTER_1_VALUES),
        random.choice(FILTER_2_VALUES),
        random.choice(FILTER_3_VALUES),
        random.choice(DENSE_VALUES),
        random.choice(DROPOUT_VALUES),
        random.choice(LEARNING_RATE_VALUES),
        random.choice(BATCH_SIZE_VALUES)
    ]



def build_model(chromosome):
    filter1, filter2, filter3, dense_neuron, dropout_rate, learning_rate, batch_size = chromosome

    model = keras.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),

        keras.layers.Conv2D(filter1, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(filter2, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(filter3, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),

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

    batch_size = chromosome[6]
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
            chromosome[gene_index] = random.choice(FILTER_1_VALUES)
        elif gene_index == 1:
            chromosome[gene_index] = random.choice(FILTER_2_VALUES)
        elif gene_index == 2:
            chromosome[gene_index] = random.choice(FILTER_3_VALUES)
        elif gene_index == 3:
            chromosome[gene_index] = random.choice(DENSE_VALUES)
        elif gene_index == 4:
            chromosome[gene_index] = random.choice(DROPOUT_VALUES)
        elif gene_index == 5:
            chromosome[gene_index] = random.choice(LEARNING_RATE_VALUES)
        elif gene_index == 6:
            chromosome[gene_index] = random.choice(BATCH_SIZE_VALUES)

        print("Mutasyon oldu! Yeni kromozom:", chromosome)

    return chromosome


population = [create_chromosome() for _ in range(POPULATION_SIZE)]
best_chromosome = None
best_fitness = 0
for generation in range(GENERATIONS):
    print(f"{generation + 1}. NESİL")
    fitness_scores = []
    for chromosome in population:
        score = fitness(chromosome)
        fitness_scores.append(score)
        if score > best_fitness:
            best_fitness = score
            best_chromosome = chromosome.copy()
    print("\nBu neslin fitness değerleri:", fitness_scores)
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
print("GA TAMAMLANDI")
print("En iyi kromozom:", best_chromosome)
print("En iyi validation accuracy:", best_fitness)



best_model = build_model(best_chromosome)

best_model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=best_chromosome[6],
    validation_data=(X_val, y_val),
    verbose=1
)

test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)

print("\nGA-CNN Test Accuracy:", test_acc)
print("GA-CNN Test Loss:", test_loss)




