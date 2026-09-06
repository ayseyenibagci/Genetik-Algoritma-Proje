import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ga_all_chromosomes.csv")

# ==========================
# 1. DENSE NEURON - FITNESS
# ==========================

plt.figure(figsize=(8,5))
plt.scatter(df["dense_neuron"], df["fitness"], s=100, color="#6A5ACD", edgecolor="black")
plt.title("Dense Nöron Sayısının Fitness Üzerindeki Etkisi", fontweight="bold")
plt.xlabel("Dense Nöron Sayısı")
plt.ylabel("Fitness / Validation Accuracy")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("06_dense_fitness.png", dpi=300)
plt.show()

# ==========================
# 2. DROPOUT - FITNESS
# ==========================

plt.figure(figsize=(8,5))
plt.scatter(df["dropout"], df["fitness"], s=100, color="#FF1493", edgecolor="black")
plt.title("Dropout Oranının Fitness Üzerindeki Etkisi", fontweight="bold")
plt.xlabel("Dropout Oranı")
plt.ylabel("Fitness / Validation Accuracy")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("07_dropout_fitness.png", dpi=300)
plt.show()

# ==========================
# 3. LEARNING RATE - FITNESS
# ==========================

plt.figure(figsize=(8,5))
plt.scatter(df["learning_rate"], df["fitness"], s=100, color="#2E8B57", edgecolor="black")
plt.title("Learning Rate Değerinin Fitness Üzerindeki Etkisi", fontweight="bold")
plt.xlabel("Learning Rate")
plt.ylabel("Fitness / Validation Accuracy")
plt.xscale("log")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("08_learningrate_fitness.png", dpi=300)
plt.show()

# ==========================
# 4. BATCH SIZE - FITNESS
# ==========================

plt.figure(figsize=(8,5))
plt.scatter(df["batch_size"], df["fitness"], s=100, color="#FFA500", edgecolor="black")
plt.title("Batch Size Değerinin Fitness Üzerindeki Etkisi", fontweight="bold")
plt.xlabel("Batch Size")
plt.ylabel("Fitness / Validation Accuracy")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("09_batchsize_fitness.png", dpi=300)
plt.show()

# ==========================
# 5. NESİLLERE GÖRE DENSE DEĞİŞİMİ
# ==========================

plt.figure(figsize=(9,5))
for gen in sorted(df["generation"].unique()):
    temp = df[df["generation"] == gen]
    plt.scatter(
        [gen] * len(temp),
        temp["dense_neuron"],
        s=90,
        color="#6A5ACD",
        edgecolor="black",
        alpha=0.75
    )

plt.title("Nesillere Göre Dense Nöron Değerleri", fontweight="bold")
plt.xlabel("Nesil")
plt.ylabel("Dense Nöron Sayısı")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("10_nesil_dense_degisim.png", dpi=300)
plt.show()

# ==========================
# 6. NESİLLERE GÖRE DROPOUT DEĞİŞİMİ
# ==========================

plt.figure(figsize=(9,5))
for gen in sorted(df["generation"].unique()):
    temp = df[df["generation"] == gen]
    plt.scatter(
        [gen] * len(temp),
        temp["dropout"],
        s=90,
        color="#FF1493",
        edgecolor="black",
        alpha=0.75
    )

plt.title("Nesillere Göre Dropout Değerleri", fontweight="bold")
plt.xlabel("Nesil")
plt.ylabel("Dropout Oranı")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("11_nesil_dropout_degisim.png", dpi=300)
plt.show()

# ==========================
# 7. NESİLLERE GÖRE LEARNING RATE DEĞİŞİMİ
# ==========================

plt.figure(figsize=(9,5))
for gen in sorted(df["generation"].unique()):
    temp = df[df["generation"] == gen]
    plt.scatter(
        [gen] * len(temp),
        temp["learning_rate"],
        s=90,
        color="#2E8B57",
        edgecolor="black",
        alpha=0.75
    )

plt.title("Nesillere Göre Learning Rate Değerleri", fontweight="bold")
plt.xlabel("Nesil")
plt.ylabel("Learning Rate")
plt.yscale("log")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("12_nesil_learningrate_degisim.png", dpi=300)
plt.show()

# ==========================
# 8. NESİLLERE GÖRE BATCH SIZE DEĞİŞİMİ
# ==========================

plt.figure(figsize=(9,5))
for gen in sorted(df["generation"].unique()):
    temp = df[df["generation"] == gen]
    plt.scatter(
        [gen] * len(temp),
        temp["batch_size"],
        s=90,
        color="#FFA500",
        edgecolor="black",
        alpha=0.75
    )

plt.title("Nesillere Göre Batch Size Değerleri", fontweight="bold")
plt.xlabel("Nesil")
plt.ylabel("Batch Size")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("13_nesil_batchsize_degisim.png", dpi=300)
plt.show()

print("Detay GA grafikleri oluşturuldu:")
print("06_dense_fitness.png")
print("07_dropout_fitness.png")
print("08_learningrate_fitness.png")
print("09_batchsize_fitness.png")
print("10_nesil_dense_degisim.png")
print("11_nesil_dropout_degisim.png")
print("12_nesil_learningrate_degisim.png")
print("13_nesil_batchsize_degisim.png")