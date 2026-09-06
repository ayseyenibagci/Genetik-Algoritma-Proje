import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# VERİ
# ==========================

df = pd.read_csv("ga_all_chromosomes.csv")

# ==========================
# 1. DENSE FREKANSI
# ==========================

plt.figure(figsize=(8,5))

dense_counts = df["dense_neuron"].value_counts().sort_index()

bars = plt.bar(
    dense_counts.index.astype(str),
    dense_counts.values,
    color="#6A5ACD"
)

plt.title(
    "Dense Nöron Geninin Seçilme Frekansı",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Dense Nöron")
plt.ylabel("Seçilme Sayısı")

for bar in bars:
    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.2,
        str(int(bar.get_height())),
        ha="center",
        fontweight="bold"
    )

plt.tight_layout()
plt.savefig("14_gen_frekansi.png", dpi=300)
plt.show()

# ==========================
# 2. FITNESS HISTOGRAM
# ==========================

plt.figure(figsize=(8,5))

plt.hist(
    df["fitness"],
    bins=10,
    color="#20B2AA",
    edgecolor="black"
)

plt.title(
    "Tüm Kromozomların Fitness Dağılımı",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Fitness")
plt.ylabel("Kromozom Sayısı")

plt.tight_layout()
plt.savefig("15_fitness_histogram.png", dpi=300)
plt.show()

# ==========================
# 3. HEATMAP
# ==========================

heatmap_data = df.pivot_table(
    values="fitness",
    index="dense_neuron",
    columns="learning_rate",
    aggfunc="mean"
)

plt.figure(figsize=(8,6))

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".3f",
    cmap="RdYlGn"
)

plt.title(
    "Dense Nöron - Learning Rate Fitness Isı Haritası",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig("16_dense_lr_heatmap.png", dpi=300)
plt.show()

print("\nGrafikler oluşturuldu:")
print("14_gen_frekansi.png")
print("15_fitness_histogram.png")
print("16_dense_lr_heatmap.png")
