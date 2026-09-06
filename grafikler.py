import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================
# CSV DOSYALARINI OKU
# ==========================

summary = pd.read_csv("ga_generation_summary.csv")
all_chromosomes = pd.read_csv("ga_all_chromosomes.csv")
final_result = pd.read_csv("ga_final_result.csv")

# ==========================
# 1. MODEL KARŞILAŞTIRMA GRAFİĞİ
# ==========================

models = [
    "Baseline CNN",
    "Hafif CNN",
    "CNN + Augmentation",
    "CNN + GA",
    "MobileNetV2",
    "MobileNetV2 Fine Tuning",
    "MobileNetV2 + GA"
]

accuracies = [
    81.81,
    74.16,
    73.20,
    68.90,
    90.19,
    88.52,
    90.67
]

colors = [
    "#6A5ACD",
    "#20B2AA",
    "#FFA500",
    "#DC143C",
    "#2E8B57",
    "#8A2BE2",
    "#FF1493"
]

plt.figure(figsize=(12, 6))
bars = plt.bar(models, accuracies, color=colors)

plt.title("Model Performans Karşılaştırması", fontsize=15, fontweight="bold")
plt.ylabel("Test Accuracy (%)", fontsize=12)
plt.ylim(0, 100)
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{acc:.2f}%",
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

plt.tight_layout()
plt.savefig("01_model_karsilastirma.png", dpi=300)
plt.show()

# ==========================
# 2. GA FITNESS GRAFİĞİ
# ==========================

plt.figure(figsize=(9, 5))

plt.plot(
    summary["generation"],
    summary["best_fitness"],
    marker="o",
    linewidth=2.5,
    color="#FF1493",
    label="En iyi fitness"
)

plt.plot(
    summary["generation"],
    summary["avg_fitness"],
    marker="s",
    linewidth=2.5,
    color="#1E90FF",
    label="Ortalama fitness"
)

plt.title("GA Nesil Bazlı Fitness Değişimi", fontsize=15, fontweight="bold")
plt.xlabel("Nesil", fontsize=12)
plt.ylabel("Fitness / Validation Accuracy", fontsize=12)
plt.xticks(summary["generation"])
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

plt.tight_layout()
plt.savefig("02_ga_fitness_degisim.png", dpi=300)
plt.show()

# ==========================
# 3. POPÜLASYON FITNESS DAĞILIMI
# ==========================

plt.figure(figsize=(10, 6))

for gen in sorted(all_chromosomes["generation"].unique()):
    gen_data = all_chromosomes[all_chromosomes["generation"] == gen]

    plt.scatter(
        [gen] * len(gen_data),
        gen_data["fitness"],
        s=90,
        alpha=0.75,
        color="#2E8B57",
        edgecolor="black"
    )

plt.plot(
    summary["generation"],
    summary["best_fitness"],
    color="#DC143C",
    marker="o",
    linewidth=2.5,
    label="Neslin en iyi fitness değeri"
)

plt.title("Popülasyon Fitness Dağılımı", fontsize=15, fontweight="bold")
plt.xlabel("Nesil", fontsize=12)
plt.ylabel("Fitness / Validation Accuracy", fontsize=12)
plt.xticks(summary["generation"])
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

plt.tight_layout()
plt.savefig("03_populasyon_fitness_dagilimi.png", dpi=300)
plt.show()

# ==========================
# 4. EN İYİ KROMOZOMUN GÖSTERİMİ
# ==========================

best = final_result.iloc[0]

genes = ["Dense Neuron", "Dropout", "Learning Rate", "Batch Size"]
values = [
    str(best["dense_neuron"]),
    str(best["dropout"]),
    str(best["learning_rate"]),
    str(best["batch_size"])
]

fig, ax = plt.subplots(figsize=(10, 2.8))
ax.axis("off")

table_data = [genes, values]

table = ax.table(
    cellText=table_data,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("black")
    cell.set_linewidth(1.5)

    if row == 0:
        cell.set_facecolor("#6A5ACD")
        cell.set_text_props(color="white", weight="bold")
    else:
        cell.set_facecolor("#E6E6FA")
        cell.set_text_props(color="black", weight="bold")

plt.title("En İyi Kromozom Yapısı", fontsize=15, fontweight="bold", pad=20)

plt.tight_layout()
plt.savefig("04_en_iyi_kromozom.png", dpi=300)
plt.show()

# ==========================
# 5. GA AKIŞ DİYAGRAMI
# ==========================

fig, ax = plt.subplots(figsize=(8, 10))
ax.axis("off")

steps = [
    "Başlangıç Popülasyonu",
    "Fitness Hesabı",
    "Seçilim",
    "Çaprazlama",
    "Mutasyon",
    "Yeni Nesil",
    "En İyi Kromozom",
    "Final Model Testi"
]

y_positions = np.linspace(0.9, 0.1, len(steps))

for i, (step, y) in enumerate(zip(steps, y_positions)):
    ax.text(
        0.5,
        y,
        step,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#4169E1" if i < len(steps) - 2 else "#DC143C",
            edgecolor="black",
            linewidth=1.5
        )
    )

    if i < len(steps) - 1:
        ax.annotate(
            "",
            xy=(0.5, y_positions[i + 1] + 0.04),
            xytext=(0.5, y - 0.04),
            arrowprops=dict(
                arrowstyle="->",
                lw=2,
                color="black"
            )
        )

plt.title("Genetik Algoritma Akış Diyagramı", fontsize=16, fontweight="bold")

plt.tight_layout()
plt.savefig("05_ga_akis_diyagrami.png", dpi=300)
plt.show()

print("Grafikler oluşturuldu:")
print("01_model_karsilastirma.png")
print("02_ga_fitness_degisim.png")
print("03_populasyon_fitness_dagilimi.png")
print("04_en_iyi_kromozom.png")
print("05_ga_akis_diyagrami.png")