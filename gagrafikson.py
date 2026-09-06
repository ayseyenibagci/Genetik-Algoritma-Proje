import matplotlib.pyplot as plt

# ==========================
# ÖRNEK KROMOZOMLAR
# MobileNetV2 + GA kromozomu:
# [dense_neuron, dropout, learning_rate, batch_size]
# ==========================

parent1 = [64, 0.3, 0.001, 16]
parent2 = [128, 0.5, 0.0005, 32]

gene_names = ["Dense", "Dropout", "Learning Rate", "Batch Size"]

# Çaprazlama noktası
crossover_point = 2

child1 = parent1[:crossover_point] + parent2[crossover_point:]
child2 = parent2[:crossover_point] + parent1[crossover_point:]

# Mutasyon örneği
before_mutation = child1.copy()
after_mutation = child1.copy()

# Örneğin dropout geni mutasyona uğrasın
mutated_gene_index = 1
after_mutation[mutated_gene_index] = 0.5

# ==========================
# 1. ÇAPRAZLAMA ŞEMASI
# ==========================

fig, ax = plt.subplots(figsize=(12, 5))
ax.axis("off")

rows = [
    ["Parent 1"] + parent1,
    ["Parent 2"] + parent2,
    ["Child 1"] + child1,
    ["Child 2"] + child2,
]

columns = ["Kromozom"] + gene_names

table = ax.table(
    cellText=rows,
    colLabels=columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("black")
    cell.set_linewidth(1.2)

    if row == 0:
        cell.set_facecolor("#4B0082")
        cell.set_text_props(color="white", weight="bold")

    elif row in [1, 2]:
        cell.set_facecolor("#E6E6FA")

    elif row in [3, 4]:
        cell.set_facecolor("#E0FFFF")
        cell.set_text_props(weight="bold")

# Çaprazlama noktası açıklaması
ax.text(
    0.5,
    0.08,
    f"Tek noktalı çaprazlama noktası: {crossover_point}. genden sonra",
    ha="center",
    fontsize=12,
    fontweight="bold"
)

plt.title(
    "Genetik Algoritmada Tek Noktalı Çaprazlama Örneği",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig("20_crossover_ornegi.png", dpi=300)
plt.show()

# ==========================
# 2. MUTASYON ŞEMASI
# ==========================

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis("off")

rows = [
    ["Mutasyon Öncesi"] + before_mutation,
    ["Mutasyon Sonrası"] + after_mutation,
]

columns = ["Kromozom"] + gene_names

table = ax.table(
    cellText=rows,
    colLabels=columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("black")
    cell.set_linewidth(1.2)

    if row == 0:
        cell.set_facecolor("#8B0000")
        cell.set_text_props(color="white", weight="bold")

    elif row == 1:
        cell.set_facecolor("#FFF0F5")

    elif row == 2:
        cell.set_facecolor("#FFE4E1")

# Mutasyona uğrayan hücreyi vurgula
# table row 2 = Mutasyon Sonrası, col mutated_gene_index + 1
mut_cell = table[(2, mutated_gene_index + 1)]
mut_cell.set_facecolor("#FFD700")
mut_cell.set_text_props(weight="bold", color="black")

ax.text(
    0.5,
    0.08,
    f"Mutasyon oranı: 0.20 | Değişen gen: {gene_names[mutated_gene_index]}",
    ha="center",
    fontsize=12,
    fontweight="bold"
)

plt.title(
    "Genetik Algoritmada Mutasyon Örneği",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig("21_mutasyon_ornegi.png", dpi=300)
plt.show()

print("Şemalar oluşturuldu:")
print("20_crossover_ornegi.png")
print("21_mutasyon_ornegi.png")