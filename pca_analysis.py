import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -----------------------------------
# LOAD CLASS LABELS
# -----------------------------------

classes = pd.read_csv(
    "data/class.tsv",
    sep="\t",
    header=None
)

labels = classes.iloc[:, 0]

# -----------------------------------
# LOAD EXPRESSION MATRIX
# -----------------------------------

expr = pd.read_csv(
    "data/filtered.tsv.gz",
    sep="\t"
)

# Remove extra spaces from column names
expr.columns = expr.columns.astype(str).str.strip()

print("Expression matrix shape:")
print(expr.shape)

print("\nFirst few columns:")
print(expr.columns[:10])

# -----------------------------------
# LOAD GENE ANNOTATIONS
# -----------------------------------

cols = pd.read_csv(
    "data/columns.tsv.gz",
    sep="\t",
    comment="#"
)

# -----------------------------------
# FIND GENE IDs
# -----------------------------------

xbp1_id = str(
    cols[cols["GeneSymbol"] == "XBP1"]["ID"].values[0]
).strip()

gata3_id = str(
    cols[cols["GeneSymbol"] == "GATA3"]["ID"].values[0]
).strip()

print("\nXBP1 ID:", xbp1_id)
print("GATA3 ID:", gata3_id)

# -----------------------------------
# EXTRACT GENE EXPRESSIONS
# -----------------------------------

XBP1 = pd.to_numeric(expr[xbp1_id])
GATA3 = pd.to_numeric(expr[gata3_id])

# -----------------------------------
# FIGURE 1A
# -----------------------------------

colors = [
    "red" if x == 1 else "black"
    for x in labels
]

plt.figure(figsize=(6, 6))

plt.scatter(
    GATA3,
    XBP1,
    c=colors
)

plt.xlabel("GATA3")
plt.ylabel("XBP1")

plt.title("Figure 1a")

plt.savefig("figure1a.png")

plt.show()

# -----------------------------------
# PCA
# -----------------------------------

X = np.column_stack((GATA3, XBP1))

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

# -----------------------------------
# FIGURE 1C
# -----------------------------------

pc1 = X_pca[:, 0]

plt.figure(figsize=(8, 3))

for i in range(len(pc1)):

    y = 1 if labels[i] == 1 else 0

    color = "red" if labels[i] == 1 else "black"

    plt.scatter(
        pc1[i],
        y,
        c=color
    )

plt.yticks(
    [0, 1],
    ["ER-", "ER+"]
)

plt.xlabel("Projection onto PC1")

plt.title("Figure 1c")

plt.savefig("figure1c.png")

plt.show()

print("\nDone!")
print("Saved:")
print("- figure1a.png")
print("- figure1c.png")
