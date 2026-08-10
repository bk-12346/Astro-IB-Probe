import json
import matplotlib.pyplot as plt
from pathlib import Path

with open("mi-results/summary_pca.json") as f:
    data = json.load(f)

LABELS = ["smooth", "disc", "artefact", "edge_on", "tight_spiral"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for size, ls in [("015M", "--"), ("095M", "-")]:
    layers = sorted(data[size].keys(), key=lambda x: int(x.split("_")[1]))
    i_x_t = [data[size][l]["I_X_T_approx"] for l in layers]
    axes[0].plot(range(len(layers)), i_x_t, ls, marker="o", label=size)

axes[0].set_title("PCA-based I(X;T)")
axes[0].set_xlabel("Layer depth")
axes[0].set_ylabel("I(X;T) [nats, approx.]")
axes[0].legend()

for label in LABELS:
    layers = sorted(data["095M"].keys(), key=lambda x: int(x.split("_")[1]))
    vals = [data["095M"][l][f"I_T_{label}_approx"] for l in layers]
    axes[1].plot(range(len(layers)), vals, marker="o", label=label)

axes[1].set_title("PCA-based I(T;label), 095M")
axes[1].set_xlabel("Layer depth")
axes[1].set_ylabel("I(T;label) [nats, approx.]")
axes[1].legend()

plt.tight_layout()
Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/pca_mi_trends.png", dpi=150, bbox_inches="tight")
print("Saved figures/pca_mi_trends.png")