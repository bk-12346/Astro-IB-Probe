import json
import matplotlib.pyplot as plt
from pathlib import Path

with open("probe-results/summary.json") as f:
    probe_data = json.load(f)
with open("mi-results/summary_k_sweep.json") as f:
    kmeans_data = json.load(f)["K_30"]
with open("mi-results/summary_pca.json") as f:
    pca_data = json.load(f)
with open("mi-results/summary_linear_gaussian.json") as f:
    linear_data = json.load(f)

SIZE = "095M"  # deeper model, cleanest signal — master figure focuses here
LABEL = "artefact"  # strongest, most consistent signal across all methods

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Panel 1: probe balanced accuracy
layers = sorted(probe_data[SIZE].keys(), key=lambda x: int(x.split("_")[1]))
vals = [probe_data[SIZE][l][LABEL]["balanced_accuracy"] for l in layers]
axes[0,0].plot(range(len(layers)), vals, marker="o", color="tab:blue")
axes[0,0].axhline(0.5, color="gray", linestyle="--")
axes[0,0].set_title("Linear probe: balanced accuracy")
axes[0,0].set_ylabel("Balanced accuracy")

# Panel 2: linear-Gaussian I(T;label)
vals = [linear_data[SIZE][l][LABEL]["I_linear_bits"] for l in layers]
axes[0,1].plot(range(len(layers)), vals, marker="o", color="tab:green")
axes[0,1].set_title("Linear-Gaussian: I(T;label)")
axes[0,1].set_ylabel("I(T;label) [bits]")

# Panel 3: PCA-based I(T;label) — the one that DIDN'T cleanly recover the trend
vals = [pca_data[SIZE][l][f"I_T_{LABEL}_approx"] for l in layers]
axes[1,0].plot(range(len(layers)), vals, marker="o", color="tab:orange")
axes[1,0].set_title("PCA-based: I(T;label) — noisy, no clean trend")
axes[1,0].set_ylabel("I(T;label) [nats, approx.]")
axes[1,0].set_xlabel("Layer depth")

# Panel 4: K-means I(X;T) — flat/saturated, the compression question
vals = [kmeans_data[SIZE][l]["I_X_T"] for l in layers]
i_max = kmeans_data[SIZE][layers[0]]["I_X_T_max"]
axes[1,1].plot(range(len(layers)), vals, marker="o", color="tab:red")
axes[1,1].axhline(i_max, color="gray", linestyle="--", label="theoretical max")
axes[1,1].set_title("K-means: I(X;T) — flat, no compression")
axes[1,1].set_ylabel("I(X;T) [bits]")
axes[1,1].set_xlabel("Layer depth")
axes[1,1].legend()

fig.suptitle(f"AstroPT {SIZE}, label='{LABEL}': fitting is real (panels 1-2), compression is not resolved (panels 3-4)", fontsize=11)
plt.tight_layout()
plt.savefig("figures/master_comparison.png", dpi=150, bbox_inches="tight")
print("Saved figures/master_comparison.png")