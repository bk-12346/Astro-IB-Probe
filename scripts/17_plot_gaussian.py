import json
import matplotlib.pyplot as plt
from pathlib import Path

with open("mi-results/summary_linear_gaussian.json") as f:
    data = json.load(f)

LABELS = ["smooth", "disc", "artefact", "edge_on", "tight_spiral"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, size in zip(axes, ["015M", "095M"]):
    layers = sorted(data[size].keys(), key=lambda x: int(x.split("_")[1]))
    for label in LABELS:
        vals = [data[size][l][label]["I_linear_bits"] if data[size][l][label] else None for l in layers]
        xs = [i for i, v in enumerate(vals) if v is not None]
        vals = [v for v in vals if v is not None]
        ax.plot(xs, vals, marker="o", label=label)
    ax.set_title(f"AstroPT {size}")
    ax.set_xlabel("Layer depth")

axes[0].set_ylabel("I(T;label) [bits, linear-Gaussian estimate]")
axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("figures/linear_gaussian_mi_trends.png", dpi=150, bbox_inches="tight")
print("Saved figures/linear_gaussian_mi_trends.png")