import json
import matplotlib.pyplot as plt
from pathlib import Path

with open("probe-results/summary.json") as f:
    results = json.load(f)

LABELS = ["smooth", "disc", "artefact", "edge_on", "tight_spiral"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, size in zip(axes, ["015M", "095M"]):
    layers = sorted(results[size].keys(), key=lambda x: int(x.split("_")[1]))
    for label in LABELS:
        ys = [results[size][l][label]["balanced_accuracy"] if results[size][l][label] else None for l in layers]
        xs = [i for i, y in enumerate(ys) if y is not None]
        ys = [y for y in ys if y is not None]
        ax.plot(xs, ys, marker="o", label=label)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set_title(f"AstroPT {size}")
    ax.set_xlabel("Layer depth")

axes[0].set_ylabel("Balanced accuracy")
axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/probe_accuracy_by_layer.png", dpi=150, bbox_inches="tight")
print("Saved figures/probe_accuracy_by_layer.png")