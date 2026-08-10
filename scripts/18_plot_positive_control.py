import json
import matplotlib.pyplot as plt
from pathlib import Path

with open("mi-results/summary_k_sweep.json") as f:
    astropt_data = json.load(f)["K_30"]
with open("mi-results/summary_tanh_control.json") as f:
    tanh_data = json.load(f)

fig, ax = plt.subplots(figsize=(8, 5))

for size in ["015M", "095M"]:
    layers = sorted(astropt_data[size].keys(), key=lambda x: int(x.split("_")[1]))
    vals = [astropt_data[size][l]["I_X_T"] for l in layers]
    n_layers = len(layers)
    xs_normalized = [i / (n_layers - 1) for i in range(n_layers)]  # normalize depth 0-1 for fair comparison
    ax.plot(xs_normalized, vals, marker="o", label=f"AstroPT {size}")

tanh_layers = sorted(tanh_data.keys(), key=lambda x: int(x.split("_")[1]))
tanh_vals = [tanh_data[l]["I_X_T"] for l in tanh_layers]
xs_tanh = [i / (len(tanh_layers) - 1) for i in range(len(tanh_layers))]
ax.plot(xs_tanh, tanh_vals, marker="s", color="red", linewidth=2, label="Tanh MLP (positive control)")

ax.set_xlabel("Normalized layer depth (0=first, 1=last)")
ax.set_ylabel("I(X;T) [bits, K=30]")
ax.set_title("Same estimator, K=30: AstroPT shows no compression, tanh control does")
ax.legend()
plt.tight_layout()
plt.savefig("figures/positive_control_comparison.png", dpi=150, bbox_inches="tight")
print("Saved figures/positive_control_comparison.png")