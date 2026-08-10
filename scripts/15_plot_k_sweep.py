import json
import matplotlib.pyplot as plt
from pathlib import Path

with open("mi-results/summary_k_sweep.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
for ax, k in zip(axes, ["K_10", "K_30", "K_50"]):
    for size, marker in [("015M", "o"), ("095M", "s")]:
        layers = sorted(data[k][size].keys(), key=lambda x: int(x.split("_")[1]))
        i_x_t = [data[k][size][l]["I_X_T"] for l in layers]
        i_x_t_max = data[k][size][layers[0]]["I_X_T_max"]
        ax.plot(range(len(layers)), i_x_t, marker=marker, label=f"{size}")
    ax.axhline(i_x_t_max, color="gray", linestyle="--", linewidth=1, label="theoretical max")
    ax.set_title(k.replace("_", "="))
    ax.set_xlabel("Layer depth")
    ax.set_ylabel("I(X;T) [bits]")
    ax.legend()

plt.tight_layout()
Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/ib_saturation_by_k.png", dpi=150, bbox_inches="tight")
print("Saved figures/ib_saturation_by_k.png")