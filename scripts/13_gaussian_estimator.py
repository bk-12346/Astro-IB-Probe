from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import pointbiserialr

LABEL_COLS = ["smooth", "disc", "artefact", "edge_on", "tight_spiral"]
SIZES = ["015M", "095M"]

manifest = pd.read_parquet("data/eval_subset/manifest.parquet")

linear_mi_results = {}
for size in SIZES:
    embeddings = np.load(f"data/embeddings_{size}.npy")
    n_layers = embeddings.shape[1]
    linear_mi_results[size] = {}

    for layer_idx in range(n_layers):
        T = embeddings[:, layer_idx, :]
        halfway = len(T) // 2
        layer_result = {}

        for label in LABEL_COLS:
            y = manifest[label].values
            y_train, y_test = y[:halfway], y[halfway:]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                layer_result[label] = None
                continue

            probe = LogisticRegression(max_iter=1000, class_weight="balanced")
            probe.fit(T[:halfway], y_train)
            # continuous decision score (pre-threshold), on held-out test set
            scores = probe.decision_function(T[halfway:])

            rho, _ = pointbiserialr(y_test, scores)
            rho = np.clip(rho, -0.999, 0.999)  # avoid log(0)
            i_linear_gaussian = -0.5 * np.log2(1 - rho**2)

            layer_result[label] = {"rho": float(rho), "I_linear_bits": float(i_linear_gaussian)}

        linear_mi_results[size][f"layer_{layer_idx}"] = layer_result
        print(f"{size} layer {layer_idx}: " + ", ".join(
            f"{l}={layer_result[l]['I_linear_bits']:.3f}" if layer_result[l] else f"{l}=skipped"
            for l in LABEL_COLS
        ))

Path("mi-results").mkdir(exist_ok=True)
with open("mi-results/summary_linear_gaussian.json", "w") as f:
    json.dump(linear_mi_results, f, indent=2)
print("\nSaved mi-results/summary_linear_gaussian.json")