from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy

LABEL_COLS = ["smooth", "disc", "artefact", "edge_on", "tight_spiral"]
SIZES = ["015M", "095M"]
K_VALUES = [10, 30, 50]  # sweep — original run was K=30 only

manifest = pd.read_parquet("data/eval_subset/manifest.parquet")

all_k_results = {}
for K in K_VALUES:
    print(f"\n########## K = {K} ##########")
    ib_results = {}
    for size in SIZES:
        embeddings = np.load(f"data/embeddings_{size}.npy")
        n_layers = embeddings.shape[1]
        ib_results[size] = {}

        for layer_idx in range(n_layers):
            T = embeddings[:, layer_idx, :]

            kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
            cluster_assignments = kmeans.fit_predict(T)

            _, counts = np.unique(cluster_assignments, return_counts=True)
            probs = counts / counts.sum()
            i_x_t = entropy(probs, base=2)

            layer_result = {"I_X_T": float(i_x_t), "I_X_T_max": float(np.log2(K))}
            for label in LABEL_COLS:
                y = manifest[label].values
                mi_nats = mutual_info_score(cluster_assignments, y)
                layer_result[f"I_T_{label}"] = float(mi_nats / np.log(2))

            ib_results[size][f"layer_{layer_idx}"] = layer_result

        print(f"{size}: I(X;T) range = "
              f"[{min(ib_results[size][l]['I_X_T'] for l in ib_results[size]):.3f}, "
              f"{max(ib_results[size][l]['I_X_T'] for l in ib_results[size]):.3f}] "
              f"out of max {np.log2(K):.3f}")

    all_k_results[f"K_{K}"] = ib_results

Path("mi-results").mkdir(exist_ok=True)
with open("mi-results/summary_k_sweep.json", "w") as f:
    json.dump(all_k_results, f, indent=2)
print("\nSaved mi-results/summary_k_sweep.json")


# from pathlib import Path
# import json
# import numpy as np
# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.metrics import mutual_info_score
# from scipy.stats import entropy

# LABEL_COLS = ["smooth", "disc", "artefact", "edge_on", "tight_spiral"]
# SIZES = ["015M", "095M"]
# N_CLUSTERS = 30  # matches the spirit of classical ~30-bin discretization; we'll sanity-check sensitivity after

# manifest = pd.read_parquet("data/eval_subset/manifest.parquet")

# ib_results = {}
# for size in SIZES:
#     embeddings = np.load(f"data/embeddings_{size}.npy")  # (N, n_layers, n_embd)
#     n_layers = embeddings.shape[1]
#     ib_results[size] = {}

#     for layer_idx in range(n_layers):
#         T = embeddings[:, layer_idx, :]

#         kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
#         cluster_assignments = kmeans.fit_predict(T)

#         # I(X; T) = H(cluster distribution), in bits
#         _, counts = np.unique(cluster_assignments, return_counts=True)
#         probs = counts / counts.sum()
#         i_x_t = entropy(probs, base=2)

#         layer_result = {"I_X_T": float(i_x_t)}
#         for label in LABEL_COLS:
#             y = manifest[label].values
#             mi_nats = mutual_info_score(cluster_assignments, y)
#             layer_result[f"I_T_{label}"] = float(mi_nats / np.log(2))  # convert nats -> bits

#         ib_results[size][f"layer_{layer_idx}"] = layer_result
#         print(f"{size} layer {layer_idx}: I(X;T)={i_x_t:.3f} bits, " +
#               ", ".join(f"I(T;{l})={layer_result[f'I_T_{l}']:.3f}" for l in LABEL_COLS))

# Path("mi-results").mkdir(exist_ok=True)
# with open("mi-results/summary.json", "w") as f:
#     json.dump(ib_results, f, indent=2)
# print("\nSaved mi-results/summary.json")