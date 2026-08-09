from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from PIL import Image

from astropt.model_utils import load_astropt
from astropt.local_datasets import GalaxyImageDataset

def normalise(x):
    std, mean = torch.std_mean(x, dim=1, keepdim=True)
    return (x - mean) / (std + 1e-8)

data_transforms = transforms.Compose([transforms.Lambda(normalise)])
LABEL_COLS = ["smooth", "disc", "artefact", "edge_on", "tight_spiral"]
SIZES = ["015M", "095M"]
N_INPUT_PCA = 5
N_LAYER_PCA = 10

manifest = pd.read_parquet("data/eval_subset/manifest.parquet")

# --- Step 1: build the raw-input proxy (mean-pooled normalized patches, pre-model) ---
model_tmp = load_astropt(repo_id="smith42/astropt_v2.0", path="astropt/015M", weights_filename="ckpt.pt")
galproc = GalaxyImageDataset(None, spiral=True, transform={"images": data_transforms},
                              modality_registry=model_tmp.modality_registry)
del model_tmp

print("Building raw-input proxy...")
raw_input_vectors = []
for path in manifest["image_path"]:
    img = Image.open(path).convert("RGB")
    arr = torch.from_numpy(np.array(img).swapaxes(0, 2)).to(torch.float)
    galaxy = galproc.process_galaxy(arr).to(torch.float)
    raw_input_vectors.append(galaxy.mean(dim=0).numpy())  # mean-pool over patches
raw_input_vectors = np.stack(raw_input_vectors)
input_pca = PCA(n_components=N_INPUT_PCA, random_state=42).fit_transform(raw_input_vectors)
print(f"Input PCA shape: {input_pca.shape}, explained variance: {PCA(n_components=N_INPUT_PCA).fit(raw_input_vectors).explained_variance_ratio_.sum():.3f}")

# --- Step 2: per layer, PCA-reduce T and estimate continuous MI ---
pca_results = {}
for size in SIZES:
    embeddings = np.load(f"data/embeddings_{size}.npy")
    n_layers = embeddings.shape[1]
    pca_results[size] = {}

    for layer_idx in range(n_layers):
        T = embeddings[:, layer_idx, :]
        T_pca = PCA(n_components=N_LAYER_PCA, random_state=42).fit_transform(T)

        # I(X;T): sum of per-input-PC MI against T_pca features (approximate joint, overestimate)
        i_x_t_total = 0.0
        for j in range(N_INPUT_PCA):
            mi_per_feature = mutual_info_regression(T_pca, input_pca[:, j], random_state=42)
            i_x_t_total += mi_per_feature.sum() / N_INPUT_PCA  # average across input PCs, sum across T features

        layer_result = {"I_X_T_approx": float(i_x_t_total)}
        for label in LABEL_COLS:
            y = manifest[label].values
            mi_per_feature = mutual_info_classif(T_pca, y, random_state=42)
            layer_result[f"I_T_{label}_approx"] = float(mi_per_feature.sum())

        pca_results[size][f"layer_{layer_idx}"] = layer_result
        print(f"{size} layer {layer_idx}: I(X;T)~{layer_result['I_X_T_approx']:.4f}, " +
              ", ".join(f"I(T;{l})~{layer_result[f'I_T_{l}_approx']:.4f}" for l in LABEL_COLS))

Path("mi-results").mkdir(exist_ok=True)
with open("mi-results/summary_pca.json", "w") as f:
    json.dump(pca_results, f, indent=2)
print("\nSaved mi-results/summary_pca.json")