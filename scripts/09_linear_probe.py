from pathlib import Path
from functools import partial
import json
import time
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from PIL import Image
import boto3
from dotenv import load_dotenv
import os
import gc

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
s3 = boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION"))
bucket = os.getenv("S3_BUCKET")

from astropt.model_utils import load_astropt
from astropt.local_datasets import GalaxyImageDataset

def normalise(x):
    std, mean = torch.std_mean(x, dim=1, keepdim=True)
    return (x - mean) / (std + 1e-8)

data_transforms = transforms.Compose([transforms.Lambda(normalise)])

LABEL_COLS = ["smooth", "disc", "artefact", "edge_on", "tight_spiral"]
SIZES = {"015M": "astropt/015M", "095M": "astropt/095M"}
N_SAMPLES = 3000  # adjust as decided
CHECKPOINT_EVERY = 250

manifest = pd.read_parquet("data/eval_subset/manifest.parquet").head(N_SAMPLES)

def load_and_process_image(path, galproc):
    img = Image.open(path).convert("RGB")
    arr = torch.from_numpy(np.array(img).swapaxes(0, 2)).to(torch.float)
    galaxy = galproc.process_galaxy(arr).to(torch.float)
    positions = torch.arange(0, len(galaxy), dtype=torch.long)
    return galaxy, positions

@torch.no_grad()
def get_all_layer_embeddings(model, images, positions):
    tok_emb = model.transformer.wte["images"](images)
    pos_emb = model.transformer.wpe["images"](positions)
    x = model.transformer.drop(tok_emb + pos_emb)
    per_layer = []
    for block in model.transformer.h:
        x = block(x)
        per_layer.append(x.mean(dim=1))
    return torch.stack(per_layer, dim=1)

results = {}
for size_name, hf_path in SIZES.items():
    print(f"\n=== {size_name} ===")
    embeddings_path = Path(f"data/embeddings_{size_name}.npy")

    if embeddings_path.exists():
        print(f"Found existing {embeddings_path}, loading instead of recomputing")
        all_embeddings = np.load(embeddings_path)
    else:
        model = load_astropt(repo_id="smith42/astropt_v2.0", path=hf_path, weights_filename="ckpt.pt")
        model.eval()
        galproc = GalaxyImageDataset(
            None, spiral=True, transform={"images": data_transforms},
            modality_registry=model.modality_registry,
        )

        all_embeddings = []
        t0 = time.time()
        for i, path in enumerate(manifest["image_path"]):
            galaxy, positions = load_and_process_image(path, galproc)
            emb = get_all_layer_embeddings(model, galaxy.unsqueeze(0), positions.unsqueeze(0))
            all_embeddings.append(emb.squeeze(0).numpy())

            if (i + 1) % CHECKPOINT_EVERY == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                remaining = (len(manifest) - (i + 1)) / rate
                print(f"  {i+1}/{len(manifest)} done, {elapsed:.0f}s elapsed, ~{remaining/60:.1f} min remaining")
                np.save(embeddings_path, np.stack(all_embeddings))
                s3.upload_file(str(embeddings_path), bucket, f"activations/{embeddings_path.name}")

        all_embeddings = np.stack(all_embeddings)
        np.save(embeddings_path, all_embeddings)
        s3.upload_file(str(embeddings_path), bucket, f"activations/{embeddings_path.name}")
        print(f"Extracted embeddings: {all_embeddings.shape}")

        del model
        gc.collect()

    n_layers = all_embeddings.shape[1]
    results[size_name] = {}
    for layer_idx in range(n_layers):
        X = all_embeddings[:, layer_idx, :]
        layer_results = {}
        for label in LABEL_COLS:
            y = manifest[label].values
            halfway = len(X) // 2
            y_train, y_test = y[:halfway], y[halfway:]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                layer_results[label] = None
                continue
            probe = LogisticRegression(max_iter=1000, class_weight="balanced")
            probe.fit(X[:halfway], y_train)
            preds = probe.predict(X[halfway:])
            layer_results[label] = {
                "accuracy": accuracy_score(y_test, preds),
                "balanced_accuracy": balanced_accuracy_score(y_test, preds),
                "majority_baseline": max(y_test.mean(), 1 - y_test.mean()),
            }
        results[size_name][f"layer_{layer_idx}"] = layer_results
        print(f"  layer {layer_idx}: " + ", ".join(
            f"{l}={layer_results[l]['balanced_accuracy']:.3f}" if layer_results[l] is not None else f"{l}=skipped"
            for l in LABEL_COLS
        ))

Path("probe-results").mkdir(exist_ok=True)
summary_path = Path("probe-results/summary.json")
with open(summary_path, "w") as f:
    json.dump(results, f, indent=2)
s3.upload_file(str(summary_path), bucket, "probe-results/summary.json")
print(f"\nSaved and uploaded {summary_path}")

# from pathlib import Path
# from functools import partial
# import json
# import numpy as np
# import pandas as pd
# import torch
# from torchvision import transforms
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, balanced_accuracy_score
# from PIL import Image

# from astropt.model_utils import load_astropt
# from astropt.local_datasets import GalaxyImageDataset

# def normalise(x):
#     std, mean = torch.std_mean(x, dim=1, keepdim=True)
#     return (x - mean) / (std + 1e-8)

# data_transforms = transforms.Compose([transforms.Lambda(normalise)])

# LABEL_COLS = ["smooth", "disc", "artefact", "edge_on", "tight_spiral"]
# SIZES = {"015M": "astropt/015M", "095M": "astropt/095M"}

# # manifest = pd.read_parquet("data/eval_subset/manifest.parquet")
# manifest = pd.read_parquet("data/eval_subset/manifest.parquet").head(50)    #running for a small subset first

# def load_and_process_image(path, galproc):
#     img = Image.open(path).convert("RGB")
#     arr = torch.from_numpy(np.array(img).swapaxes(0, 2)).to(torch.float)
#     galaxy = galproc.process_galaxy(arr).to(torch.float)
#     positions = torch.arange(0, len(galaxy), dtype=torch.long)
#     return galaxy, positions

# @torch.no_grad()
# def get_all_layer_embeddings(model, images, positions):
#     """Mirrors model.get_embeddings() internals, but captures every layer instead of just the last."""
#     tok_emb = model.transformer.wte["images"](images)
#     pos_emb = model.transformer.wpe["images"](positions)
#     x = model.transformer.drop(tok_emb + pos_emb)
#     per_layer = []
#     for block in model.transformer.h:
#         x = block(x)
#         per_layer.append(x.mean(dim=1))  # mean-pool over patch sequence, per paper's convention
#     return torch.stack(per_layer, dim=1)  # (batch, n_layers, n_embd)

# results = {}
# for size_name, hf_path in SIZES.items():
#     print(f"\n=== {size_name} ===")
#     model = load_astropt(repo_id="smith42/astropt_v2.0", path=hf_path, weights_filename="ckpt.pt")
#     model.eval()

#     galproc = GalaxyImageDataset(
#         None, spiral=True, transform={"images": data_transforms},
#         modality_registry=model.modality_registry,
#     )

#     all_embeddings = []
#     for path in manifest["image_path"]:
#         galaxy, positions = load_and_process_image(path, galproc)
#         emb = get_all_layer_embeddings(model, galaxy.unsqueeze(0), positions.unsqueeze(0))
#         all_embeddings.append(emb.squeeze(0).numpy())  # (n_layers, n_embd)

#     all_embeddings = np.stack(all_embeddings)  # (n_samples, n_layers, n_embd)
#     n_layers = all_embeddings.shape[1]
#     print(f"Extracted embeddings: {all_embeddings.shape}")

#     results[size_name] = {}
    
#     for layer_idx in range(n_layers):
#         X = all_embeddings[:, layer_idx, :]
#         layer_results = {}
#         for label in LABEL_COLS:
#             y = manifest[label].values
#             halfway = len(X) // 2
#             y_train, y_test = y[:halfway], y[halfway:]

#             if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
#                 print(f"  layer {layer_idx}: skipping '{label}' — only one class present in this sample")
#                 layer_results[label] = None
#                 continue

#             probe = LogisticRegression(max_iter=1000, class_weight="balanced")
#             probe.fit(X[:halfway], y_train)
#             preds = probe.predict(X[halfway:])
#             layer_results[label] = {
#                 "accuracy": accuracy_score(y_test, preds),
#                 "balanced_accuracy": balanced_accuracy_score(y_test, preds),
#                 "majority_baseline": max(y_test.mean(), 1 - y_test.mean()),
#             }
#         # for label in LABEL_COLS:
#         #     y = manifest[label].values
#         #     halfway = len(X) // 2
#         #     probe = LogisticRegression(max_iter=1000, class_weight="balanced")
#         #     probe.fit(X[:halfway], y[:halfway])
#         #     preds = probe.predict(X[halfway:])
#         #     layer_results[label] = {
#         #         "accuracy": accuracy_score(y[halfway:], preds),
#         #         "balanced_accuracy": balanced_accuracy_score(y[halfway:], preds),
#         #         "majority_baseline": max(y[halfway:].mean(), 1 - y[halfway:].mean()),
#         #     }
#         results[size_name][f"layer_{layer_idx}"] = layer_results
#         print(f"  layer {layer_idx}: " + ", ".join(
#             f"{l}={layer_results[l]['balanced_accuracy']:.3f}" if layer_results[l] else f"{l}=skipped"
#             for l in LABEL_COLS
#         ))
#         # print(f"  layer {layer_idx}: " + ", ".join(
#         #     f"{l}={layer_results[l]['balanced_accuracy']:.3f}" for l in LABEL_COLS
#         # ))

#     del model
#     import gc; gc.collect()

# Path("probe-results").mkdir(exist_ok=True)
# with open("probe-results/summary.json", "w") as f:
#     json.dump(results, f, indent=2)
# print("\nSaved probe-results/summary.json")