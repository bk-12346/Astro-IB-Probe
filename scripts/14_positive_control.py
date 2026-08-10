from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy

LABEL = "artefact"  # strongest, cleanest signal seen throughout
K = 30
N_LAYERS = 6  # matches 015M depth for a fair-ish comparison
HIDDEN_DIM = 128

manifest = pd.read_parquet("data/eval_subset/manifest.parquet")

# Use the AstroPT-095M final-layer embedding as a rich, fixed input feature set
# (this isolates "does depth+tanh compress" from "is the raw signal even present")
X = np.load("data/embeddings_095M.npy")[:, -1, :]  # (N, 768), final layer
X = StandardScaler().fit_transform(X)
y = manifest[LABEL].values

class TanhMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(n_layers)])
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_all_layers=False):
        activations = []
        for layer in self.layers:
            x = torch.tanh(layer(x))
            activations.append(x)
        logit = self.out(x)
        return (logit, activations) if return_all_layers else logit

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

model = TanhMLP(in_dim=X.shape[1], hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1 - y.mean()) / y.mean()]))

print("Training tanh MLP positive control...")
for epoch in range(200):
    optimizer.zero_grad()
    logits, _ = model(X_t, return_all_layers=True)
    loss = loss_fn(logits, y_t)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"  epoch {epoch+1}: loss={loss.item():.4f}")

model.eval()
with torch.no_grad():
    _, layer_activations = model(X_t, return_all_layers=True)

print(f"\nComputing I(X;T) per layer (K={K}), tanh MLP:")
tanh_control_results = {}
for i, act in enumerate(layer_activations):
    act_np = act.numpy()
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(act_np)
    _, counts = np.unique(clusters, return_counts=True)
    probs = counts / counts.sum()
    i_x_t = entropy(probs, base=2)
    tanh_control_results[f"layer_{i}"] = {"I_X_T": float(i_x_t), "I_X_T_max": float(np.log2(K))}
    print(f"  layer {i}: I(X;T) = {i_x_t:.3f} bits ({100*i_x_t/np.log2(K):.1f}% of max)")

Path("mi-results").mkdir(exist_ok=True)
with open("mi-results/summary_tanh_control.json", "w") as f:
    json.dump(tanh_control_results, f, indent=2)
print("\nSaved mi-results/summary_tanh_control.json")