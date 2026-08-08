from huggingface_hub import hf_hub_download
import torch

local_path = hf_hub_download(
    repo_id="smith42/astroPT",
    filename="models/fully_trained/0309M_params/030000_ckpt.pt",
)
print(f"Downloaded to: {local_path}")

ckpt = torch.load(local_path, map_location="cpu", weights_only=False)
print(type(ckpt))
if isinstance(ckpt, dict):
    print("Keys:", list(ckpt.keys()))
    for k, v in ckpt.items():
        if k != "model" and k != "optimizer":  # skip printing huge state dicts
            print(f"  {k}: {type(v)} = {v if not hasattr(v, 'shape') else v.shape}")