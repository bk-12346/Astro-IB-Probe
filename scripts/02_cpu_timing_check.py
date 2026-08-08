import time
from astropt.model_utils import load_astropt
import torch

for size in ["015M", "095M", "850M"]:
    t0 = time.time()
    model = load_astropt(
        repo_id="smith42/astropt_v2.0",
        path=f"astropt/{size}",
        weights_filename="ckpt.pt",
    )
    model = model.to("cpu")
    model.eval()
    load_time = time.time() - t0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{size}: {n_params:,} params, loaded in {load_time:.1f}s")

    # crude single-forward-pass timing with dummy input matching block_size
    dummy = torch.randn(1, model.config.block_size, 3, 16, 16)  # adjust shape if this errors
    t0 = time.time()
    with torch.no_grad():
        try:
            _ = model.transformer.wte["images"](dummy)
        except Exception as e:
            print(f"  (dummy forward failed, shape mismatch expected — will fix in Phase 1/3: {e})")
    print(f"  forward attempt: {time.time()-t0:.2f}s")