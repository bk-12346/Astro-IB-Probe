from pathlib import Path
from datasets import load_dataset
import pandas as pd
import boto3
from dotenv import load_dotenv
import os
import json

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

N_SAMPLES = 3000  # adjust if this proves too slow/fast in practice

LABEL_COLS = {
    "smooth": "smooth-or-featured_smooth_fraction",
    "disc": "smooth-or-featured_featured-or-disk_fraction",
    "artefact": "smooth-or-featured_artifact_fraction",
    "edge_on": "disk-edge-on_yes_fraction",
    "tight_spiral": "spiral-winding_tight_fraction",
}

print("Loading Galaxy Zoo DESI catalog...")
gz = pd.read_parquet("raw/galaxy_zoo_desi_friendly.parquet")  # adjust filename to match yours
gz = gz.set_index("dr8_id")

print("Streaming validation split and matching against catalog...")
ds = load_dataset("Smith42/galaxies", split="validation", streaming=True)

out_dir = Path("data/eval_subset/images")
out_dir.mkdir(parents=True, exist_ok=True)

records = []
matched = 0
for example in ds:
    if matched >= N_SAMPLES:
        break
    dr8_id = example["dr8_id"]
    if dr8_id not in gz.index:
        continue
    row = gz.loc[dr8_id]
    labels = {k: float(row[col] > 0.5) for k, col in LABEL_COLS.items()}
    img_path = out_dir / f"{dr8_id}.jpg"
    example["image"].save(img_path)
    records.append({"dr8_id": dr8_id, "image_path": str(img_path), **labels})
    matched += 1

print(f"Matched {matched} galaxies out of stream scanned.")

df = pd.DataFrame(records)
local_manifest = Path("data/eval_subset/manifest.parquet")
df.to_parquet(local_manifest)
print(f"Saved manifest: {local_manifest}, shape={df.shape}")

print("Uploading to S3...")
s3 = boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION"))
bucket = os.getenv("S3_BUCKET")

s3.upload_file(str(local_manifest), bucket, "raw-data/eval_subset/manifest.parquet")
for record in records:
    s3.upload_file(record["image_path"], bucket, f"raw-data/eval_subset/images/{record['dr8_id']}.jpg")

print("Done.")