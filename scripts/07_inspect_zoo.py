from pathlib import Path
import pandas as pd

raw_dir = Path("raw")
parquet_files = list(raw_dir.glob("*.parquet"))

if not parquet_files:
    print("No .parquet files found in raw/. Contents of raw/:")
    for f in raw_dir.iterdir():
        print(" -", f.name)
else:
    for f in parquet_files:
        print(f"\n=== {f.name} ===")
        df = pd.read_parquet(f)
        print("Shape:", df.shape)
        print("\nColumns:")
        for col in df.columns:
            print(" -", col)
        print("\nFirst few rows:")
        print(df.head())


# ***** OUTPUT *****

# Shape: (8689370, 40)

# Columns:
#  - dr8_id
#  - ra
#  - dec
#  - brickid
#  - objid
#  - hdf5_loc
#  - smooth-or-featured_smooth_fraction
#  - smooth-or-featured_featured-or-disk_fraction
#  - smooth-or-featured_artifact_fraction
#  - disk-edge-on_yes_fraction
#  - disk-edge-on_no_fraction
#  - has-spiral-arms_yes_fraction
#  - has-spiral-arms_no_fraction
#  - bar_strong_fraction
#  - bar_weak_fraction
#  - bar_no_fraction
#  - bulge-size_dominant_fraction
#  - bulge-size_large_fraction
#  - bulge-size_moderate_fraction
#  - bulge-size_small_fraction
#  - bulge-size_none_fraction
#  - how-rounded_round_fraction
#  - how-rounded_in-between_fraction
#  - how-rounded_cigar-shaped_fraction
#  - edge-on-bulge_boxy_fraction
#  - edge-on-bulge_none_fraction
#  - edge-on-bulge_rounded_fraction
#  - spiral-winding_tight_fraction
#  - spiral-winding_medium_fraction
#  - spiral-winding_loose_fraction
#  - spiral-arm-count_1_fraction
#  - spiral-arm-count_2_fraction
#  - spiral-arm-count_3_fraction
#  - spiral-arm-count_4_fraction
#  - spiral-arm-count_more-than-4_fraction
#  - spiral-arm-count_cant-tell_fraction
#  - merging_none_fraction
#  - merging_minor-disturbance_fraction
#  - merging_major-disturbance_fraction
#  - merging_merger_fraction

# First few rows:
#         dr8_id         ra        dec  ...  merging_minor-disturbance_fraction  merging_major-disturbance_fraction merging_merger_fraction
# 0  100000_1081  32.084931 -44.311422  ...                            0.121508                            0.023102                0.010543
# 1  100000_1401  32.140085 -44.293668  ...                            0.149447                            0.048128                0.206199
# 2  100000_1483  32.275015 -44.288957  ...                            0.171522                            0.044866                0.194324
# 3  100000_1509  32.045648 -44.287172  ...                            0.069307                            0.052556                0.711001
# 4  100000_1869  32.170627 -44.267273  ...                            0.254479                            0.045426                0.016788

# [5 rows x 40 columns]
