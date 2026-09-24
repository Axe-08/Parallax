"""
Parallax Kaggle-to-S3 Bridge Template
====================================
Copy and paste this script into a Kaggle Notebook code cell.
Provides instant authentication to the team's central S3 bucket,
pulls the Golden Benchmark split, and sets up local evaluation.
"""

# ==============================================================================
# 1. AWS Credentials Configuration
# ==============================================================================
# Option A: Paste directly from TEAM_MANUAL.md credentials
# Option B: Use Kaggle Secrets (Add-ons -> Secrets -> AWS_KEY, AWS_SECRET)

import os
from pathlib import Path

import boto3
import pandas as pd

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "PASTE_YOUR_KEY_ID_HERE")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "PASTE_YOUR_SECRET_KEY_HERE")
AWS_REGION = "us-east-1"
S3_BUCKET = "amazon-ml-challange-2026-parallax"

# Configure boto3 session
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)
s3 = session.client("s3")

# ==============================================================================
# 2. Pull Golden Benchmark Split from S3
# ==============================================================================
data_dir = Path("/kaggle/working/data")
data_dir.mkdir(parents=True, exist_ok=True)

train_file = data_dir / "golden_train.parquet"
val_file = data_dir / "golden_val.parquet"

print("==> Pulling Golden Benchmark Split from S3...")
try:
    s3.download_file(S3_BUCKET, "challenge-data/golden_train.parquet", str(train_file))
    s3.download_file(S3_BUCKET, "challenge-data/golden_val.parquet", str(val_file))
    print(f"✓ Downloaded: {train_file.name} ({train_file.stat().st_size / 1024:.1f} KB)")
    print(f"✓ Downloaded: {val_file.name} ({val_file.stat().st_size / 1024:.1f} KB)")

    train_df = pd.read_parquet(train_file)
    val_df = pd.read_parquet(val_file)
    print(f"Loaded Golden Train: {train_df.shape}")
    print(f"Loaded Golden Val:   {val_df.shape}")
except Exception as e:
    print(f"⚠️ S3 Pull failed: {e}")
    print("If bucket does not have golden split yet, wait for Teammate A to upload it.")

# ==============================================================================
# 3. Model Training & Inference (Teammate Prototyping Section)
# ==============================================================================
# Write your DeBERTa / Florence-2 / LightGBM pipeline here!
#
# Example:
# val_df["prediction"] = your_model.predict(val_df)


# ==============================================================================
# 4. Save & Validate Local Predictions
# ==============================================================================
def save_and_verify_predictions(
    val_dataframe: pd.DataFrame,
    pred_col: str = "prediction",
    id_col: str = "index",
    out_filename: str = "val_predictions.csv",
) -> None:
    """Save validation predictions and perform quick sanity checks."""
    out_path = Path("/kaggle/working") / out_filename
    sub = val_dataframe[[id_col, pred_col]].copy()

    # Check for NaNs
    nan_count = sub[pred_col].isna().sum()
    if nan_count > 0:
        print(f"⚠️ WARNING: Found {nan_count} NaNs in predictions! Filling with empty string.")
        sub[pred_col] = sub[pred_col].fillna("")

    sub.to_csv(out_path, index=False)
    print(f"🎉 Saved {len(sub)} predictions to {out_path}")
    print(sub.head(5))
