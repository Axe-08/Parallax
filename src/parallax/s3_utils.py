"""
Parallax S3 Transfer Utilities
==============================
Robust S3 asset downloader and uploader with progress tracking,
automatic credential discovery, and resume capability.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import boto3
from tqdm import tqdm

DEFAULT_BUCKET = "amazon-ml-challange-2026-parallax"
DEFAULT_REGION = "us-east-1"


def get_s3_credentials() -> tuple[str, str, str, str]:
    """
    Retrieve AWS credentials from environment, default downloads file, or user profile.
    Returns (access_key_id, secret_access_key, region, bucket).
    """
    ak = os.getenv("AWS_ACCESS_KEY_ID", "")
    sk = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    region = os.getenv("AWS_DEFAULT_REGION", DEFAULT_REGION)
    bucket = os.getenv("PARALLAX_S3_BUCKET", DEFAULT_BUCKET)

    if ak and sk:
        return ak, sk, region, bucket

    creds_csv = Path("/home/akshit/Downloads/ml-team-storage-user_accessKeys.csv")
    if creds_csv.exists():
        try:
            with open(creds_csv, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    row = rows[0]
                    ak = (
                        row.get("Access key ID")
                        or row.get("\ufeffAccess key ID")
                        or row.get("AccessKeyId")
                        or ""
                    )
                    sk = row.get("Secret access key") or row.get("SecretAccessKey") or ""
                    if ak and sk:
                        return ak.strip(), sk.strip(), region, bucket
        except Exception:
            pass

    return ak, sk, region, bucket


def get_s3_client(
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    region: str | None = None,
) -> Any:
    """Instantiate a boto3 S3 client."""
    ak, sk, reg, _ = get_s3_credentials()
    final_ak = access_key_id or ak
    final_sk = secret_access_key or sk
    final_reg = region or reg

    if final_ak and final_sk:
        return boto3.client(
            "s3",
            aws_access_key_id=final_ak,
            aws_secret_access_key=final_sk,
            region_name=final_reg,
        )
    return boto3.client("s3", region_name=final_reg)


def download_s3_file(
    s3_key: str,
    local_path: Path | str,
    bucket: str | None = None,
    client: Any | None = None,
    overwrite: bool = False,
) -> Path:
    """Download an S3 object with progress bar and integrity check."""
    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _, _, _, default_bucket = get_s3_credentials()
    b = bucket or default_bucket
    s3 = client or get_s3_client()

    meta = s3.head_object(Bucket=b, Key=s3_key)
    total_bytes = meta["ContentLength"]

    if path.exists() and not overwrite and path.stat().st_size == total_bytes:
        print(f"  ⚡ Already cached locally: {path.name} ({total_bytes / 1e6:.1f} MB)")
        return path

    print(f"  📥 Downloading s3://{b}/{s3_key} ({total_bytes / 1e6:.1f} MB)...")
    with tqdm(total=total_bytes, unit="B", unit_scale=True, desc=f"  ↳ {path.name}") as pbar:

        def callback(bytes_transferred: int) -> None:
            pbar.update(bytes_transferred)

        s3.download_file(b, s3_key, str(path), Callback=callback)

    return path


def upload_s3_file(
    local_path: Path | str,
    s3_key: str,
    bucket: str | None = None,
    client: Any | None = None,
) -> str:
    """Upload a local file to S3 with progress bar."""
    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"Local file does not exist: {path}")

    _, _, _, default_bucket = get_s3_credentials()
    b = bucket or default_bucket
    s3 = client or get_s3_client()

    total_bytes = path.stat().st_size
    print(f"  📤 Uploading {path.name} ({total_bytes / 1e6:.1f} MB) -> s3://{b}/{s3_key}...")
    with tqdm(total=total_bytes, unit="B", unit_scale=True, desc=f"  ↳ {path.name}") as pbar:

        def callback(bytes_transferred: int) -> None:
            pbar.update(bytes_transferred)

        s3.upload_file(str(path), b, s3_key, Callback=callback)

    uri = f"s3://{b}/{s3_key}"
    print(f"  ✓ Successfully uploaded: {uri}")
    return uri
