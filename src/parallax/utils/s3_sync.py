"""
Parallax S3 Cloud Sync Utility
==============================
Synchronizes Golden Splits, model checkpoints, and processed datasets
with the team's central AWS S3 bucket.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from parallax.config import get_config


def get_s3_client(region: str | None = None) -> Any:
    """Create a configured boto3 S3 client."""
    cfg = get_config()
    reg = region or cfg.s3.region
    session = boto3.Session(region_name=reg)
    return session.client("s3")


def upload_path_to_s3(
    local_path: Path,
    s3_prefix: str | None = None,
    bucket_name: str | None = None,
) -> bool:
    """Upload a file or directory recursively to S3."""
    cfg = get_config()
    bucket = bucket_name or cfg.s3.bucket_name
    s3 = get_s3_client()

    if not local_path.exists():
        print(f"❌ Error: Local path does not exist: {local_path}", file=sys.stderr)
        return False

    prefix = s3_prefix or local_path.name
    files_to_upload: list[Path] = []
    if local_path.is_file():
        files_to_upload.append(local_path)
    else:
        files_to_upload = [p for p in local_path.rglob("*") if p.is_file()]

    print(f"==> Uploading {len(files_to_upload)} file(s) to s3://{bucket}/{prefix}...")
    for f in files_to_upload:
        if local_path.is_file():
            key = prefix
        else:
            rel = f.relative_to(local_path)
            key = f"{prefix}/{rel.as_posix()}"

        try:
            s3.upload_file(str(f), bucket, key)
            print(f"  ✓ Uploaded: {key}")
        except (NoCredentialsError, ClientError) as exc:
            print(f"❌ S3 Upload failed: {exc}", file=sys.stderr)
            return False

    print("🎉 S3 Upload complete!")
    return True


def download_prefix_from_s3(
    s3_prefix: str,
    local_dest: Path,
    bucket_name: str | None = None,
) -> bool:
    """Download all objects matching an S3 prefix to local directory."""
    cfg = get_config()
    bucket = bucket_name or cfg.s3.bucket_name
    s3 = get_s3_client()

    local_dest.mkdir(parents=True, exist_ok=True)
    print(f"==> Pulling s3://{bucket}/{s3_prefix} to {local_dest}...")

    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix)

        found = False
        for page in pages:
            for obj in page.get("Contents", []):
                found = True
                key = obj["Key"]
                rel_path = key[len(s3_prefix) :].lstrip("/")
                dest_file = local_dest / rel_path if rel_path else local_dest / Path(key).name
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(dest_file))
                print(f"  ✓ Downloaded: {dest_file.name}")

        if not found:
            print(f"⚠️ Warning: No objects found matching prefix: {s3_prefix}")
            return False

        print("🎉 S3 Download complete!")
        return True

    except (NoCredentialsError, ClientError) as exc:
        print(f"❌ S3 Download failed: {exc}", file=sys.stderr)
        return False


def list_s3_contents(bucket_name: str | None = None, prefix: str = "") -> None:
    """List objects in the team S3 bucket."""
    cfg = get_config()
    bucket = bucket_name or cfg.s3.bucket_name
    s3 = get_s3_client()

    print(f"==> Contents of s3://{bucket}/{prefix}:")
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        contents = response.get("Contents", [])
        if not contents:
            print("   (Bucket is empty or prefix has no objects)")
            return

        for item in contents:
            size_mb = item["Size"] / (1024 * 1024)
            print(f"   {item['LastModified']} | {size_mb:>8.2f} MB | {item['Key']}")
    except (NoCredentialsError, ClientError) as exc:
        print(f"❌ S3 list failed: {exc}", file=sys.stderr)


def main() -> None:
    """CLI Entry point."""
    parser = argparse.ArgumentParser(description="Parallax S3 Sync Utility")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # push
    p_push = subparsers.add_parser("push", help="Upload local file or directory to S3")
    p_push.add_argument("path", type=Path, help="Local file or directory")
    p_push.add_argument("--prefix", "-p", default=None, help="S3 target prefix")
    p_push.add_argument("--bucket", "-b", default=None, help="S3 bucket name")

    # pull
    p_pull = subparsers.add_parser("pull", help="Download from S3 to local directory")
    p_pull.add_argument("prefix", help="S3 source prefix")
    p_pull.add_argument("dest", type=Path, help="Local destination directory")
    p_pull.add_argument("--bucket", "-b", default=None, help="S3 bucket name")

    # list
    p_list = subparsers.add_parser("list", help="List S3 objects in team bucket")
    p_list.add_argument("--prefix", "-p", default="", help="Optional prefix filter")
    p_list.add_argument("--bucket", "-b", default=None, help="S3 bucket name")

    args = parser.parse_args()

    if args.subcommand == "push":
        upload_path_to_s3(args.path, s3_prefix=args.prefix, bucket_name=args.bucket)
    elif args.subcommand == "pull":
        download_prefix_from_s3(args.prefix, local_dest=args.dest, bucket_name=args.bucket)
    elif args.subcommand == "list":
        list_s3_contents(bucket_name=args.bucket, prefix=args.prefix)


if __name__ == "__main__":
    main()
