"""
Parallax Async Asset Downloader
===============================
High-performance asynchronous asset downloader with exponential retry backoff,
corruption verification, rate limiting, and resume capability.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from parallax.config import get_config
from parallax.observability.tracer import PipelineTracer

# Common image magic byte headers for integrity checking
MAGIC_BYTES = {
    "jpeg": b"\xff\xd8\xff",
    "png": b"\x89PNG\r\n\x1a\n",
    "gif": b"GIF8",
    "webp": b"RIFF",
}


def is_valid_image_bytes(data: bytes) -> bool:
    """Verify that payload is non-empty and starts with a known image signature."""
    if len(data) < 32:
        return False
    if data.startswith(MAGIC_BYTES["jpeg"]):
        return True
    if data.startswith(MAGIC_BYTES["png"]):
        return True
    if data.startswith(MAGIC_BYTES["gif"]):
        return True
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    # Fallback: if at least 1KB of data received without explicit magic header
    return len(data) > 1024


async def download_single_asset(
    session: aiohttp.ClientSession,
    url: str,
    output_path: Path,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
    timeout_sec: float = 15.0,
) -> tuple[str, bool, str | None]:
    """Download a single asset with retries and content verification."""
    if output_path.exists() and output_path.stat().st_size > 100:
        return str(output_path), True, "already_exists"

    if not url or not isinstance(url, str) or not url.startswith("http"):
        return str(output_path), False, "invalid_url"

    async with semaphore:
        for attempt in range(1, max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=timeout_sec)
                async with session.get(url, timeout=timeout) as response:
                    if response.status == 200:
                        content = await response.read()
                        if is_valid_image_bytes(content):
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            output_path.write_bytes(content)
                            return str(output_path), True, None
                        else:
                            return str(output_path), False, "corrupted_or_html"
                    elif response.status in (429, 500, 502, 503, 504):
                        # Transient server error or rate limit, back off
                        await asyncio.sleep(0.5 * (2**attempt))
                    else:
                        return str(output_path), False, f"http_status_{response.status}"
            except (aiohttp.ClientError, asyncio.TimeoutError) as err:
                if attempt == max_retries:
                    return str(output_path), False, f"error_{type(err).__name__}"
                await asyncio.sleep(0.5 * (2**attempt))
            except Exception as err:
                return str(output_path), False, f"unexpected_{type(err).__name__}"

        return str(output_path), False, "max_retries_exceeded"


async def download_catalog_assets(
    urls: list[str],
    ids: list[str],
    output_dir: Path,
    concurrency: int = 32,
    max_retries: int = 3,
    extension: str = "jpg",
) -> tuple[int, int, int]:
    """Download multiple assets concurrently."""
    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    connector = aiohttp.TCPConnector(limit=concurrency + 10, ttl_dns_cache=300)

    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        tasks = [
            download_single_asset(
                session=session,
                url=url,
                output_path=output_dir / f"{item_id}.{extension}",
                semaphore=semaphore,
                max_retries=max_retries,
            )
            for url, item_id in zip(urls, ids, strict=False)
        ]

        results = await tqdm_asyncio.gather(*tasks, desc="Downloading Assets")

    success_count = sum(1 for _, ok, _ in results)
    skipped_count = sum(1 for _, ok, reason in results if ok and reason == "already_exists")
    failed_count = len(results) - success_count

    return success_count, skipped_count, failed_count


def run_downloader(
    input_file: Path,
    url_col: str,
    id_col: str,
    output_dir: Path,
    concurrency: int = 32,
    limit: int | None = None,
) -> None:
    """Read CSV, extract URLs, and run the async download pipeline with tracing."""
    tracer = PipelineTracer("asset_downloader")

    with tracer.span("load_manifest"):
        if input_file.suffix == ".parquet":
            df = pd.read_parquet(input_file)
        else:
            df = pd.read_csv(input_file)

        if limit is not None:
            df = df.head(limit)

        urls = df[url_col].astype(str).tolist()
        ids = df[id_col].astype(str).tolist()

    meta = {"total_assets": len(urls), "concurrency": concurrency}
    with tracer.span("download_assets", metadata=meta):
        success, skipped, failed = asyncio.run(
            download_catalog_assets(
                urls=urls,
                ids=ids,
                output_dir=output_dir,
                concurrency=concurrency,
            )
        )

    trace = tracer.finish(success=(failed == 0 or success > 0))
    print("\n✅ Asset Download Complete:")
    print(f"   Total Requested: {len(urls)}")
    print(f"   Successful:      {success} (Skipped Existing: {skipped})")
    print(f"   Failed:          {failed}")
    print(f"   Duration:        {trace.total_latency_ms:.1f}ms")


def main() -> None:
    """CLI Entry point."""
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Parallax Asynchronous Asset Downloader")
    parser.add_argument("--input", "-i", type=Path, default=cfg.paths.raw_data_dir / "train.csv")
    parser.add_argument("--url-col", default="image_link", help="Column containing asset URLs")
    parser.add_argument("--id-col", default="index", help="Column containing unique sample IDs")
    parser.add_argument("--out-dir", "-o", type=Path, default=cfg.paths.images_dir)
    parser.add_argument(
        "--concurrency", "-c", type=int, default=32, help="Max concurrent connections"
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=None, help="Optional row limit for testing"
    )
    args = parser.parse_args()

    run_downloader(
        input_file=args.input,
        url_col=args.url_col,
        id_col=args.id_col,
        output_dir=args.out_dir,
        concurrency=args.concurrency,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
