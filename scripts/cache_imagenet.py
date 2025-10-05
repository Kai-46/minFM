#!/usr/bin/env python3
"""
ImageNet dataset download utility.

All data is stored under $MINFM_DATA_DIR/imagenet/

Requirements:
- MINFM_DATA_DIR environment variable set
- For mode=hf: Hugging Face account with access to ILSVRC/imagenet-1k, and HF_TOKEN
- For mode=direct: A base URL that serves files like
  train_images_0.tar.gz ... train_images_4.tar.gz, val_images.tar.gz, test_images.tar.gz
"""

import argparse
from collections.abc import Iterable
import logging
import os
from pathlib import Path
import sys
import tarfile
from urllib import error, request


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def get_data_directory() -> Path:
    """Get the data directory from environment variable."""
    data_dir = os.getenv("MINFM_DATA_DIR")
    if not data_dir:
        raise ValueError(
            "MINFM_DATA_DIR environment variable is not set. " "Please set it to specify where to download the data."
        )

    imagenet_dir = Path(data_dir) / "imagenet"
    imagenet_dir.mkdir(parents=True, exist_ok=True)
    return imagenet_dir


def _format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.2f} {units[idx]}"


def _get_remote_size(url: str, headers: dict | None = None) -> int | None:
    try:
        req = request.Request(url, method="HEAD", headers=headers or {})
        with request.urlopen(req) as resp:
            length = resp.headers.get("Content-Length")
            if length is None:
                return None
            return int(length)
    except Exception:
        return None


def _stream_download(url: str, destination: Path, headers: dict | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    remote_size = _get_remote_size(url, headers=headers)
    if destination.exists() and remote_size is not None:
        local_size = destination.stat().st_size
        if local_size == remote_size:
            logging.info(f"Already downloaded: {destination.name} ({_format_size(local_size)})")
            return

    logging.info(f"Downloading {url} -> {destination}")
    try:
        req = request.Request(url, headers=headers or {})
        with request.urlopen(req) as resp, open(destination, "wb") as out_f:
            total = resp.headers.get("Content-Length")
            total_bytes = int(total) if total is not None else None
            bytes_read = 0
            next_log_at = 0
            chunk_size = 1024 * 1024
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
                bytes_read += len(chunk)
                if total_bytes:
                    percent = int(bytes_read * 100 / total_bytes)
                    if percent >= next_log_at:
                        logging.info(
                            f"  Progress {percent}% ({_format_size(bytes_read)} / {_format_size(total_bytes)})"
                        )
                        # Log every 5%
                        next_log_at = ((percent // 5) + 1) * 5
            if total_bytes and bytes_read != total_bytes:
                raise OSError(f"Incomplete download for {destination.name}: got {bytes_read}, expected {total_bytes}")
    except error.HTTPError as e:
        logging.error(f"HTTP error while downloading {url}: {e}")
        raise
    except error.URLError as e:
        logging.error(f"URL error while downloading {url}: {e}")
        raise


def _extract_tar_gz(archive_path: Path, extract_dir: Path) -> None:
    logging.info(f"Extracting {archive_path.name} -> {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as tar:
        tar.extractall(path=extract_dir)


def _resolve_base_url(cli_base_url: str | None) -> str:
    base_url = cli_base_url or os.getenv("IMAGENET_BASE_URL")
    if not base_url:
        raise ValueError("Base URL is required for direct mode. Provide --base-url or set IMAGENET_BASE_URL.")
    return base_url.rstrip("/")


def _files_for_splits(splits: Iterable[str]) -> Iterable[tuple[str, str]]:
    """Yield (filename, split_name)."""
    normalized = {s.lower() for s in splits}
    if not normalized:
        normalized = {"train", "val", "test"}
    if "train" in normalized:
        for idx in range(5):
            yield (f"train_images_{idx}.tar.gz", "train")
    if "val" in normalized or "validation" in normalized:
        yield ("val_images.tar.gz", "val")
    if "test" in normalized:
        yield ("test_images.tar.gz", "test")


def _build_headers_for_base_url(base_url: str) -> dict:
    headers: dict = {"User-Agent": "minFM-imagenet-downloader/1.0"}
    if "huggingface.co" in base_url:
        assert (
            "HF_TOKEN" in os.environ
        ), "HF_TOKEN is not set, please check this: https://huggingface.co/settings/tokens"
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        else:
            logging.warning("Base URL points to huggingface.co but HF_TOKEN is not set; requests may be unauthorized.")
    return headers


def download_imagenet_core_direct(
    data_dir: Path,
    base_url: str,
    splits: Iterable[str],
    extract: bool,
    keep_archives: bool,
    headers: dict | None = None,
) -> None:
    archives_dir = data_dir / "archives"
    extracts_dir = data_dir / "extracted"
    archives_dir.mkdir(parents=True, exist_ok=True)
    for filename, split_name in _files_for_splits(splits):
        url = f"{base_url}/{filename}"
        destination = archives_dir / filename
        _stream_download(url, destination, headers=headers)
        if extract:
            split_extract_dir = extracts_dir / split_name
            _extract_tar_gz(destination, split_extract_dir)
            if not keep_archives:
                try:
                    destination.unlink()
                except Exception:
                    # Non-fatal if cleanup fails
                    pass


def download_imagenet() -> None:
    """Main function."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Download ImageNet data")
    parser.add_argument(
        "--mode",
        choices=["direct", "hf"],
        default="direct",
        help="Download mode: 'direct' for tar.gz archives, 'hf' for Hugging Face dataset",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        # Newer SHAs use parquet. We need the .tar.gz format dastaset.
        default="https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/1500f8c59b214ce459c0a593fa1c87993aeb7700/data/",
        help="Base URL hosting the tar.gz archives (required for --mode direct). Can also set IMAGENET_BASE_URL.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Splits to download for direct mode. Any of: train val test",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract downloaded tar.gz archives",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep tar.gz archives after extraction",
    )

    args = parser.parse_args()

    try:
        # Get data directory
        data_dir = get_data_directory()
        logging.info(f"Data directory: {data_dir}")
        base_url = _resolve_base_url(args.base_url)
        headers = _build_headers_for_base_url(base_url)
        download_imagenet_core_direct(
            data_dir=data_dir,
            base_url=base_url,
            splits=args.splits,
            extract=args.extract,
            keep_archives=args.keep_archives,
            headers=headers,
        )

        logging.info("ImageNet download completed successfully!")

    except KeyboardInterrupt:
        logging.info("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)


def download_metadata() -> None:
    remote_url = "https://huggingface.co/datasets/Kai-46/minFM/resolve/main/ilsvrc2012_meta.pt"
    save_path = Path(get_data_directory()) / "ilsvrc2012_meta.pt"
    import subprocess

    subprocess.run(["wget", remote_url, "-O", save_path])


if __name__ == "__main__":
    download_imagenet()
    download_metadata()
