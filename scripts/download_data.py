#!/usr/bin/env python3
"""Download an official D4RL dataset in raw HDF5 format."""

import argparse
from pathlib import Path

from udt.download import D4RL_URLS, download_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download an official D4RL dataset")
    parser.add_argument("dataset", choices=sorted(D4RL_URLS), help="Dataset name")
    parser.add_argument("--target-dir", default="data/raw", help="Target raw directory")
    parser.add_argument("--force", action="store_true", help="Re-download an existing file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = Path(args.target_dir)
    filename = D4RL_URLS[args.dataset].split("/")[-1]
    destination = download_dataset(
        name=args.dataset,
        target_dir=target,
        force=args.force,
        provenance_path=target / f"{filename}.provenance.json",
    )
    print(f"Downloaded {args.dataset} to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
