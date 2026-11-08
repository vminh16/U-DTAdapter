#!/usr/bin/env python3
"""Audit a raw dataset for episode boundaries, continuity, and provenance."""

import argparse
import json
from pathlib import Path

from udt.data import audit_dataset_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit raw dataset integrity")
    parser.add_argument("data", help="Input .hdf5, .h5, or .npz path")
    parser.add_argument("--horizon", type=int, required=True, help="Episode horizon")
    parser.add_argument("--starts-at-reset", action="store_true")
    parser.add_argument("--timeout-semantics", default="unknown", choices=["unknown", "horizon"])
    parser.add_argument("--dataset-id", default="unspecified")
    parser.add_argument("--output", help="Optional output JSON report path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.horizon <= 0:
        raise ValueError(f"horizon must be positive, got {args.horizon}")
    report = audit_dataset_file(
        args.data,
        horizon=args.horizon,
        starts_at_reset=args.starts_at_reset,
        timeout_semantics=args.timeout_semantics,
        dataset_id=args.dataset_id,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
        print(f"Audit report saved to {output}")
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
