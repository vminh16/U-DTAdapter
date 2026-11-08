#!/usr/bin/env python3
"""Audit pilot artifacts for variant-scoped readiness."""

import argparse
import json
from pathlib import Path

from udt.data import load_prepared_dataset
from udt.readiness import (
    SUPPORTED_VARIANTS,
    check_pilot_readiness,
    format_readiness_report,
    readiness_json,
)
from udt.return_cap import evaluate_return_capper, load_return_capper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit pilot artifact readiness")
    parser.add_argument("--manifest", help="Prepared dataset manifest.json")
    parser.add_argument("--dataset", help="Prepared dataset.npz for hash verification")
    parser.add_argument("--dynamics-checkpoint", help="Dynamics ensemble checkpoint (.pt)")
    parser.add_argument("--return-checkpoint", help="Return capper/model checkpoint")
    parser.add_argument("--parity", help="Frozen-DT parity JSON report")
    parser.add_argument(
        "--variants",
        default="B0,B3,B4,B5",
        help="Comma-separated variants required for the pilot",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--output", help="Optional output report path")
    parser.add_argument("--allow-unsafe-legacy", action="store_true")
    parser.add_argument(
        "--recompute-return-metrics",
        action="store_true",
        help="Re-evaluate the restored return checkpoint on the dataset validation split",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variants = tuple(part.strip() for part in args.variants.split(",") if part.strip())
    unknown = [variant for variant in variants if variant not in SUPPORTED_VARIANTS]
    if unknown or not variants:
        raise ValueError(
            f"variants must be a non-empty subset of {SUPPORTED_VARIANTS}, got {variants}"
        )

    manifest = json.loads(Path(args.manifest).read_text()) if args.manifest else None
    parity = json.loads(Path(args.parity).read_text()) if args.parity else None

    return_payload = None
    if args.recompute_return_metrics:
        if not args.return_checkpoint or not args.dataset:
            raise ValueError(
                "--recompute-return-metrics requires --return-checkpoint and --dataset"
            )
        capper, payload = load_return_capper(
            args.return_checkpoint, allow_unsafe_legacy=args.allow_unsafe_legacy
        )
        dataset, _ = load_prepared_dataset(args.dataset, horizon=capper.horizon)
        evaluation = evaluate_return_capper(
            dataset,
            capper,
            horizon=capper.horizon,
            tau=float(payload.get("tau") or 0.9),
        )
        return_payload = dict(payload)
        metrics = dict(payload.get("metrics") or {})
        metrics.setdefault("best_val_loss", evaluation["loss"])
        metrics["best_val_coverage"] = evaluation["coverage"]
        metrics["restored_checkpoint_evaluation"] = evaluation
        return_payload["metrics"] = metrics

    report = check_pilot_readiness(
        manifest=manifest,
        dataset_path=args.dataset,
        dynamics_checkpoint=args.dynamics_checkpoint,
        return_checkpoint=args.return_checkpoint,
        return_payload=return_payload,
        dt_parity=parity,
        required_variants=variants,
        allow_unsafe_legacy=args.allow_unsafe_legacy,
    )

    output = readiness_json(report) if args.json else format_readiness_report(report)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output)
        print(f"Readiness report saved to {path}")
    else:
        print(output)
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
