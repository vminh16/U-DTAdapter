#!/usr/bin/env python3
"""Train the Module R expectile return capper on a prepared dataset."""

import argparse
from pathlib import Path

import torch

from udt.data import load_prepared_dataset, verify_manifest_normalization
from udt.provenance import dataset_provenance
from udt.return_cap import save_return_capper, train_return_cap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Module R return capper")
    parser.add_argument("--data", required=True, help="Prepared dataset.npz path")
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--output", default="checkpoints/module_r", help="Output directory")
    parser.add_argument(
        "--mode", default="state_time", choices=["state_time", "time_only", "state_only", "global"]
    )
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--reward-scale", type=float, default=1000.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.horizon <= 0:
        raise ValueError(f"horizon must be positive, got {args.horizon}")
    if not 0.0 < args.tau < 1.0:
        raise ValueError(f"tau must lie in (0, 1), got {args.tau}")

    dataset, manifest = load_prepared_dataset(
        args.data, horizon=args.horizon, reward_scale=args.reward_scale
    )
    normalization = None
    if manifest.get("state_mean") is not None and manifest.get("state_std") is not None:
        normalization = verify_manifest_normalization(dataset, manifest)

    capper, metrics = train_return_cap(
        dataset=dataset,
        horizon=args.horizon,
        tau=args.tau,
        mode=args.mode,
        margin=args.margin,
        scale=args.reward_scale,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )
    if normalization is not None:
        metrics["normalization_provenance"] = normalization

    checkpoint = Path(args.output) / "return_capper.pt"
    save_return_capper(
        checkpoint,
        capper,
        metrics=metrics,
        seed=args.seed,
        tau=args.tau,
        training_config={"mode": args.mode, "epochs": args.epochs, "lr": args.lr},
        dataset_provenance=dataset_provenance(args.data, manifest),
    )
    print(f"Return capper saved to {checkpoint}")
    print(f"  best_epoch: {metrics['best_epoch']}")
    print(f"  best_val_loss: {metrics['best_val_loss']:.6f}")
    print(f"  best_val_coverage: {metrics['best_val_coverage']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
