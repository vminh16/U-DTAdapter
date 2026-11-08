#!/usr/bin/env python3
"""Train the Module U dynamics ensemble on a prepared dataset."""

import argparse
from pathlib import Path

import numpy as np
import torch

from udt.data import load_prepared_dataset, verify_manifest_normalization
from udt.dynamics import save_dynamics_checkpoint, train_dynamics_ensemble
from udt.provenance import dataset_provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Module U dynamics ensemble")
    parser.add_argument("--data", required=True, help="Prepared dataset.npz path")
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--output", default="checkpoints/module_u", help="Output directory")
    parser.add_argument("--num-models", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--reward-scale", type=float, default=1000.0)
    parser.add_argument("--w-reward", type=float, default=1.0)
    parser.add_argument("--w-terminal", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.horizon <= 0:
        raise ValueError(f"horizon must be positive, got {args.horizon}")
    if args.num_models < 2:
        raise ValueError("num_models must be at least 2 for U disagreement")

    dataset, manifest = load_prepared_dataset(
        args.data, horizon=args.horizon, reward_scale=args.reward_scale
    )
    if manifest.get("state_mean") is None or manifest.get("state_std") is None:
        raise ValueError("Manifest must provide state_mean/state_std for dynamics training")
    normalization = verify_manifest_normalization(dataset, manifest)

    ensemble, metrics = train_dynamics_ensemble(
        dataset=dataset,
        horizon=args.horizon,
        num_models=args.num_models,
        state_mean=np.asarray(manifest["state_mean"], dtype=np.float32),
        state_std=np.asarray(manifest["state_std"], dtype=np.float32),
        scale=args.reward_scale,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        w_reward=args.w_reward,
        w_terminal=args.w_terminal,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
        normalization_provenance=normalization,
    )

    checkpoint = save_dynamics_checkpoint(
        Path(args.output) / "dynamics_ensemble.pt",
        ensemble,
        metrics=metrics,
        dataset_provenance=dataset_provenance(args.data, manifest),
    )
    print(f"Dynamics ensemble saved to {checkpoint}")
    print(
        "  bootstrap transitions per head: "
        + ", ".join(str(m["bootstrap"]["bootstrap_transitions"]) for m in metrics["models"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
