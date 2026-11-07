#!/usr/bin/env python3
"""Train baseline Decision Transformer on prepared offline trajectory dataset.

Reuses the official baseline SequenceTrainer via `udt.dt_integration.train_decision_transformer`.
"""

import argparse
from pathlib import Path
import sys
import time
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from udt.dt_integration import train_decision_transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline Decision Transformer reusing baseline SequenceTrainer"
    )
    parser.add_argument("--data", default="data/prepared/hopper-medium-v2/dataset.npz")
    parser.add_argument("--manifest", default="data/prepared/hopper-medium-v2/manifest.json")
    parser.add_argument("--output", default="checkpoints/decision_transformer/model.pt")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--n-head", type=int, default=1)
    parser.add_argument("--k", type=int, default=20, help="Context length K")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--num-steps", type=int, default=10000, help="Total gradient steps")
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Training Decision Transformer using baseline SequenceTrainer...")
    print(f"Data: {args.data} | Manifest: {args.manifest} | Output: {args.output}")
    print(f"Steps: {args.num_steps} | Batch: {args.batch_size} | K: {args.k} | Device: {args.device}")

    start_time = time.time()
    model, summary = train_decision_transformer(
        dataset=args.data,
        manifest=args.manifest,
        output_path=args.output,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        k=args.k,
        embed_dim=args.embed_dim,
        n_layer=args.n_layer,
        n_head=args.n_head,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        device=args.device,
        seed=args.seed,
    )
    elapsed = time.time() - start_time

    print(f"\n[DONE] Finished in {elapsed:.1f}s.")
    print(f"Train trajectories: {summary['num_train_trajectories']}")
    print(f"Val trajectories:   {summary['num_val_trajectories']}")
    if summary["best_val_loss"] < float("inf"):
        print(f"Best Val MSE:       {summary['best_val_loss']:.5f}")
    print(f"Checkpoint saved to: {args.output}")


if __name__ == "__main__":
    main()
