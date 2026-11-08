#!/usr/bin/env python3
"""Prepare a raw dataset with RTG labels, train/validation split, and manifest."""

import argparse

from udt.data import prepare_dataset_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument("data", help="Input raw dataset path")
    parser.add_argument("--horizon", type=int, required=True, help="Episode horizon")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--starts-at-reset", action="store_true")
    parser.add_argument("--timeout-semantics", default="unknown", choices=["unknown", "horizon"])
    parser.add_argument("--seed", type=int, default=0, help="Split seed")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--reward-scale", type=float, default=1000.0)
    parser.add_argument("--dataset-id", default="unspecified")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.horizon <= 0:
        raise ValueError(f"horizon must be positive, got {args.horizon}")
    manifest = prepare_dataset_file(
        args.data,
        args.output,
        horizon=args.horizon,
        starts_at_reset=args.starts_at_reset,
        timeout_semantics=args.timeout_semantics,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        reward_scale=args.reward_scale,
        dataset_id=args.dataset_id,
    )
    print(f"Prepared dataset written to {args.output}")
    print(f"  transitions: {manifest['retained_transitions']}")
    print(
        "  train/validation episodes: "
        f"{len(manifest['train_episode_ids'])}/{len(manifest['validation_episode_ids'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
