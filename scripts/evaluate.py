#!/usr/bin/env python3
"""Evaluate one U-DT variant (B0/B3/B4/B5) through the shared rollout path."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from udt.dt_integration import load_dt_checkpoint, verify_b0_checkpoint_parity
from udt.experiment import build_evaluation_report, evaluate_variant
from udt.readiness import SUPPORTED_VARIANTS
from udt.return_cap import load_return_capper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a U-DT variant")
    parser.add_argument("--variant", required=True, choices=SUPPORTED_VARIANTS)
    parser.add_argument(
        "--checkpoint", default="checkpoints/decision_transformer/model.pt"
    )
    parser.add_argument("--env", default="Hopper-v4")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--target-return", type=float, default=1800.0)
    parser.add_argument("--reward-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="data/prepared/hopper-medium-v2/dataset.npz")
    parser.add_argument("--capper", help="Module R return-capper checkpoint (B3/B5)")
    parser.add_argument("--return-model", help="g_psi checkpoint (B4/B5); defaults to --capper")
    parser.add_argument("--dynamics", help="Module U dynamics ensemble checkpoint (B4/B5)")
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--parity-output", default="results/parity.json")
    parser.add_argument("--results-output", default="results/evaluation.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-unsafe-legacy", action="store_true")
    parser.add_argument("--collect-diagnostics", action="store_true")
    parser.add_argument("--rtg-alpha", type=float, default=1.0,
                        help="RTG intervention strength in [0,1]. 0=no cap, 1=hard cap (default)")
    return parser.parse_args()


def _write_new_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError(f"episodes must be positive, got {args.episodes}")
    if args.horizon <= 0:
        raise ValueError(f"horizon must be positive, got {args.horizon}")
    if not np.isfinite(args.rtg_alpha) or args.rtg_alpha < 0 or args.rtg_alpha > 1:
        raise ValueError(f"--rtg-alpha must be in [0, 1], got {args.rtg_alpha}")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Decision Transformer checkpoint not found: {checkpoint}")

    adapter, payload = load_dt_checkpoint(
        checkpoint,
        horizon=args.horizon,
        device=args.device,
        allow_unsafe_legacy=args.allow_unsafe_legacy,
    )
    reward_scale = (
        float(args.reward_scale)
        if args.reward_scale is not None
        else float(payload.get("reward_scale", 1000.0))
    )

    import gymnasium as gym

    try:
        env = gym.make(args.env)
    except Exception as exc:
        raise RuntimeError(
            f"Could not create environment {args.env!r}: {exc}. "
            "Install gymnasium[mujoco] and mujoco."
        ) from exc

    try:
        if env.observation_space.shape != (adapter.state_dim,):
            raise ValueError(
                f"Environment observation shape {env.observation_space.shape} does not "
                f"match checkpoint state_dim {adapter.state_dim}"
            )
        if env.action_space.shape != (adapter.act_dim,):
            raise ValueError(
                f"Environment action shape {env.action_space.shape} does not match "
                f"checkpoint act_dim {adapter.act_dim}"
            )
        adapter.action_low = np.asarray(env.action_space.low, dtype=np.float32)
        adapter.action_high = np.asarray(env.action_space.high, dtype=np.float32)

        parity = verify_b0_checkpoint_parity(
            adapter,
            target_return=args.target_return,
            reward_scale=reward_scale,
            max_ep_len=int(payload.get("max_ep_len", args.horizon)),
        )
        if not parity["passed"]:
            raise RuntimeError(f"Frozen-DT parity failed: {parity}")
        _write_new_json(Path(args.parity_output), parity)

        capper = return_model = ensemble = None
        if args.capper:
            capper, _ = load_return_capper(
                args.capper,
                device=args.device,
                expected_horizon=args.horizon,
                expected_scale=reward_scale,
                allow_unsafe_legacy=args.allow_unsafe_legacy,
            )
        if args.return_model:
            return_model, _ = load_return_capper(
                args.return_model,
                device=args.device,
                expected_horizon=args.horizon,
                expected_scale=reward_scale,
                allow_unsafe_legacy=args.allow_unsafe_legacy,
            )
        elif capper is not None:
            return_model = capper
        if args.dynamics:
            from udt.dynamics import load_dynamics_checkpoint

            ensemble, _ = load_dynamics_checkpoint(
                args.dynamics,
                device=args.device,
                expected_horizon=args.horizon,
                expected_reward_scale=reward_scale,
                allow_unsafe_legacy=args.allow_unsafe_legacy,
            )

        summary = evaluate_variant(
            args.variant,
            env,
            adapter,
            episodes=args.episodes,
            seed=args.seed,
            capper=capper,
            ensemble=ensemble,
            return_model=return_model,
            num_candidates=args.num_candidates,
            radius=args.radius,
            beta=args.beta,
            horizon=args.horizon,
            initial_target=args.target_return,
            reward_scale=reward_scale,
            state_std=adapter.state_std,
            action_low=adapter.action_low,
            action_high=adapter.action_high,
            collect_diagnostics=args.collect_diagnostics,
            rtg_alpha=args.rtg_alpha,
        )
    finally:
        env.close()

    dataset_path = Path(args.dataset)
    manifest_path = dataset_path.parent / "manifest.json"
    dataset_manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    report = build_evaluation_report(
        variant=args.variant,
        env_id=args.env,
        summary=summary,
        config={
            "episodes": args.episodes,
            "horizon": args.horizon,
            "target_return": args.target_return,
            "num_candidates": args.num_candidates,
            "radius": args.radius,
            "beta": args.beta,
            "rtg_alpha": args.rtg_alpha,
        },
        checkpoints={
            "decision_transformer": checkpoint,
            "capper": args.capper,
            "return_model": args.return_model or args.capper,
            "dynamics": args.dynamics,
        },
        dataset_path=dataset_path,
        dataset_manifest=dataset_manifest,
        parity=parity,
        device=args.device,
    )
    _write_new_json(Path(args.results_output), report)
    print(
        f"{args.variant}: mean_return={report['mean_return']:.2f} "
        f"± {report['std_return']:.2f} over {args.episodes} episodes"
    )
    print(f"Results saved to {args.results_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
