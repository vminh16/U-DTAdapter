#!/usr/bin/env python3
"""Run Baseline B0 (Frozen Decision Transformer, R off, U off) closed-loop rollout.

Performs:
1. Frozen DT parameter verification (all parameters requires_grad=False).
2. Deterministic B0 adapter parity check against direct model inference -> parity.json.
3. Multi-episode closed-loop evaluation on MuJoCo environment using evaluate_with_budget.
4. Telemetry and metric summary reporting (Mean, Std, Min, Max return).
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from udt.dt_adapter import DTAdapter
from udt.dt_integration import (
    assert_parameters_frozen,
    load_decision_transformer,
    verify_b0_parity,
)
from udt.evaluation import evaluate_with_budget


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Baseline B0 Decision Transformer")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/decision_transformer/model.pt",
        help="Path to trained Decision Transformer checkpoint",
    )
    parser.add_argument(
        "--env",
        default="Hopper-v4",
        help="Gymnasium MuJoCo environment ID (e.g. Hopper-v4, Hopper-v5)",
    )
    parser.add_argument(
        "--target-return",
        type=float,
        default=1800.0,
        help="Target unscaled return for B0 conditioning (e.g. 1800.0 or 3600.0)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1000,
        help="Episode horizon",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for evaluation",
    )
    parser.add_argument(
        "--parity-output",
        default="checkpoints/decision_transformer/parity.json",
        help="Output path for frozen-DT parity check report",
    )
    parser.add_argument(
        "--results-output",
        default="results/b0_evaluation.json",
        help="Output path for episode metrics JSON",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    # 1. Load checkpoint and construct model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}", file=sys.stderr)
        print("Please train the model first using `python scripts/train_dt.py`", file=sys.stderr)
        return 1

    print(f"Loading Decision Transformer from {ckpt_path}...")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_dim = int(payload.get("state_dim", 11))
    act_dim = int(payload.get("act_dim", 3))
    hidden_size = int(payload.get("hidden_size", 128))
    max_length = int(payload.get("max_length", 20))
    max_ep_len = int(payload.get("max_ep_len", args.horizon))
    n_layer = int(payload.get("n_layer", 3))
    n_head = int(payload.get("n_head", 1))
    state_mean = payload.get("state_mean")
    state_std = payload.get("state_std")
    reward_scale = float(payload.get("reward_scale", 1000.0))

    model, _ = load_decision_transformer(
        checkpoint_path=ckpt_path,
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=hidden_size,
        max_length=max_length,
        max_ep_len=max_ep_len,
        n_layer=n_layer,
        n_head=n_head,
        device=device,
    )

    # 2. Assert strictly frozen
    assert_parameters_frozen(model)
    print("✓ Model parameters verified frozen (requires_grad=False for all weights).")

    # 3. Create Gymnasium environment to query action bounds
    try:
        import gymnasium as gym
    except ImportError as exc:
        print("Error: `gymnasium` is not installed. Run `pip install gymnasium[mujoco]`", file=sys.stderr)
        return 1

    try:
        env = gym.make(args.env)
    except Exception as exc:
        print(f"Error initializing environment '{args.env}': {exc}", file=sys.stderr)
        print("Tip: ensure `mujoco` is installed: pip install mujoco imageio", file=sys.stderr)
        return 1

    action_low = env.action_space.low
    action_high = env.action_space.high

    # 4. Wrap with DTAdapter
    adapter = DTAdapter(
        model=model,
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=max_length,
        state_mean=state_mean,
        state_std=state_std,
        action_low=action_low,
        action_high=action_high,
        device=device,
    )

    # 5. Run Parity Verification
    print("Verifying B0 inference parity (DTAdapter vs Direct Model)...")
    parity_rng = np.random.default_rng(12345)
    test_obs = [parity_rng.normal(size=state_dim).astype(np.float32) for _ in range(30)]
    test_rtgs = [float(args.target_return / reward_scale) * (1.0 - t / 30.0) for t in range(30)]

    parity_report = verify_b0_parity(
        model=model,
        adapter=adapter,
        observations=test_obs,
        rtg_tokens=test_rtgs,
        atol=1e-6,
        model_source="real",
    )

    parity_file = Path(args.parity_output)
    parity_file.parent.mkdir(parents=True, exist_ok=True)
    parity_file.write_text(json.dumps(parity_report, indent=2))
    print(f"✓ Parity check: passed={parity_report['passed']} (max_diff={parity_report['max_abs_difference']:.2e}). Saved to {parity_file}")

    if not parity_report["passed"]:
        print("Warning: Parity check failed! Rollout results may be ungrounded.", file=sys.stderr)

    # 6. Run Closed-Loop B0 Evaluation Rollouts
    print(f"\nEvaluating Baseline B0 on {args.env} ({args.episodes} episodes, target return={args.target_return})...")
    episode_results = []
    episode_returns = []
    episode_lengths = []

    for ep in range(1, args.episodes + 1):
        ep_seed = args.seed + ep
        adapter.reset()

        rollout = evaluate_with_budget(
            env=env,
            policy_fn=adapter.policy_fn,
            capper=None,  # Baseline B0 has RTG Capper OFF
            initial_target=args.target_return,
            horizon=args.horizon,
            reward_scale=reward_scale,
            action_low=action_low,
            action_high=action_high,
            seed=ep_seed,
            policy_reset_fn=adapter.reset,
        )

        episode_returns.append(rollout.total_reward)
        episode_lengths.append(rollout.length)
        episode_results.append({
            "episode": ep,
            "seed": ep_seed,
            "total_reward": rollout.total_reward,
            "length": rollout.length,
            "terminated": rollout.terminated,
            "truncated": rollout.truncated,
            "final_raw_budget": rollout.final_raw_budget,
        })

        term_str = "Terminated (Early)" if rollout.terminated else ("Truncated (Horizon)" if rollout.truncated else "Done")
        print(f"  Episode {ep:3d}/{args.episodes} | Return: {rollout.total_reward:8.2f} | Length: {rollout.length:4d} | Status: {term_str}")

    env.close()

    # 7. Summary Metrics
    returns_arr = np.array(episode_returns)
    lengths_arr = np.array(episode_lengths)

    summary = {
        "variant": "B0",
        "env": args.env,
        "checkpoint": str(ckpt_path),
        "target_return": args.target_return,
        "reward_scale": reward_scale,
        "horizon": args.horizon,
        "episodes": args.episodes,
        "mean_return": float(returns_arr.mean()),
        "std_return": float(returns_arr.std()),
        "median_return": float(np.median(returns_arr)),
        "min_return": float(returns_arr.min()),
        "max_return": float(returns_arr.max()),
        "mean_length": float(lengths_arr.mean()),
        "parity": parity_report,
        "episodes_detail": episode_results,
    }

    results_file = Path(args.results_output)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 55)
    print("BASELINE B0 EVALUATION SUMMARY")
    print("=" * 55)
    print(f"Environment:       {args.env}")
    print(f"Target Return G_0: {args.target_return}")
    print(f"Episodes:          {args.episodes}")
    print(f"Mean Return:       {summary['mean_return']:.2f} ± {summary['std_return']:.2f}")
    print(f"Median Return:     {summary['median_return']:.2f} [Min: {summary['min_return']:.2f}, Max: {summary['max_return']:.2f}]")
    print(f"Mean Length:       {summary['mean_length']:.1f} steps")
    print(f"Results saved to:  {results_file}")
    print("=" * 55)

    return 0


if __name__ == "__main__":
    sys.exit(main())
