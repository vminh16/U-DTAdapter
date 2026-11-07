#!/usr/bin/env python3
"""UDT-Adapter: Unified Runner for Offline RL Intervention with Frozen Decision Transformers.

Provides transparent CLI entrypoints:
- download: Fetch official D4RL benchmarks in raw HDF5 format.
- demo-data: Create synthetic transitions for pipeline verification.
- audit: Audit transition dataset integrity and boundaries.
- prepare: Generate complete trajectory dataset with RTG labels and split.
- train-r: Train Module R expectile return capper (B2 global, B3 state-time, B9 time-only).
- verify-r: Run rigorous mathematical, numerical, and edge-case verifications for Module R.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

from udt.data import audit_dataset, load_dataset, prepare_dataset
from udt.download import D4RL_URLS, download_dataset
from udt.intervention import ReturnBudget
from udt.return_cap import (
    ExpectileLoss,
    GlobalReturnCap,
    ReturnCapMLP,
    ReturnCapper,
    compute_global_expectile,
    train_return_cap,
)


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# Subcommand Handlers
# -----------------------------------------------------------------------------

def cmd_download(args: argparse.Namespace) -> int:
    """Download an official D4RL dataset."""
    dest = download_dataset(
        name=args.dataset,
        target_dir=args.target_dir,
        force=args.force,
    )
    print(f"Successfully downloaded {args.dataset} to {dest} ({dest.stat().st_size:,} bytes)")
    return 0


def cmd_demo_data(args: argparse.Namespace) -> int:
    """Generate a synthetic test dataset."""
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 4 complete episodes of length 3, plus an unclosed suffix
    obs = np.array([
        [0.0], [1.0], [2.0],
        [10.0], [11.0], [12.0],
        [20.0], [21.0], [22.0],
        [30.0], [31.0], [32.0],
        [40.0],
    ], dtype=np.float32)
    next_obs = np.array([
        [1.0], [2.0], [3.0],
        [11.0], [12.0], [13.0],
        [21.0], [22.0], [23.0],
        [31.0], [32.0], [33.0],
        [41.0],
    ], dtype=np.float32)
    actions = np.zeros((len(obs), 1), dtype=np.float32)
    rewards = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 5.0], dtype=np.float32)
    terminals = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int32)
    timeouts = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], dtype=np.int32)

    data = {
        "observations": obs,
        "next_observations": next_obs,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
    }
    np.savez(out_path, **data)
    print(f"Demo dataset written to {out_path} ({len(obs)} transitions)")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Audit transition dataset integrity."""
    in_path = Path(args.data)
    data = load_dataset(in_path)

    report = audit_dataset(
        data=data,
        horizon=args.horizon,
        starts_at_reset=args.starts_at_reset,
        timeout_semantics=args.timeout_semantics,
    )

    report["provenance"] = {
        "dataset_id": getattr(args, "dataset_id", None) or "unspecified",
        "source_sha256": compute_file_sha256(in_path),
    }

    out_path = Path(args.output) if args.output else None
    rep_json = json.dumps(report, indent=2)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rep_json)
        print(f"Audit report saved to {out_path}")
    else:
        print(rep_json)

    print(f"Audit Summary: {report['rtg_eligible_episodes']} eligible episodes, "
          f"{report['rtg_eligible_transitions']}/{report['transitions']} transitions retained.")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    """Prepare clean dataset with RTG labels and validation split."""
    in_path = Path(args.data)
    data = load_dataset(in_path)

    arrays, manifest = prepare_dataset(
        data=data,
        horizon=args.horizon,
        starts_at_reset=args.starts_at_reset,
        timeout_semantics=args.timeout_semantics,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        reward_scale=args.reward_scale,
    )

    out_dir = Path(args.output)
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"Error: Output directory {out_dir} already exists and is not empty.", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "dataset.npz", **arrays)

    manifest["provenance"] = {
        "command": ["prepare", str(in_path)],
        "source_sha256": compute_file_sha256(in_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Prepared dataset written to {out_dir}:")
    print(f"  - Transitions: {manifest['retained_transitions']:,}")
    print(f"  - Train episodes: {len(manifest['train_episode_ids'])}, Val episodes: {len(manifest['validation_episode_ids'])}")
    return 0


def cmd_train_r(args: argparse.Namespace) -> int:
    """Train Module R expectile model."""
    in_path = Path(args.data)
    with np.load(in_path, allow_pickle=False) as loaded:
        dataset = {k: loaded[k] for k in loaded.files}

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
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(capper.model, torch.nn.Module):
        torch.save({
            "model_state_dict": capper.model.state_dict(),
            "mode": args.mode,
            "horizon": args.horizon,
            "tau": args.tau,
            "margin": args.margin,
            "scale": args.reward_scale,
            "state_mean": capper.state_mean,
            "state_std": capper.state_std,
            "metrics": metrics,
        }, out_dir / "return_capper.pt")
        print(f"Trained neural capper saved to {out_dir / 'return_capper.pt'}")
    else:
        # Baseline B2 global cap
        (out_dir / "return_capper_global.json").write_text(json.dumps({
            "cap_value": metrics["cap_value"],
            "tau": args.tau,
            "horizon": args.horizon,
            "margin": args.margin,
            "scale": args.reward_scale,
        }, indent=2))
        print(f"Global scalar capper saved to {out_dir / 'return_capper_global.json'}")

    print(f"Final Validation Loss: {metrics['val_loss'][-1]:.6f}, Empirical Coverage: {metrics['val_coverage'][-1]*100:.2f}%")
    return 0


def cmd_verify_r(args: argparse.Namespace) -> int:
    """Execute rigorous mathematical, numerical, and edge-case verification for Module R."""
    print("=" * 80)
    print("MODULE R: RIGOROUS MATHEMATICAL & NUMERICAL VERIFICATION")
    print("=" * 80)

    # 1. Asymmetric Expectile Loss Gradient Verification
    print("\n[1] Verifying Asymmetric Expectile Loss and Gradients (tau=0.9)")
    loss_fn = ExpectileLoss(tau=0.9, reduction="none")

    pred = torch.tensor([5.0, 5.0], requires_grad=True)
    # Target 1: Under-estimation (target > pred, diff = +2.0)
    # Target 2: Over-estimation (target < pred, diff = -2.0)
    target = torch.tensor([7.0, 3.0])

    loss = loss_fn(pred, target)
    expected_loss_under = 0.9 * (2.0 ** 2)  # 3.6
    expected_loss_over = 0.1 * (2.0 ** 2)   # 0.4

    print(f"  Under-estimation loss (diff=+2): got {loss[0].item():.4f}, expected {expected_loss_under:.4f}")
    print(f"  Over-estimation loss  (diff=-2): got {loss[1].item():.4f}, expected {expected_loss_over:.4f}")
    assert np.isclose(loss[0].item(), expected_loss_under)
    assert np.isclose(loss[1].item(), expected_loss_over)

    # Autograd gradients: dL/d(pred) = -2 * w * diff
    # For under: -2 * 0.9 * (+2.0) = -3.6
    # For over:  -2 * 0.1 * (-2.0) = +0.4
    loss.sum().backward()
    grad = pred.grad
    print(f"  Gradient for under-estimation: {grad[0].item():.4f} (pushes prediction UP)")
    print(f"  Gradient for over-estimation:  {grad[1].item():.4f} (pushes prediction DOWN gently)")
    assert np.isclose(grad[0].item(), -3.6)
    assert np.isclose(grad[1].item(), 0.4)
    ratio = abs(grad[0].item() / grad[1].item())
    print(f"  Penalty ratio: {ratio:.1f}x (exactly tau / (1 - tau) = 0.9 / 0.1 = 9.0x)")
    assert np.isclose(ratio, 9.0)

    # 2. Iteratively Reweighted Least Squares (IRLS) Analytical Solutions
    print("\n[2] Verifying IRLS Global Expectile Against Analytical Solutions")
    # Two-point distribution {0, 100} with equal probability:
    # First-order condition: (1 - tau) * c = tau * (100 - c) => c = 100 * tau
    two_point = np.array([0.0, 100.0])
    for tau in [0.5, 0.7, 0.9, 0.99]:
        computed_c = compute_global_expectile(two_point, tau=tau)
        expected_c = 100.0 * tau
        print(f"  tau={tau:.2f} on {{0, 100}}: computed={computed_c:.6f}, exact analytical={expected_c:.6f}")
        assert np.isclose(computed_c, expected_c, atol=1e-5)

    # Symmetric standard normal distribution: expectile at tau=0.5 must be exactly 0.0
    rng = np.random.default_rng(42)
    norm_samples = rng.normal(loc=0.0, scale=1.0, size=500_000)
    c_norm_half = compute_global_expectile(norm_samples, tau=0.5)
    print(f"  tau=0.5 on N(0, 1) (500k samples): mean={np.mean(norm_samples):.6f}, expectile={c_norm_half:.6f}")
    assert np.isclose(c_norm_half, 0.0, atol=1e-2)

    # 3. Budget Recurrence Decoupling Invariant
    print("\n[3] Verifying Budget Recurrence Decoupling Invariant")
    print("  Mathematical Rule: B_{t+1}^{raw} = B_t^{raw} - r_t / c, Token B_t = min(B_t^{raw}, C_t)")
    raw_budget_0 = 10.0  # Inflated requested target = 10,000 / 1000
    capper = ReturnCapper(model=GlobalReturnCap(cap_value=3.2), horizon=1000, margin=0.0)
    budget = ReturnBudget(raw=raw_budget_0, reward_scale=1000.0)

    # Step 0
    token_0 = capper.step(budget, state=np.zeros(4), step=0)
    print(f"  Step 0: Raw budget = {budget.raw:.4f}, Cap = 3.2000 => Token sent to DT = {token_0:.4f}")
    assert np.isclose(token_0, 3.2)
    assert np.isclose(budget.raw, 10.0), "CRITICAL ERROR: Raw budget was modified by token clipping!"

    # Environment returns reward 100.0 (scaled reward = 0.1)
    budget.observe_reward(100.0)
    print(f"  Observed reward r_0 = 100.0. New raw budget = {budget.raw:.4f} (10.0 - 0.1 = 9.9)")
    assert np.isclose(budget.raw, 9.9), "Raw budget update failed!"

    # Step 1
    token_1 = capper.step(budget, state=np.zeros(4), step=1)
    print(f"  Step 1: Raw budget = {budget.raw:.4f}, Cap = 3.2000 => Token sent to DT = {token_1:.4f}")
    assert np.isclose(token_1, 3.2)
    assert np.isclose(budget.raw, 9.9)

    # 4. Domain & Boundary Edge Cases
    print("\n[4] Verifying Domain & Boundary Edge Cases")
    mlp = ReturnCapMLP(state_dim=4, hidden_dim=32, num_layers=2, mode="state_time")
    boundary_capper = ReturnCapper(model=mlp, horizon=100)

    # t = 0 (start)
    cap_t0 = boundary_capper.get_cap(np.zeros(4), step=0)
    print(f"  Initial step t=0 (t/H=0.0): cap = {cap_t0:.4f} (valid finite)")
    assert np.isfinite(cap_t0)

    # t = H (horizon boundary)
    cap_tH = boundary_capper.get_cap(np.zeros(4), step=100)
    print(f"  Horizon boundary t=100 (t/H=1.0): cap = {cap_tH:.4f} (valid finite)")
    assert np.isfinite(cap_tH)

    # t > H (beyond horizon)
    cap_t_beyond = boundary_capper.get_cap(np.zeros(4), step=250)
    print(f"  Beyond horizon t=250 (clamped to t/H=1.0): cap = {cap_t_beyond:.4f} (equals cap_tH)")
    assert np.isclose(cap_t_beyond, cap_tH)

    # Out-of-distribution observation (extreme magnitudes)
    extreme_state = np.array([1e5, -1e5, 1e6, -1e6], dtype=np.float32)
    cap_extreme = boundary_capper.get_cap(extreme_state, step=50)
    print(f"  Extreme state [1e5, -1e5, 1e6, -1e6]: cap = {cap_extreme:.4f} (no overflow/NaN)")
    assert np.isfinite(cap_extreme)

    # 5. Real Data Verification (if available)
    data_path = Path(args.data) if args.data else Path("data/prepared/hopper-medium-v2/dataset.npz")
    if data_path.exists():
        print(f"\n[5] Verifying on Real Prepared Dataset: {data_path}")
        with np.load(data_path, allow_pickle=False) as d:
            rtgs = d["returns_to_go"]
            val_mask = d["is_validation"]
            train_rtgs = rtgs[~val_mask]
            val_rtgs = rtgs[val_mask]

            print(f"  Total RTG labels: {len(rtgs):,} (Train: {len(train_rtgs):,}, Val: {len(val_rtgs):,})")
            print(f"  Train RTG Range: [{np.min(train_rtgs):.4f}, {np.max(train_rtgs):.4f}], Mean = {np.mean(train_rtgs):.4f}")

            # Compute Baseline B2 Global Expectile
            cap_b2 = compute_global_expectile(train_rtgs, tau=0.9)
            val_cov_b2 = np.mean(val_rtgs <= cap_b2)
            print(f"  Baseline B2 Global Expectile (tau=0.9): {cap_b2:.4f}")
            print(f"  Baseline B2 Empirical Validation Coverage: {val_cov_b2*100:.2f}% (Expected ~85-95%)")
            assert 0.80 <= val_cov_b2 <= 0.99, f"Unexpected validation coverage: {val_cov_b2}"

            cap_b2_99 = compute_global_expectile(train_rtgs, tau=0.99)
            val_cov_b2_99 = np.mean(val_rtgs <= cap_b2_99)
            print(f"  Baseline B2 Global Expectile (tau=0.99): {cap_b2_99:.4f}")
            print(f"  Baseline B2 (tau=0.99) Validation Coverage: {val_cov_b2_99*100:.2f}%")
            assert val_cov_b2_99 > val_cov_b2

    print("\n" + "=" * 80)
    print("ALL NUMERICAL, MATHEMATICAL & EDGE-CASE INVARIANTS VERIFIED!")
    print("=" * 80)
    return 0


# -----------------------------------------------------------------------------
# Main Argument Parser
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Unified CLI for UDT-Adapter offline RL verification and experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to execute")

    # download
    p_dl = subparsers.add_parser("download", help="Download official D4RL datasets")
    p_dl.add_argument("dataset", choices=sorted(D4RL_URLS.keys()), help="Dataset name")
    p_dl.add_argument("--target-dir", default="data/raw", help="Target raw directory")
    p_dl.add_argument("--force", action="store_true", help="Force re-download")
    p_dl.set_defaults(func=cmd_download)

    # demo-data
    p_demo = subparsers.add_parser("demo-data", help="Create synthetic demo dataset")
    p_demo.add_argument("--output", required=True, help="Output .npz path")
    p_demo.set_defaults(func=cmd_demo_data)

    # audit
    p_audit = subparsers.add_parser("audit", help="Audit dataset boundary and integrity")
    p_audit.add_argument("data", help="Input .hdf5, .h5, or .npz path")
    p_audit.add_argument("--horizon", type=int, required=True, help="Episode horizon")
    p_audit.add_argument("--starts-at-reset", action="store_true", help="Dataset starts with reset")
    p_audit.add_argument("--timeout-semantics", default="unknown", choices=["unknown", "horizon"])
    p_audit.add_argument("--dataset-id", default="unspecified", help="Dataset identifier for provenance")
    p_audit.add_argument("--output", help="Optional output JSON report path")
    p_audit.set_defaults(func=cmd_audit)

    # prepare
    p_prep = subparsers.add_parser("prepare", help="Prepare dataset with RTG labels and splits")
    p_prep.add_argument("data", help="Input raw dataset path")
    p_prep.add_argument("--horizon", type=int, required=True, help="Episode horizon")
    p_prep.add_argument("--output", required=True, help="Output directory")
    p_prep.add_argument("--starts-at-reset", action="store_true", help="Dataset starts with reset")
    p_prep.add_argument("--timeout-semantics", default="unknown", choices=["unknown", "horizon"])
    p_prep.add_argument("--seed", type=int, default=0, help="Split seed")
    p_prep.add_argument("--validation-fraction", type=float, default=0.1, help="Validation fraction")
    p_prep.add_argument("--reward-scale", type=float, default=1000.0, help="Reward/return scale")
    p_prep.set_defaults(func=cmd_prepare)

    # train-r
    p_train = subparsers.add_parser("train-r", help="Train Module R return capper")
    p_train.add_argument("--data", required=True, help="Prepared dataset.npz path")
    p_train.add_argument("--horizon", type=int, required=True, help="Episode horizon")
    p_train.add_argument("--output", default="checkpoints/module_r", help="Output directory")
    p_train.add_argument("--mode", default="state_time", choices=["state_time", "time_only", "state_only", "global"])
    p_train.add_argument("--tau", type=float, default=0.9, help="Expectile parameter tau in (0, 1)")
    p_train.add_argument("--margin", type=float, default=0.0, help="Inference delta margin")
    p_train.add_argument("--reward-scale", type=float, default=1000.0, help="Return scaling factor c")
    p_train.add_argument("--epochs", type=int, default=15, help="Training epochs")
    p_train.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p_train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_train.set_defaults(func=cmd_train_r)

    # verify-r
    p_verify = subparsers.add_parser("verify-r", help="Run numerical & edge-case verification for Module R")
    p_verify.add_argument("--data", help="Optional path to prepared dataset.npz")
    p_verify.set_defaults(func=cmd_verify_r)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
