import numpy as np
import pytest
import torch

from udt.return_cap import (
    GlobalReturnCap,
    ReturnCapMLP,
    ReturnCapper,
    evaluate_return_capper,
    train_return_cap,
)


def _make_dataset(n_samples: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(n_samples, 2)).astype(np.float32)
    timesteps = rng.integers(0, 10, size=n_samples)
    returns = (obs[:, 0] * 2.0 + rng.normal(scale=0.1, size=n_samples)).astype(np.float32)
    is_val = np.zeros(n_samples, dtype=bool)
    is_val[int(n_samples * 0.8):] = True
    return {
        "observations": obs,
        "returns_to_go": returns,
        "timesteps": timesteps,
        "is_validation": is_val,
    }


def compute_expectile_loss_direct(y: np.ndarray, y_hat: np.ndarray, tau: float) -> float:
    diff = y - y_hat
    weights = np.where(diff < 0.0, 1.0 - tau, tau)
    return float(np.mean(weights * (diff ** 2)))


def test_evaluate_return_capper_margin_zero():
    """evaluate_return_capper matches direct expectile loss when margin=0."""
    ds = _make_dataset()
    val_returns = ds["returns_to_go"][ds["is_validation"]]
    cap_val = 1.5
    capper = ReturnCapper(
        model=GlobalReturnCap(cap_val),
        horizon=10,
        margin=0.0,
        scale=1.0,
    )
    result = evaluate_return_capper(ds, capper, horizon=10, margin=0.0, tau=0.9, split="validation")
    direct_loss = compute_expectile_loss_direct(val_returns, np.full_like(val_returns, cap_val), tau=0.9)
    assert np.isclose(result["loss"], direct_loss)
    direct_cov = float(np.mean(val_returns <= cap_val))
    assert np.isclose(result["coverage"], direct_cov)


def test_evaluate_return_capper_margin_positive():
    """Positive margin changes diagnostic coverage but does NOT change expectile loss."""
    ds = _make_dataset()
    val_returns = ds["returns_to_go"][ds["is_validation"]]
    cap_val = 1.5
    capper = ReturnCapper(
        model=GlobalReturnCap(cap_val),
        horizon=10,
        margin=0.5,
        scale=1.0,
    )
    res_m0 = evaluate_return_capper(ds, capper, horizon=10, margin=0.0, tau=0.9, split="validation")
    res_m5 = evaluate_return_capper(ds, capper, horizon=10, margin=0.5, tau=0.9, split="validation")

    direct_loss = compute_expectile_loss_direct(val_returns, np.full_like(val_returns, cap_val), tau=0.9)
    # The expectile loss MUST NOT change when margin > 0
    assert np.isclose(res_m5["loss"], direct_loss)
    assert np.isclose(res_m5["loss"], res_m0["loss"])

    # Coverage MUST incorporate margin: returns <= cap_val + 0.5
    expected_cov_m5 = float(np.mean(val_returns <= cap_val + 0.5))
    assert np.isclose(res_m5["coverage"], expected_cov_m5)


def test_neural_train_return_cap_best_checkpoint_matches_val_loss_with_margin():
    """Neural train_return_cap with margin > 0 has best_val_loss matching validation expectile loss."""
    ds = _make_dataset(n_samples=200, seed=123)
    capper, metrics = train_return_cap(
        dataset=ds,
        horizon=10,
        tau=0.9,
        mode="state_time",
        margin=0.5,
        epochs=3,
        batch_size=32,
        seed=42,
    )

    best_epoch = metrics["best_epoch"]
    training_val_loss_at_best_epoch = metrics["val_loss"][best_epoch]
    reported_best_val_loss = metrics["best_val_loss"]

    # best_val_loss reported must match training validation loss at best epoch
    assert np.isclose(reported_best_val_loss, training_val_loss_at_best_epoch, rtol=1e-4)

    # Independent evaluation must also match
    indep = evaluate_return_capper(ds, capper, horizon=10, margin=0.5, tau=0.9, split="validation")
    assert np.isclose(indep["loss"], reported_best_val_loss, rtol=1e-4)


def test_global_train_return_cap_metrics_consistent_with_evaluation():
    """GlobalReturnCap metrics have consistent expectile loss regardless of margin."""
    ds = _make_dataset(seed=999)
    capper_m0, met_m0 = train_return_cap(ds, horizon=10, tau=0.9, mode="global", margin=0.0)
    capper_m5, met_m5 = train_return_cap(ds, horizon=10, tau=0.9, mode="global", margin=0.5)

    # Both models have identical cap value and expectile loss
    assert np.isclose(met_m0["best_val_loss"], met_m5["best_val_loss"])

    # evaluate_return_capper must agree with best_val_loss for margin > 0
    indep_m5 = evaluate_return_capper(ds, capper_m5, horizon=10, margin=0.5, tau=0.9, split="validation")
    assert np.isclose(indep_m5["loss"], met_m5["best_val_loss"])
