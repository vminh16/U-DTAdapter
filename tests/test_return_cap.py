import numpy as np
import pytest
import torch
import torch.nn.functional as F

from udt.return_cap import (
    ExpectileLoss,
    GlobalReturnCap,
    ReturnCapMLP,
    ReturnCapper,
    compute_expectile_foc_residual,
    compute_global_expectile,
    evaluate_return_capper,
    load_return_capper,
    save_return_capper,
    train_return_cap,
)
from udt.intervention import ReturnBudget



def test_expectile_loss_symmetric_tau_half():
    criterion = ExpectileLoss(tau=0.5, reduction="mean")
    pred = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    target = torch.tensor([2.0, 0.0, 5.0])
    loss = criterion(pred, target)
    expected_mse = F.mse_loss(pred, target)
    assert torch.isclose(loss, 0.5 * expected_mse)


def test_expectile_loss_asymmetry():
    criterion = ExpectileLoss(tau=0.9, reduction="none")
    # Under-estimation: target > pred (diff > 0), weight = tau = 0.9
    pred_under = torch.tensor([8.0])
    target_under = torch.tensor([10.0])
    loss_under = criterion(pred_under, target_under)
    assert torch.isclose(loss_under, torch.tensor([0.9 * 4.0]))

    # Over-estimation: target < pred (diff < 0), weight = 1 - tau = 0.1
    pred_over = torch.tensor([10.0])
    target_over = torch.tensor([8.0])
    loss_over = criterion(pred_over, target_over)
    assert torch.isclose(loss_over, torch.tensor([0.1 * 4.0]))

    ratio = loss_under / loss_over
    assert torch.isclose(ratio, torch.tensor([0.9 / 0.1]))


def test_expectile_loss_rejects_invalid_tau():
    with pytest.raises(ValueError, match="tau must be in"):
        ExpectileLoss(tau=0.0)
    with pytest.raises(ValueError, match="tau must be in"):
        ExpectileLoss(tau=1.0)
    with pytest.raises(ValueError, match="tau must be in"):
        ExpectileLoss(tau=-0.2)
    with pytest.raises(ValueError, match="tau must be in"):
        ExpectileLoss(tau=1.5)


def test_compute_global_expectile():
    # Symmetric uniform distribution: expectile at tau=0.5 is 0.0
    uniform_data = np.linspace(-10.0, 10.0, 1001)
    e_half = compute_global_expectile(uniform_data, tau=0.5)
    assert np.isclose(e_half, 0.0, atol=1e-3)
    res_half = compute_expectile_foc_residual(uniform_data, e_half, tau=0.5)
    assert np.isclose(res_half, 0.0, atol=1e-5)

    # Two-point distribution {0, 100} with equal probability:
    # Expectile equation: (1 - tau) * c = tau * (100 - c) => c = 100 * tau
    two_point_data = np.array([0.0, 100.0])
    e_09 = compute_global_expectile(two_point_data, tau=0.9)
    assert np.isclose(e_09, 90.0, atol=1e-3)
    res_09 = compute_expectile_foc_residual(two_point_data, e_09, tau=0.9)
    assert np.isclose(res_09, 0.0, atol=1e-5)


def test_expectile_is_not_quantile_mandatory_counterexample():
    """Mandatory verification: Expectile at level tau is NOT a quantile.

    For Y in {0, 100} with equal probability:
        e_{0.9} = 90, but P(Y <= 90) = P(Y = 0) = 0.5 != 0.9.
    First-order condition:
        (1/n) sum_i w_tau(Y_i - e)(Y_i - e) = 0.5 * (1 - 0.9) * (0 - 90) + 0.5 * 0.9 * (100 - 90) = -4.5 + 4.5 = 0.
    Coverage at tau=0.5 and tau=0.9 are both 0.5, proving coverage does not strictly increase.
    """
    two_point = np.array([0.0, 100.0])
    e_05 = compute_global_expectile(two_point, tau=0.5)
    e_09 = compute_global_expectile(two_point, tau=0.9)

    assert np.isclose(e_05, 50.0)
    assert np.isclose(e_09, 90.0)

    cov_05 = np.mean(two_point <= e_05)
    cov_09 = np.mean(two_point <= e_09)

    # Both coverages are 0.5!
    assert cov_05 == 0.5
    assert cov_09 == 0.5  # NOT 0.9! Expectile is not a quantile!
    assert cov_09 == cov_05  # Coverage does NOT strictly increase with tau!

    # FOC residuals are exactly 0
    assert np.isclose(compute_expectile_foc_residual(two_point, e_05, 0.5), 0.0)
    assert np.isclose(compute_expectile_foc_residual(two_point, e_09, 0.9), 0.0)


def test_compute_global_expectile_edge_cases_and_skewed_distributions():
    # 1. Constant array
    constant_arr = np.full(50, 42.0)
    for tau in [0.1, 0.5, 0.9]:
        e_const = compute_global_expectile(constant_arr, tau=tau)
        assert np.isclose(e_const, 42.0)
        assert np.isclose(compute_expectile_foc_residual(constant_arr, e_const, tau), 0.0)

    # 2. Single element
    single_arr = np.array([7.25])
    e_single = compute_global_expectile(single_arr, tau=0.8)
    assert np.isclose(e_single, 7.25)
    assert np.isclose(compute_expectile_foc_residual(single_arr, e_single, 0.8), 0.0)

    # 3. Skewed distribution (Exponential)
    rng = np.random.default_rng(123)
    exp_samples = rng.exponential(scale=2.0, size=50_000)
    for tau in [0.2, 0.5, 0.8, 0.95]:
        e_exp = compute_global_expectile(exp_samples, tau=tau)
        foc_res = compute_expectile_foc_residual(exp_samples, e_exp, tau)
        assert np.isclose(foc_res, 0.0, atol=1e-4)


def test_global_return_cap():
    cap = GlobalReturnCap(cap_value=90.0)
    dummy_state = np.zeros(10)
    assert cap.get_cap(dummy_state, step=0) == 90.0
    assert cap.get_cap(dummy_state, step=500) == 90.0


def test_return_cap_mlp_forward_shapes():
    state_dim = 17
    batch_size = 32

    # Mode: state_time (Baseline B3)
    mlp_st = ReturnCapMLP(state_dim=state_dim, hidden_dim=128, num_layers=3, mode="state_time")
    states = torch.randn(batch_size, state_dim)
    t_rel = torch.rand(batch_size, 1)
    out_st = mlp_st(states, t_rel)
    assert out_st.shape == (batch_size, 1)

    # 1D single-step input
    out_st_single = mlp_st(states[0], t_rel[0])
    assert out_st_single.shape == (1, 1)

    # Batched states with scalar float or 0D tensor timestep (crucial for candidate evaluation)
    out_st_scalar_t = mlp_st(states, 0.5)
    assert out_st_scalar_t.shape == (batch_size, 1)

    out_st_0d_tensor_t = mlp_st(states, torch.tensor(0.5))
    assert out_st_0d_tensor_t.shape == (batch_size, 1)

    # Mode: time_only (Baseline B9)

    mlp_to = ReturnCapMLP(state_dim=state_dim, hidden_dim=64, num_layers=2, mode="time_only")
    out_to = mlp_to(None, t_rel)
    assert out_to.shape == (batch_size, 1)

    # Mode: state_only (Ablation)
    mlp_so = ReturnCapMLP(state_dim=state_dim, hidden_dim=64, num_layers=2, mode="state_only")
    out_so = mlp_so(states, None)
    assert out_so.shape == (batch_size, 1)


def test_return_cap_mlp_parameter_count():
    # Strict parameter budget from RRD: < 300k parameters for auxiliary networks
    mlp = ReturnCapMLP(state_dim=24, hidden_dim=128, num_layers=3, mode="state_time")
    num_params = sum(p.numel() for p in mlp.parameters())
    assert num_params < 300_000
    assert num_params > 1_000


def test_return_capper_clips_budget_without_mutating_raw():
    global_cap = GlobalReturnCap(cap_value=1.0)
    capper = ReturnCapper(model=global_cap, horizon=1000, margin=0.0)
    budget = ReturnBudget(raw=1.5, reward_scale=1000.0)
    dummy_state = np.zeros(10)

    token = capper.step(budget, dummy_state, step=10)
    # Token is clipped to cap
    assert np.isclose(token, 1.0)
    # Raw budget must NOT be mutated by clipping (critical mathematical requirement)
    assert np.isclose(budget.raw, 1.5)

    # Verify that observe_reward operates on raw budget
    budget.observe_reward(reward=200.0)  # raw becomes 1.5 - 0.2 = 1.3
    assert np.isclose(budget.raw, 1.3)



def test_return_capper_respects_safety_margin():
    global_cap = GlobalReturnCap(cap_value=1.0)
    capper = ReturnCapper(model=global_cap, horizon=1000, margin=0.2)
    budget = ReturnBudget(raw=1.5)
    dummy_state = np.zeros(10)

    token = capper.step(budget, dummy_state, step=10)
    # Token = min(1.5, 1.0 + 0.2) = 1.2
    assert np.isclose(token, 1.2)


def test_return_capper_handles_horizon_boundary():
    mlp = ReturnCapMLP(state_dim=4, hidden_dim=32, num_layers=2, mode="state_time")
    capper = ReturnCapper(model=mlp, horizon=100, margin=0.0)
    budget = ReturnBudget(raw=1.0)
    state = np.zeros(4)

    # Step at or beyond horizon should not crash or produce NaNs
    token_at_h = capper.step(budget, state, step=100)
    token_beyond_h = capper.step(budget, state, step=150)
    assert not np.isnan(token_at_h)
    assert not np.isnan(token_beyond_h)


def test_train_return_cap_synthetic_dataset():
    # Build a synthetic dataset with 10 episodes of 20 steps each
    rng = np.random.default_rng(42)
    state_dim = 4
    num_episodes = 20
    ep_len = 25
    horizon = 25

    obs_list, rtg_list, t_list, ep_id_list = [], [], [], []
    for ep in range(num_episodes):
        # States drift slightly with step
        for t in range(ep_len):
            s = rng.normal(loc=float(t) / 10.0, scale=0.5, size=state_dim)
            # Remaining return roughly (ep_len - t) with positive reward
            r_remaining = (ep_len - t) * 0.1 + rng.normal(0, 0.05)
            obs_list.append(s)
            rtg_list.append(r_remaining)
            t_list.append(t)
            ep_id_list.append(ep)

    dataset = {
        "observations": np.array(obs_list, dtype=np.float32),
        "returns_to_go": np.array(rtg_list, dtype=np.float32),
        "timesteps": np.array(t_list, dtype=np.int64),
        "episode_ids": np.array(ep_id_list, dtype=np.int64),
        "is_validation": np.isin(ep_id_list, [0, 1, 2, 3]),
    }

    capper, metrics = train_return_cap(
        dataset=dataset,
        horizon=horizon,
        tau=0.9,
        mode="state_time",
        epochs=15,
        lr=5e-3,
        batch_size=64,
        device="cpu",
    )

    assert isinstance(capper, ReturnCapper)
    assert "val_loss" in metrics
    assert len(metrics["val_loss"]) > 0
    # Val loss should decrease from epoch 0 to final
    assert metrics["val_loss"][-1] < metrics["val_loss"][0]
    # Check diagnostic coverage on validation (must be valid probability in [0, 1])
    final_coverage = metrics["val_coverage"][-1]
    assert 0.0 <= final_coverage <= 1.0


def test_return_capper_accepts_tensor_with_grad():
    mlp = ReturnCapMLP(state_dim=4, hidden_dim=32, num_layers=2, mode="state_time")
    capper = ReturnCapper(model=mlp, horizon=100)
    # PyTorch tensor with gradient
    state_tensor = torch.randn(4, requires_grad=True)
    cap = capper.get_cap(state_tensor, step=10)
    assert np.isfinite(cap)


def test_return_capper_rejects_non_positive_horizon():
    mlp = ReturnCapMLP(state_dim=4, hidden_dim=32, num_layers=2, mode="state_time")
    with pytest.raises(ValueError, match="horizon must be positive"):
        ReturnCapper(model=mlp, horizon=0)
    with pytest.raises(ValueError, match="horizon must be positive"):
        ReturnCapper(model=mlp, horizon=-10)


def test_train_return_cap_normalization_does_not_use_validation_samples():
    # Training obs: mean ~ 0.0
    train_obs = np.random.default_rng(1).normal(loc=0.0, scale=1.0, size=(100, 2))
    # Validation obs: extreme values, mean ~ 1000.0
    val_obs = np.full((50, 2), 1000.0)

    all_obs = np.vstack([train_obs, val_obs]).astype(np.float32)
    is_val = np.array([False] * 100 + [True] * 50)
    dataset = {
        "observations": all_obs,
        "returns_to_go": np.ones(150, dtype=np.float32),
        "timesteps": np.zeros(150, dtype=np.int64),
        "is_validation": is_val,
    }

    capper, _ = train_return_cap(
        dataset=dataset,
        horizon=100,
        mode="state_time",
        epochs=1,
        device="cpu",
    )

    # capper.state_mean must match training obs only, completely ignoring validation samples!
    expected_train_mean = np.mean(train_obs, axis=0)
    assert np.allclose(capper.state_mean, expected_train_mean, atol=1e-5)
    # If validation samples were contaminated, mean would be ~333.3
    assert not np.any(capper.state_mean > 50.0)


def test_train_return_cap_rejects_dataset_scale_and_horizon_mismatch():
    dataset = {
        "observations": np.zeros((20, 2), dtype=np.float32),
        "returns_to_go": np.ones(20, dtype=np.float32),
        "timesteps": np.zeros(20, dtype=np.int64),
        "is_validation": np.array([False] * 15 + [True] * 5),
        "horizon": 1000,
        "reward_scale": 1000.0,
    }

    with pytest.raises(ValueError, match="Horizon mismatch"):
        train_return_cap(dataset, horizon=500, scale=1000.0)

    with pytest.raises(ValueError, match="Reward scale mismatch"):
        train_return_cap(dataset, horizon=1000, scale=500.0)


def test_raw_rtg_recurrence_and_clipping_preservation():
    # Invariant: B_{t+1}^{raw} = B_t^{raw} - r_t / c
    # B_t = min(B_t^{raw}, cap + delta)
    budget = ReturnBudget(raw=10.0, reward_scale=1000.0)
    capper = ReturnCapper(model=GlobalReturnCap(cap_value=2.0), horizon=1000, scale=1000.0, margin=0.5)

    # Step 0: requested 10.0, cap is 2.0 + 0.5 = 2.5
    token_0 = capper.step(budget, state=np.zeros(2), step=0)
    assert np.isclose(token_0, 2.5)
    assert np.isclose(budget.raw, 10.0)  # Clipping MUST NOT mutate raw budget!

    # Observe reward 250.0 -> scaled reward = 0.25 -> raw becomes 9.75
    budget.observe_reward(250.0)
    assert np.isclose(budget.raw, 9.75)

    # Step 1: token still clipped to 2.5
    token_1 = capper.step(budget, state=np.zeros(2), step=1)
    assert np.isclose(token_1, 2.5)
    assert np.isclose(budget.raw, 9.75)


def test_train_return_cap_global_baseline():
    dataset = {
        "observations": np.zeros((100, 4), dtype=np.float32),
        "returns_to_go": np.linspace(0.0, 10.0, 100, dtype=np.float32),
        "timesteps": np.zeros(100, dtype=np.int64),
        "is_validation": np.array([False] * 80 + [True] * 20),
    }
    capper, metrics = train_return_cap(
        dataset=dataset,
        horizon=100,
        tau=0.9,
        mode="global",
    )
    assert isinstance(capper, ReturnCapper)
    assert isinstance(capper.model, GlobalReturnCap)
    assert "cap_value" in metrics
    assert np.isclose(capper.get_cap(np.zeros(4), step=0), metrics["cap_value"])

    # Issue B3: Loss for GlobalReturnCap must be the TRUE non-zero expectile loss, not 0.0!
    assert metrics["train_loss"][0] > 0.0
    assert metrics["val_loss"][0] > 0.0
    assert "best_val_loss" in metrics
    assert np.isclose(metrics["best_val_loss"], metrics["val_loss"][0])


def test_validation_loss_sample_weighting_outlier_batch():
    """Issue B1: Regression test for sample-weighted validation loss.

    257 samples with batch_size 256.
    Batch 1 (256 samples): returns = 0.0
    Batch 2 (1 sample): return = 100.0 (outlier)
    Model initialized to output 0.0.
    At tau=0.5 (MSE * 0.5):
      Batch 1 error = 0 -> loss = 0
      Batch 2 error = 100 -> loss = 0.5 * 100^2 = 5000.0
    True sample-weighted mean loss = (256 * 0 + 1 * 5000) / 257 = 5000 / 257 ~= 19.455.
    Unweighted batch mean would be (0 + 5000) / 2 = 2500.0!
    """
    n_train = 50
    n_val = 257
    val_obs = np.zeros((n_val, 2), dtype=np.float32)
    val_rtgs = np.zeros(n_val, dtype=np.float32)
    val_rtgs[-1] = 100.0  # single outlier sample in second batch

    train_obs = np.zeros((n_train, 2), dtype=np.float32)
    train_rtgs = np.zeros(n_train, dtype=np.float32)

    all_obs = np.vstack([train_obs, val_obs])
    all_rtgs = np.concatenate([train_rtgs, val_rtgs])
    is_val = np.array([False] * n_train + [True] * n_val)

    dataset = {
        "observations": all_obs,
        "returns_to_go": all_rtgs,
        "timesteps": np.zeros(len(all_obs), dtype=np.int64),
        "is_validation": is_val,
    }

    # Mode time_only with 0 epochs or lr=0 to observe initial model loss
    capper, metrics = train_return_cap(
        dataset=dataset,
        horizon=100,
        tau=0.5,
        mode="time_only",
        epochs=1,
        lr=0.0,  # Zero lr keeps weights unchanged
        batch_size=256,
        seed=42,
    )

    # Initial MLP bias and weights: we can evaluate directly on all 257 val samples
    # to compare against reported val_loss[0]
    expected_direct_val_loss = float(
        np.mean(
            0.5 * (val_rtgs - np.array([capper.get_cap(val_obs[i], 0) for i in range(n_val)])) ** 2
        )
    )
    reported_val_loss = metrics["val_loss"][0]
    assert np.isclose(reported_val_loss, expected_direct_val_loss, rtol=1e-3)
    # Ensure it's not the erroneous unweighted batch average (which would be > 1000)
    assert reported_val_loss < 200.0


def test_metrics_match_best_checkpoint():
    """Issue B2: Metrics must match the restored best checkpoint, not the last epoch."""
    n_samples = 80
    dataset = {
        "observations": np.zeros((n_samples, 2), dtype=np.float32),
        "returns_to_go": np.linspace(1.0, 5.0, n_samples, dtype=np.float32),
        "timesteps": np.zeros(n_samples, dtype=np.int64),
        "is_validation": np.array([False] * 50 + [True] * 30),
    }

    capper, metrics = train_return_cap(
        dataset=dataset,
        horizon=100,
        tau=0.9,
        mode="state_time",
        epochs=10,
        batch_size=16,
        seed=123,
    )

    assert "best_epoch" in metrics
    assert "best_val_loss" in metrics
    assert metrics["best_val_loss"] <= min(metrics["val_loss"]) + 1e-6

    # Re-evaluating the returned capper on validation data must match best_val_loss
    val_obs = dataset["observations"][dataset["is_validation"]]
    val_rtgs = dataset["returns_to_go"][dataset["is_validation"]]
    preds = np.array([capper.get_cap(val_obs[i], 0) for i in range(len(val_obs))])
    diff = val_rtgs - preds
    weights = np.where(diff < 0, 1.0 - 0.9, 0.9)
    computed_val_loss = float(np.mean(weights * (diff ** 2)))
    assert np.isclose(computed_val_loss, metrics["best_val_loss"], rtol=1e-3)


def test_reproducible_training_with_seed():
    """Issue B4: Training reproducibility with seed."""
    n_samples = 60
    dataset = {
        "observations": np.random.default_rng(0).normal(size=(n_samples, 3)).astype(np.float32),
        "returns_to_go": np.linspace(0.5, 3.0, n_samples, dtype=np.float32),
        "timesteps": np.zeros(n_samples, dtype=np.int64),
        "is_validation": np.array([False] * 40 + [True] * 20),
    }

    capper1, m1 = train_return_cap(dataset, horizon=100, mode="state_time", epochs=3, seed=777)
    capper2, m2 = train_return_cap(dataset, horizon=100, mode="state_time", epochs=3, seed=777)

    # Losses must be identical
    np.testing.assert_allclose(m1["train_loss"], m2["train_loss"])
    np.testing.assert_allclose(m1["val_loss"], m2["val_loss"])

    # Output caps on arbitrary inputs must match exactly
    test_obs = np.array([0.5, -1.0, 2.0], dtype=np.float32)
    assert np.isclose(capper1.get_cap(test_obs, 10), capper2.get_cap(test_obs, 10))

    # Different seed gives different initialization
    capper3, m3 = train_return_cap(dataset, horizon=100, mode="state_time", epochs=3, seed=888)
    # Invariant holds
    assert np.isfinite(capper3.get_cap(test_obs, 10))


def test_training_input_validation():
    """Issue B5: Rigorous training input validation."""
    valid_ds = {
        "observations": np.zeros((20, 2), dtype=np.float32),
        "returns_to_go": np.ones(20, dtype=np.float32),
        "timesteps": np.zeros(20, dtype=np.int64),
        "is_validation": np.array([False] * 15 + [True] * 5),
    }

    # 1. Non-boolean is_validation
    bad_ds = dict(valid_ds, is_validation=np.array([0] * 15 + [1] * 5, dtype=np.int32))
    with pytest.raises(ValueError, match="is_validation.*boolean"):
        train_return_cap(bad_ds, horizon=100)

    # 2. Non-finite values
    bad_obs = valid_ds["observations"].copy()
    bad_obs[0, 0] = np.nan
    with pytest.raises(ValueError, match="observations.*non-finite"):
        train_return_cap(dict(valid_ds, observations=bad_obs), horizon=100)

    bad_rtg = valid_ds["returns_to_go"].copy()
    bad_rtg[2] = np.inf
    with pytest.raises(ValueError, match="returns_to_go.*non-finite"):
        train_return_cap(dict(valid_ds, returns_to_go=bad_rtg), horizon=100)

    # 3. Timesteps domain
    bad_t_neg = valid_ds["timesteps"].copy()
    bad_t_neg[0] = -1
    with pytest.raises(ValueError, match="timesteps.*domain"):
        train_return_cap(dict(valid_ds, timesteps=bad_t_neg), horizon=100)

    bad_t_large = valid_ds["timesteps"].copy()
    bad_t_large[0] = 500
    with pytest.raises(ValueError, match="timesteps.*domain"):
        train_return_cap(dict(valid_ds, timesteps=bad_t_large), horizon=100)

    # 4. Inconsistent transition count
    bad_len = dict(valid_ds, returns_to_go=np.ones(19, dtype=np.float32))
    with pytest.raises(ValueError, match="returns_to_go.*transition count mismatch"):
        train_return_cap(bad_len, horizon=100)

    # 5. Empty dataset or missing train/val samples
    empty_ds = {
        "observations": np.zeros((0, 2), dtype=np.float32),
        "returns_to_go": np.zeros(0, dtype=np.float32),
        "timesteps": np.zeros(0, dtype=np.int64),
        "is_validation": np.zeros(0, dtype=bool),
    }
    with pytest.raises(ValueError, match="empty"):
        train_return_cap(empty_ds, horizon=100)

    no_val = dict(valid_ds, is_validation=np.zeros(20, dtype=bool))
    with pytest.raises(ValueError, match="validation samples"):
        train_return_cap(no_val, horizon=100)


def test_reported_best_metrics_match_restored_checkpoint():
    rng = np.random.default_rng(3)
    n = 120
    obs = rng.normal(size=(n, 3)).astype(np.float32)
    timesteps = np.tile(np.arange(10), 12).astype(np.int64)
    returns = (obs[:, 0] * 0.3 + rng.normal(scale=0.05, size=n)).astype(np.float32)
    dataset = {
        "observations": obs,
        "returns_to_go": returns,
        "timesteps": timesteps,
        "is_validation": np.repeat([False, True], n // 2),
    }

    capper, metrics = train_return_cap(
        dataset, horizon=10, mode="state_time", tau=0.9,
        epochs=6, patience=10, batch_size=16, seed=5,
    )
    independent = evaluate_return_capper(
        dataset, capper, horizon=10, tau=0.9, split="validation"
    )
    assert np.isclose(metrics["best_val_loss"], independent["loss"])
    assert np.isclose(metrics["best_val_coverage"], independent["coverage"])
    assert metrics["best_val_coverage"] == metrics["val_coverage"][metrics["best_epoch"]]

    # Global-cap branch also reports best (single) metrics explicitly.
    _, global_metrics = train_return_cap(dataset, horizon=10, mode="global", tau=0.9)
    assert "best_val_coverage" in global_metrics
    assert global_metrics["best_val_loss"] == global_metrics["val_loss"][-1]




def test_save_load_return_capper_roundtrip(tmp_path):
    model = ReturnCapMLP(state_dim=3, hidden_dim=8, num_layers=2, mode="state_time")
    capper = ReturnCapper(
        model=model,
        horizon=10,
        margin=0.1,
        scale=1000.0,
        state_mean=np.zeros(3, dtype=np.float32),
        state_std=np.ones(3, dtype=np.float32),
    )
    path = save_return_capper(
        tmp_path / "return.pt",
        capper,
        metrics={"best_val_loss": 1.0},
        seed=7,
        tau=0.9,
    )
    loaded, payload = load_return_capper(path, expected_horizon=10, expected_scale=1000.0)
    assert payload["seed"] == 7
    state = np.array([0.5, -0.5, 0.0], dtype=np.float32)
    assert np.isclose(loaded.get_cap(state, step=3), capper.get_cap(state, step=3))


def test_load_legacy_return_capper_infers_architecture(tmp_path):
    model = ReturnCapMLP(state_dim=3, hidden_dim=8, num_layers=2, mode="state_time")
    payload = {
        "model_state_dict": model.state_dict(),
        "mode": "state_time",
        "horizon": 10,
        "tau": 0.9,
        "margin": 0.0,
        "scale": 1000.0,
        "state_mean": [0.0, 0.0, 0.0],
        "state_std": [1.0, 1.0, 1.0],
        "metrics": {},
    }
    path = tmp_path / "legacy.pt"
    torch.save(payload, path)
    loaded, _ = load_return_capper(path, expected_horizon=10, expected_scale=1000.0)
    assert loaded.model.hidden_dim == 8
    assert loaded.model.state_dim == 3
    assert loaded.horizon == 10


def test_load_legacy_global_json_capper(tmp_path):
    path = tmp_path / "global.json"
    path.write_text('{"cap_value": 1.25, "horizon": 10, "scale": 1000.0}')
    loaded, _ = load_return_capper(path, expected_horizon=10, expected_scale=1000.0)
    assert isinstance(loaded.model, GlobalReturnCap)
    assert np.isclose(loaded.get_cap(np.zeros(3), step=0), 1.25)
