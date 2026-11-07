import numpy as np
import pytest
import torch
import torch.nn.functional as F

from udt.return_cap import (
    ExpectileLoss,
    GlobalReturnCap,
    ReturnCapMLP,
    ReturnCapper,
    compute_global_expectile,
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

    # Two-point distribution {0, 100} with equal probability:
    # Expectile equation: (1 - tau) * c = tau * (100 - c) => c = 100 * tau
    two_point_data = np.array([0.0, 100.0])
    e_09 = compute_global_expectile(two_point_data, tau=0.9)
    assert np.isclose(e_09, 90.0, atol=1e-3)


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

