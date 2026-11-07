import numpy as np
import pytest
import torch
import torch.nn.functional as F

from udt.return_cap import ExpectileLoss, GlobalReturnCap, compute_global_expectile


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
