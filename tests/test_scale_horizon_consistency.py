import numpy as np
import pytest
import torch
import torch.nn as nn

from udt.dt_adapter import DTAdapter
from udt.dynamics import DynamicsEnsemble, ModuleUController
from udt.experiment import run_variant
from udt.return_cap import GlobalReturnCap, ReturnCapper


class DummyDT(nn.Module):
    def __init__(self, act_dim: int = 1):
        super().__init__()
        self.act_dim = act_dim

    def get_action(self, *args, **kwargs):
        return torch.zeros(self.act_dim)


class DummyEnv:
    def __init__(self, max_steps: int = 2):
        self.max_steps = max_steps
        self.step_count = 0

    def reset(self, seed=None):
        self.step_count = 0
        return np.ones(2, dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        return np.ones(2, dtype=np.float32), 1.0, self.step_count >= self.max_steps, {}


def _make_adapter():
    return DTAdapter(DummyDT(1), state_dim=2, act_dim=1)


def test_ensemble_horizon_differs_from_controller_horizon_rejected():
    """Ensemble H=1000, controller H=500 -> reject."""
    adapter = _make_adapter()
    ensemble = DynamicsEnsemble(
        state_dim=2, act_dim=1, num_models=2, hidden_dim=16,
        reward_scale=1000.0, horizon=1000,
    )
    return_model = ReturnCapper(
        model=GlobalReturnCap(1.0), horizon=500, scale=1000.0,
    )
    with pytest.raises(ValueError, match=r"[Hh]orizon"):
        ModuleUController(
            dt_adapter=adapter,
            ensemble=ensemble,
            return_model=return_model,
            horizon=500,
        )


def test_dynamics_scale_differs_from_b4_rollout_scale_rejected():
    """Dynamics c=1000, B4 rollout c=10 -> reject."""
    adapter = _make_adapter()
    ensemble = DynamicsEnsemble(
        state_dim=2, act_dim=1, num_models=2, hidden_dim=16,
        reward_scale=1000.0, horizon=100,
    )
    return_model = ReturnCapper(
        model=GlobalReturnCap(1.0), horizon=100, scale=1000.0,
    )
    with pytest.raises(ValueError, match=r"[Ss]cale|[Rr]eward"):
        run_variant(
            "B4",
            DummyEnv(),
            adapter,
            ensemble=ensemble,
            return_model=return_model,
            horizon=100,
            reward_scale=10.0,
        )


def test_return_model_scale_differs_from_ensemble_rejected():
    """Return model scale differs from ensemble -> reject."""
    adapter = _make_adapter()
    ensemble = DynamicsEnsemble(
        state_dim=2, act_dim=1, num_models=2, hidden_dim=16,
        reward_scale=1000.0, horizon=100,
    )
    return_model = ReturnCapper(
        model=GlobalReturnCap(1.0), horizon=100, scale=500.0,
    )
    with pytest.raises(ValueError, match=r"[Ss]cale|[Rr]eward"):
        ModuleUController(
            dt_adapter=adapter,
            ensemble=ensemble,
            return_model=return_model,
            horizon=100,
        )


def test_b5_capper_scale_differs_from_rollout_rejected():
    """B5 capper scale differs from rollout -> reject."""
    adapter = _make_adapter()
    ensemble = DynamicsEnsemble(
        state_dim=2, act_dim=1, num_models=2, hidden_dim=16,
        reward_scale=1000.0, horizon=100,
    )
    return_model = ReturnCapper(
        model=GlobalReturnCap(1.0), horizon=100, scale=1000.0,
    )
    capper = ReturnCapper(
        model=GlobalReturnCap(1.0), horizon=100, scale=500.0,
    )
    with pytest.raises(ValueError, match=r"[Ss]cale|[Rr]eward"):
        run_variant(
            "B5",
            DummyEnv(),
            adapter,
            capper=capper,
            ensemble=ensemble,
            return_model=return_model,
            horizon=100,
            reward_scale=1000.0,
        )


def test_matching_configuration_passes():
    """Matching scale and horizon passes across all components in B5."""
    adapter = _make_adapter()
    ensemble = DynamicsEnsemble(
        state_dim=2, act_dim=1, num_models=2, hidden_dim=16,
        reward_scale=1000.0, horizon=100,
    )
    return_model = ReturnCapper(
        model=GlobalReturnCap(1.0), horizon=100, scale=1000.0,
    )
    capper = ReturnCapper(
        model=GlobalReturnCap(1.0), horizon=100, scale=1000.0,
    )
    run = run_variant(
        "B5",
        DummyEnv(),
        adapter,
        capper=capper,
        ensemble=ensemble,
        return_model=return_model,
        horizon=100,
        reward_scale=1000.0,
        num_candidates=3,
        seed=1,
    )
    assert run.rollout.length == 2


def test_b0_does_not_require_ensemble_or_return_model():
    """B0 does not require ensemble or return model."""
    adapter = _make_adapter()
    run = run_variant(
        "B0",
        DummyEnv(),
        adapter,
        horizon=100,
        reward_scale=1000.0,
        seed=1,
    )
    assert run.rollout.length == 2
