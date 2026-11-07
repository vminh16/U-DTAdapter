import numpy as np
import pytest
import torch
import torch.nn as nn

from udt.dt_adapter import DTAdapter
from udt.return_cap import GlobalReturnCap, ReturnCapper
from udt.dynamics import DynamicsMLP, DynamicsEnsemble, train_dynamics_ensemble, ModuleUController


class MockDTModel(nn.Module):
    def __init__(self, state_dim=4, act_dim=2):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # Always output zeros for test baseline
        return torch.zeros(self.act_dim)


def test_dynamics_mlp_forward_shapes():
    state_dim = 4
    act_dim = 2
    mlp = DynamicsMLP(state_dim=state_dim, act_dim=act_dim, hidden_dim=32, num_layers=2)

    batch_size = 5
    z = torch.randn(batch_size, state_dim)
    a = torch.randn(batch_size, act_dim)
    t_rel = torch.rand(batch_size, 1)

    delta_z, reward, p_term = mlp(z, a, t_rel)
    assert delta_z.shape == (batch_size, state_dim)
    assert reward.shape == (batch_size, 1)
    assert p_term.shape == (batch_size, 1)
    # p_term must be valid probabilities in [0, 1]
    assert torch.all(p_term >= 0.0) and torch.all(p_term <= 1.0)


def test_dynamics_ensemble_predict():
    state_dim = 3
    act_dim = 2
    M = 3
    ensemble = DynamicsEnsemble(
        state_dim=state_dim,
        act_dim=act_dim,
        num_models=M,
        hidden_dim=16,
        num_layers=2,
    )

    N = 4
    z = np.zeros(state_dim, dtype=np.float32)
    actions = np.ones((N, act_dim), dtype=np.float32)
    t_rel = 0.2

    next_states, rewards, terminals = ensemble.predict(z, actions, t_rel)
    assert next_states.shape == (M, N, state_dim)
    assert rewards.shape == (M, N)
    assert terminals.shape == (M, N)
    assert np.all(terminals >= 0.0) and np.all(terminals <= 1.0)


def test_train_dynamics_ensemble():
    n_train = 40
    n_val = 20
    n_total = n_train + n_val
    state_dim = 2
    act_dim = 1

    obs = np.ones((n_total, state_dim), dtype=np.float32)
    next_obs = np.ones((n_total, state_dim), dtype=np.float32) * 1.5
    actions = np.zeros((n_total, act_dim), dtype=np.float32)
    rewards = np.ones(n_total, dtype=np.float32) * 0.5
    terminals = np.zeros(n_total, dtype=bool)
    timesteps = np.zeros(n_total, dtype=np.int64)
    is_val = np.array([False] * n_train + [True] * n_val)
    episode_ids = np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10
                           + [10] * 10 + [11] * 10, dtype=np.int64)

    dataset = {
        "observations": obs,
        "next_observations": next_obs,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timesteps": timesteps,
        "episode_ids": episode_ids,
        "is_validation": is_val,
    }

    ensemble, metrics = train_dynamics_ensemble(
        dataset=dataset,
        horizon=100,
        num_models=2,
        hidden_dim=16,
        num_layers=2,
        epochs=3,
        batch_size=16,
        seed=42,
    )

    assert isinstance(ensemble, DynamicsEnsemble)
    assert len(ensemble.models) == 2
    assert "models" in metrics
    assert len(metrics["models"]) == 2
    for m in metrics["models"]:
        assert "best_val_loss" in m
        assert "train_loss" in m
        assert len(m["train_loss"]) == 3


def test_module_u_controller_n1_matches_original_action():
    dt_model = MockDTModel(state_dim=4, act_dim=2)
    dt_adapter = DTAdapter(model=dt_model, state_dim=4, act_dim=2)
    ensemble = DynamicsEnsemble(state_dim=4, act_dim=2, num_models=2, hidden_dim=16)
    capper = ReturnCapper(model=GlobalReturnCap(1.0), horizon=100)

    controller = ModuleUController(
        dt_adapter=dt_adapter,
        ensemble=ensemble,
        capper=capper,
        num_candidates=1,  # N = 1
        action_low=np.array([-1.0, -1.0]),
        action_high=np.array([1.0, 1.0]),
    )

    obs = np.ones(4, dtype=np.float32)
    action, info = controller.get_action(obs=obs, rtg_token=1.0, step=0)
    # With N=1, strictly returns original action (zeros)
    np.testing.assert_allclose(action, [0.0, 0.0])
    assert info["candidate_count"] == 1
    assert info["selected_index"] == 0


def test_module_u_controller_tie_breaking_keeps_original():
    dt_model = MockDTModel(state_dim=2, act_dim=1)
    dt_adapter = DTAdapter(model=dt_model, state_dim=2, act_dim=1)
    ensemble = DynamicsEnsemble(state_dim=2, act_dim=1, num_models=2, hidden_dim=16)
    capper = ReturnCapper(model=GlobalReturnCap(1.0), horizon=100)

    # Force ensemble to predict constant outputs across all candidates
    for m in ensemble.models:
        for p in m.parameters():
            p.data.zero_()

    controller = ModuleUController(
        dt_adapter=dt_adapter,
        ensemble=ensemble,
        capper=capper,
        num_candidates=5,
        action_low=np.array([-1.0]),
        action_high=np.array([1.0]),
        seed=123,
    )

    action, info = controller.get_action(obs=np.ones(2), rtg_token=1.0, step=0)
    # When all scores are tied, original action index 0 is strictly chosen
    assert info["selected_index"] == 0
    np.testing.assert_allclose(action, [0.0])


def test_module_u_controller_horizon_boundary_zero_continuation():
    dt_model = MockDTModel(state_dim=2, act_dim=1)
    dt_adapter = DTAdapter(model=dt_model, state_dim=2, act_dim=1)
    ensemble = DynamicsEnsemble(state_dim=2, act_dim=1, num_models=2, hidden_dim=16)

    # Global return capper with very high cap
    capper = ReturnCapper(model=GlobalReturnCap(999.0), horizon=10)

    controller = ModuleUController(
        dt_adapter=dt_adapter,
        ensemble=ensemble,
        capper=capper,
        horizon=10,
        num_candidates=3,
        action_low=np.array([-1.0]),
        action_high=np.array([1.0]),
        beta=0.0,
    )

    # At step t = 9 (t + 1 == 10 == H): continuation must be 0!
    action, info = controller.get_action(obs=np.ones(2), rtg_token=1.0, step=9)
    assert info["at_horizon"] is True
    # Scores must reflect only immediate reward, not continuation 999.0
    for score in info["candidate_scores"]:
        assert score < 100.0


def test_module_u_controller_penalty_penalizes_high_disagreement():
    dt_model = MockDTModel(state_dim=2, act_dim=1)
    dt_adapter = DTAdapter(model=dt_model, state_dim=2, act_dim=1)
    ensemble = DynamicsEnsemble(state_dim=2, act_dim=1, num_models=2, hidden_dim=16)
    capper = ReturnCapper(model=GlobalReturnCap(0.0), horizon=10)

    # Controller with beta = 0 vs beta = 10
    controller_b0 = ModuleUController(
        dt_adapter=dt_adapter,
        ensemble=ensemble,
        capper=capper,
        num_candidates=4,
        action_low=np.array([-1.0]),
        action_high=np.array([1.0]),
        beta=0.0,
        seed=42,
    )
    _, info_b0 = controller_b0.get_action(obs=np.ones(2), rtg_token=1.0, step=0)

    controller_b10 = ModuleUController(
        dt_adapter=dt_adapter,
        ensemble=ensemble,
        capper=capper,
        num_candidates=4,
        action_low=np.array([-1.0]),
        action_high=np.array([1.0]),
        beta=10.0,
        seed=42,
    )
    _, info_b10 = controller_b10.get_action(obs=np.ones(2), rtg_token=1.0, step=0)

    # Candidates with disagreement > 0 must have strictly lower score with beta=10 than beta=0
    disagreements = info_b0["disagreements"]
    scores_b0 = info_b0["candidate_scores"]
    scores_b10 = info_b10["candidate_scores"]
    for i in range(len(disagreements)):
        if disagreements[i] > 1e-4:
            assert scores_b10[i] < scores_b0[i]
