"""Checkpoint, normalization-coordinate, and U-only continuation tests."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from udt.dynamics import (
    DynamicsEnsemble,
    ModuleUController,
    load_dynamics_checkpoint,
    save_dynamics_checkpoint,
    train_dynamics_ensemble,
)
from udt.dt_adapter import DTAdapter
from udt.return_cap import ReturnCapper


class MockDTModel(nn.Module):
    def __init__(self, state_dim=2, act_dim=1):
        super().__init__()
        self.act_dim = act_dim

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        return torch.zeros(self.act_dim)


class SumModel(nn.Module):
    def forward(self, states, t_rel):
        return states.sum(dim=-1, keepdim=True)


def make_dataset(state_dim=2, act_dim=1):
    n_per_episode = 4
    n_episodes = 4
    n = n_per_episode * n_episodes
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(n, state_dim)).astype(np.float32)
    return {
        "observations": obs,
        "next_observations": (obs + 0.1).astype(np.float32),
        "actions": rng.normal(size=(n, act_dim)).astype(np.float32),
        "rewards": rng.normal(size=n).astype(np.float32),
        "terminals": np.zeros(n, dtype=bool),
        "timesteps": np.tile(np.arange(n_per_episode), n_episodes).astype(np.int64),
        "episode_ids": np.repeat(np.arange(4), n_per_episode),
        "is_validation": np.repeat([False, True], n // 2),
        "next_observations_valid": np.ones(n, dtype=bool),
    }


def trained_ensemble():
    data = make_dataset()
    ensemble, metrics = train_dynamics_ensemble(
        data,
        horizon=10,
        num_models=2,
        hidden_dim=8,
        num_layers=2,
        epochs=2,
        batch_size=8,
        seed=3,
    )
    return ensemble, metrics


def test_checkpoint_round_trip_preserves_predictions_and_metadata(tmp_path):
    ensemble, metrics = trained_ensemble()
    path = save_dynamics_checkpoint(
        tmp_path / "dynamics.pt",
        ensemble,
        metrics=metrics,
        dataset_provenance={"dataset_id": "unit-test", "source_sha256": "a" * 64},
    )
    loaded, meta = load_dynamics_checkpoint(
        path,
        expected_state_dim=2,
        expected_act_dim=1,
        expected_horizon=10,
        expected_reward_scale=1000.0,
    )

    np.testing.assert_allclose(loaded.state_mean, ensemble.state_mean)
    np.testing.assert_allclose(loaded.state_std, ensemble.state_std)
    assert meta["normalization"]["source"] == "dynamics_training_split"
    assert meta["dataset_provenance"]["source_sha256"] == "a" * 64
    assert len(meta["bootstrap"]) == 2
    assert meta["training_config"]["w_reward"] == 1.0

    z = np.zeros(2, dtype=np.float32)
    actions = np.zeros((3, 1), dtype=np.float32)
    a = ensemble.predict(z, actions, 0.2)
    b = loaded.predict(z, actions, 0.2)
    for left, right in zip(a, b):
        np.testing.assert_allclose(left, right, rtol=0, atol=0)


def test_checkpoint_rejects_incompatible_configuration(tmp_path):
    ensemble, metrics = trained_ensemble()
    path = save_dynamics_checkpoint(tmp_path / "dynamics.pt", ensemble, metrics=metrics)

    with pytest.raises(ValueError, match="state_dim"):
        load_dynamics_checkpoint(path, expected_state_dim=3)
    with pytest.raises(ValueError, match="horizon"):
        load_dynamics_checkpoint(path, expected_horizon=11)
    with pytest.raises(ValueError, match="reward_scale"):
        load_dynamics_checkpoint(path, expected_reward_scale=500.0)


class SpyReturnModel:
    def __init__(self, value: float):
        self.value = value
        self.seen_states = None
        self.scale = 1000.0
        self.horizon = 100

    def get_cap_batch(self, states, step):
        self.seen_states = np.asarray(states).copy()
        return np.full(len(states), self.value, dtype=np.float32)


def test_controller_uses_dynamics_normalization_not_dt_normalization(monkeypatch):
    ensemble = DynamicsEnsemble(
        state_dim=2,
        act_dim=1,
        num_models=2,
        hidden_dim=8,
        state_mean=np.array([0.0, 0.0], dtype=np.float32),
        state_std=np.array([2.0, 4.0], dtype=np.float32),
        horizon=100,
        reward_scale=1000.0,
    )
    # Predict a known normalized next state for every head and candidate.
    def fake_predict(z, actions, t_rel):
        n = actions.shape[0]
        return (
            np.ones((2, n, 2), dtype=np.float32),
            np.zeros((2, n), dtype=np.float32),
            np.zeros((2, n), dtype=np.float32),
        )

    monkeypatch.setattr(ensemble, "predict", fake_predict)

    dt = DTAdapter(
        MockDTModel(), state_dim=2, act_dim=1,
        state_mean=np.array([100.0, 100.0], dtype=np.float32),
        state_std=np.array([0.5, 0.5], dtype=np.float32),
    )
    spy = SpyReturnModel(1.0)
    controller = ModuleUController(
        dt_adapter=dt,
        ensemble=ensemble,
        return_model=spy,
        num_candidates=2,
        horizon=100,
        action_low=np.array([-1.0]),
        action_high=np.array([1.0]),
        beta=0.0,
    )
    _, info = controller.get_action(obs=np.array([0.0, 0.0]), rtg_token=1.0, step=0)

    # dynamics unnormalization: [1,1] * [2,4] + [0,0] = [2,4]
    np.testing.assert_allclose(spy.seen_states, np.tile([2.0, 4.0], (4, 1)))
    # terminal probability 0 -> continuation is exactly the return value.
    np.testing.assert_allclose(info["next_returns"], np.ones((2, 2)))


def test_dt_and_dynamics_have_different_normalization_without_double_scaling():
    dt_mean = np.array([10.0, 20.0], dtype=np.float32)
    dt_std = np.array([2.0, 5.0], dtype=np.float32)
    dyn_mean = np.array([0.0, 0.0], dtype=np.float32)
    dyn_std = np.array([2.0, 4.0], dtype=np.float32)
    ret_mean = np.array([1.0, 3.0], dtype=np.float32)
    ret_std = np.array([1.0, 2.0], dtype=np.float32)

    ensemble = DynamicsEnsemble(
        state_dim=2, act_dim=1, num_models=1, hidden_dim=8,
        state_mean=dyn_mean, state_std=dyn_std, horizon=100, reward_scale=1000.0,
    )
    for p in ensemble.models[0].parameters():
        p.data.zero_()

    def fake_predict(z, actions, t_rel):
        n = actions.shape[0]
        return (
            np.ones((1, n, 2), dtype=np.float32),
            np.zeros((1, n), dtype=np.float32),
            np.zeros((1, n), dtype=np.float32),
        )

    ensemble.predict = fake_predict
    dt = DTAdapter(MockDTModel(), state_dim=2, act_dim=1,
                   state_mean=dt_mean, state_std=dt_std)
    return_model = ReturnCapper(
        SumModel(), horizon=100, scale=1000.0,
        state_mean=ret_mean, state_std=ret_std,
    )
    controller = ModuleUController(
        dt_adapter=dt, ensemble=ensemble, return_model=return_model,
        num_candidates=2, horizon=100,
        action_low=np.array([-1.0]), action_high=np.array([1.0]), beta=0.0,
    )
    _, info = controller.get_action(obs=np.array([0.0, 0.0]), rtg_token=1.0, step=0)

    # dynamics-normalized [1,1] -> raw [2,4] -> return-normalized [1,0.5] -> sum 1.5
    np.testing.assert_allclose(info["next_returns"], np.full((1, 2), 1.5))
    # Sanity: using DT statistics here would have produced a very different value.
    wrong_raw = np.array([1.0, 1.0]) * dt_std + dt_mean
    wrong_value = float(np.sum((wrong_raw - ret_mean) / ret_std))
    assert not np.isclose(wrong_value, 1.5)


def test_u_only_requires_return_model_unless_named_ablation():
    dt = DTAdapter(MockDTModel(), state_dim=2, act_dim=1)
    ensemble = DynamicsEnsemble(
        state_dim=2, act_dim=1, num_models=1, hidden_dim=8,
        state_mean=np.zeros(2, dtype=np.float32),
        state_std=np.ones(2, dtype=np.float32),
        horizon=10, reward_scale=1000.0,
    )
    with pytest.raises(ValueError, match="return model g_psi"):
        ModuleUController(
            dt_adapter=dt, ensemble=ensemble, num_candidates=2, horizon=10,
            action_low=np.array([-1.0]), action_high=np.array([1.0]),
        )

    # Explicitly named immediate-reward ablation is allowed and reported.
    ablation = ModuleUController(
        dt_adapter=dt, ensemble=ensemble, scorer_mode="immediate_reward",
        num_candidates=2, horizon=10,
        action_low=np.array([-1.0]), action_high=np.array([1.0]),
    )
    _, info = ablation.get_action(obs=np.zeros(2), rtg_token=1.0, step=0)
    assert info["scorer_mode"] == "immediate_reward"
    np.testing.assert_allclose(info["next_returns"], 0.0)


def test_u_only_without_r_cap_still_computes_continuation():
    dt = DTAdapter(MockDTModel(), state_dim=2, act_dim=1)
    ensemble = DynamicsEnsemble(
        state_dim=2, act_dim=1, num_models=2, hidden_dim=8,
        state_mean=np.zeros(2, dtype=np.float32),
        state_std=np.ones(2, dtype=np.float32),
        horizon=10, reward_scale=1000.0,
    )
    for m in ensemble.models:
        for p in m.parameters():
            p.data.zero_()
    spy = SpyReturnModel(2.0)
    spy.horizon = 10
    controller = ModuleUController(
        dt_adapter=dt, ensemble=ensemble, return_model=spy,
        num_candidates=3, horizon=10,
        action_low=np.array([-1.0]), action_high=np.array([1.0]), beta=0.0,
    )

    # No R cap object is passed. Continuation is still evaluated and used.
    _, info = controller.get_action(obs=np.zeros(2), rtg_token=1.0, step=0)
    np.testing.assert_allclose(info["next_returns"], 2.0)
    # zero weights -> terminal probability 0.5 -> continuation = 0.5 * 2.0
    np.testing.assert_allclose(info["candidate_scores"], 1.0)

    # At t+1 >= H continuation must be exactly zero.
    _, horizon_info = controller.get_action(obs=np.zeros(2), rtg_token=1.0, step=9)
    assert horizon_info["at_horizon"] is True
    np.testing.assert_allclose(horizon_info["next_returns"], 0.0)
    np.testing.assert_allclose(horizon_info["candidate_scores"], 0.0)
