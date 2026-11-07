import numpy as np
import pytest
import torch
import torch.nn as nn

from udt.dt_adapter import DTAdapter
from udt.evaluation import evaluate_with_budget


class MockEnvironment:
    """Mock Gym-like environment for closed-loop evaluation testing."""

    def __init__(self, step_reward: float = 10.0, max_steps: int = 50):
        self.step_reward = step_reward
        self.max_steps = max_steps
        self.step_count = 0
        self.state_dim = 4

    def reset(self):
        self.step_count = 0
        return np.ones(self.state_dim, dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        obs = np.ones(self.state_dim, dtype=np.float32) * float(self.step_count)
        reward = self.step_reward
        done = self.step_count >= self.max_steps
        return obs, reward, done, {}


class MockDTModel(nn.Module):
    """Mock Decision Transformer model implementing get_action."""

    def __init__(self, state_dim=4, act_dim=2):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.linear = nn.Linear(state_dim, act_dim)
        self.call_history = []

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # Record inputs for shape and value inspection
        self.call_history.append({
            "states": states.detach().cpu().numpy(),
            "actions": actions.detach().cpu().numpy(),
            "returns_to_go": returns_to_go.detach().cpu().numpy(),
            "timesteps": timesteps.detach().cpu().numpy(),
        })
        # Deterministic linear projection of latest state
        latest_state = states[:, -1]
        return self.linear(latest_state)[0]


def test_dt_adapter_freezes_model_parameters():
    model = MockDTModel(state_dim=4, act_dim=2)
    assert any(p.requires_grad for p in model.parameters())

    adapter = DTAdapter(model=model, state_dim=4, act_dim=2, max_length=5)
    # Model must be frozen and in eval mode
    assert not model.training
    for p in model.parameters():
        assert not p.requires_grad


def test_dt_adapter_context_window_and_slicing():
    model = MockDTModel(state_dim=4, act_dim=2)
    K = 3
    adapter = DTAdapter(model=model, state_dim=4, act_dim=2, max_length=K)

    # Step 0: history length 1
    a0 = adapter.step(obs=np.ones(4), rtg_token=1.0, step=0)
    assert a0.shape == (2,)
    assert len(model.call_history) == 1
    assert model.call_history[0]["states"].shape[1] == 1
    assert model.call_history[0]["timesteps"][0, -1] == 0

    # Step 1: history length 2
    a1 = adapter.step(obs=np.ones(4) * 2, rtg_token=0.9, step=1)
    assert len(model.call_history) == 2
    assert model.call_history[1]["states"].shape[1] == 2
    assert model.call_history[1]["timesteps"][0, -1] == 1

    # Step 2: history length 3 (= K)
    a2 = adapter.step(obs=np.ones(4) * 3, rtg_token=0.8, step=2)
    assert len(model.call_history) == 3
    assert model.call_history[2]["states"].shape[1] == 3

    # Step 3: history length should be capped at K = 3
    a3 = adapter.step(obs=np.ones(4) * 4, rtg_token=0.7, step=3)
    assert len(model.call_history) == 4
    assert model.call_history[3]["states"].shape[1] == 3
    # Check that oldest state (from step 0) was evicted
    assert np.allclose(model.call_history[3]["states"][0, 0], 2.0)
    assert np.allclose(model.call_history[3]["states"][0, -1], 4.0)
    assert model.call_history[3]["timesteps"][0, -1] == 3


def test_dt_adapter_state_normalization():
    model = MockDTModel(state_dim=2, act_dim=1)
    state_mean = np.array([10.0, 20.0], dtype=np.float32)
    state_std = np.array([2.0, 5.0], dtype=np.float32)

    adapter = DTAdapter(
        model=model,
        state_dim=2,
        act_dim=1,
        max_length=5,
        state_mean=state_mean,
        state_std=state_std,
    )

    raw_obs = np.array([12.0, 30.0], dtype=np.float32)
    adapter.step(obs=raw_obs, rtg_token=1.0, step=0)

    # Normalized obs: (12 - 10)/2 = 1.0, (30 - 20)/5 = 2.0
    passed_state = model.call_history[0]["states"][0, -1]
    expected_norm = np.array([1.0, 2.0], dtype=np.float32)
    np.testing.assert_allclose(passed_state, expected_norm, rtol=1e-5)


def test_dt_adapter_action_clipping():
    model = MockDTModel(state_dim=2, act_dim=2)
    # Set weights to produce large positive and negative actions
    with torch.no_grad():
        model.linear.weight.fill_(100.0)
        model.linear.bias.fill_(100.0)

    adapter = DTAdapter(
        model=model,
        state_dim=2,
        act_dim=2,
        action_low=np.array([-1.0, -2.0]),
        action_high=np.array([1.0, 2.0]),
    )

    action = adapter.step(obs=np.ones(2), rtg_token=1.0, step=0)
    assert np.all(action >= np.array([-1.0, -2.0]))
    assert np.all(action <= np.array([1.0, 2.0]))
    np.testing.assert_allclose(action, [1.0, 2.0])


def test_dt_adapter_commanded_action_override():
    model = MockDTModel(state_dim=2, act_dim=1)
    adapter = DTAdapter(model=model, state_dim=2, act_dim=1, max_length=5)

    # Step 0
    a0 = adapter.step(obs=np.ones(2), rtg_token=1.0, step=0)
    # Module U reranks and commands action 99.0
    adapter.record_commanded_action(np.array([99.0]))

    # Step 1
    adapter.step(obs=np.ones(2), rtg_token=0.9, step=1)
    # The previous action seen by step 1 must be 99.0, not a0!
    actions_seen = model.call_history[1]["actions"]
    assert np.isclose(actions_seen[0, 0, 0], 99.0)


def test_dt_adapter_reset_clears_history():
    model = MockDTModel(state_dim=2, act_dim=1)
    adapter = DTAdapter(model=model, state_dim=2, act_dim=1, max_length=5)

    adapter.step(obs=np.ones(2), rtg_token=1.0, step=0)
    adapter.step(obs=np.ones(2), rtg_token=0.9, step=1)
    assert len(adapter.history_states) == 2

    adapter.reset()
    assert len(adapter.history_states) == 0
    assert len(adapter.history_actions) == 0

    # Step after reset starts fresh at length 1
    adapter.step(obs=np.ones(2), rtg_token=2.0, step=0)
    assert len(adapter.history_states) == 1


def test_dt_adapter_policy_fn_in_evaluate_with_budget():
    model = MockDTModel(state_dim=4, act_dim=1)
    adapter = DTAdapter(model=model, state_dim=4, act_dim=1, max_length=5)
    env = MockEnvironment(step_reward=5.0, max_steps=10)

    adapter.reset()
    result = evaluate_with_budget(
        env=env,
        policy_fn=adapter.policy_fn,
        capper=None,
        initial_target=1000.0,
        horizon=10,
        reward_scale=1000.0,
    )
    assert result.length == 10
    assert len(result.actions) == 10
    assert len(adapter.history_actions) == 10
