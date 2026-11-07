import numpy as np
import pytest

from udt.intervention import ReturnBudget
from udt.return_cap import GlobalReturnCap, ReturnCapMLP, ReturnCapper

from udt.evaluation import evaluate_with_budget, RolloutResult


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


def mock_policy(history_tokens, state, step):
    # Dummy policy returning action matching token
    return np.array([history_tokens[-1]], dtype=np.float32)


def test_evaluate_with_budget_unclipped_baseline_b0():
    env = MockEnvironment(step_reward=10.0, max_steps=20)
    # Baseline B0: no capper
    result = evaluate_with_budget(
        env=env,
        policy_fn=mock_policy,
        capper=None,
        initial_target=2000.0,
        horizon=20,
        reward_scale=1000.0,
    )

    assert isinstance(result, RolloutResult)
    assert result.length == 20
    assert result.total_reward == 200.0
    assert result.clipping_frequency == 0.0
    # With scale 1000, 2000.0 raw -> initial raw budget is 2.0
    # Over 20 steps of 10.0 reward, total reward 200 -> raw ends at 2.0 - 0.2 = 1.8
    assert np.isclose(result.raw_budgets[0], 2.0)
    assert np.isclose(result.final_raw_budget, 1.8)
    assert result.tokens_used == result.raw_budgets



def test_evaluate_with_budget_inflated_target_clipped_b2_global():
    env = MockEnvironment(step_reward=10.0, max_steps=20)
    # Baseline B2: Global scalar cap set to 1.0 (equivalent to 1000.0 unscaled return)
    capper = ReturnCapper(
        model=GlobalReturnCap(cap_value=1.0),
        horizon=20,
        margin=0.0,
        scale=1000.0,
    )

    # Initial target inflated to 5000.0 (raw budget = 5.0)
    result = evaluate_with_budget(
        env=env,
        policy_fn=mock_policy,
        capper=capper,
        initial_target=5000.0,
        horizon=20,
        reward_scale=1000.0,
    )

    assert isinstance(result, RolloutResult)
    # Every token must be clipped down to cap = 1.0
    assert result.clipping_frequency == 1.0
    assert all(np.isclose(t, 1.0) for t in result.tokens_used)
    # Raw budget must NOT be crushed; it starts at 5.0 and ends at 4.8
    assert np.isclose(result.raw_budgets[0], 5.0)
    assert np.isclose(result.final_raw_budget, 4.8)


def test_evaluate_with_budget_neural_mlp_b3():
    env = MockEnvironment(step_reward=5.0, max_steps=10)
    # Baseline B3: State-time neural model
    mlp = ReturnCapMLP(state_dim=4, hidden_dim=32, num_layers=2, mode="state_time")
    # Set weights so output is roughly 0.5
    for p in mlp.parameters():
        p.data.zero_()
    # Bias of final layer = 0.5
    list(mlp.parameters())[-1].data.fill_(0.5)

    capper = ReturnCapper(
        model=mlp,
        horizon=10,
        margin=0.1,  # effective cap = 0.6
        scale=1000.0,
    )

    result = evaluate_with_budget(
        env=env,
        policy_fn=mock_policy,
        capper=capper,
        initial_target=2000.0,  # raw = 2.0
        horizon=10,
        reward_scale=1000.0,
    )

    assert isinstance(result, RolloutResult)
    assert result.length == 10
    assert result.clipping_frequency == 1.0
    # Every token should be capped at 0.5 + 0.1 = 0.6
    assert np.allclose(result.tokens_used, 0.6)
    # Raw budget starts at 2.0 and ends at 2.0 - (10 * 5) / 1000 = 1.95
    assert np.isclose(result.raw_budgets[0], 2.0)
    assert np.isclose(result.final_raw_budget, 1.95)


def test_evaluate_with_budget_rejects_scale_mismatch():
    env = MockEnvironment()
    capper = ReturnCapper(model=GlobalReturnCap(1.0), horizon=20, scale=500.0)
    with pytest.raises(ValueError, match="Reward scale mismatch"):
        evaluate_with_budget(env, mock_policy, capper=capper, horizon=20, reward_scale=1000.0)


def test_evaluate_with_budget_rejects_horizon_mismatch():
    env = MockEnvironment()
    capper = ReturnCapper(model=GlobalReturnCap(1.0), horizon=50, scale=1000.0)
    with pytest.raises(ValueError, match="Horizon mismatch"):
        evaluate_with_budget(env, mock_policy, capper=capper, horizon=20, reward_scale=1000.0)


def test_evaluate_with_budget_rejects_non_positive_horizon_or_scale():
    env = MockEnvironment()
    with pytest.raises(ValueError, match="horizon must be positive"):
        evaluate_with_budget(env, mock_policy, horizon=0)
    with pytest.raises(ValueError, match="reward_scale must be positive"):
        evaluate_with_budget(env, mock_policy, reward_scale=-10.0)


def test_capper_step_rejects_scale_mismatch():
    capper = ReturnCapper(model=GlobalReturnCap(1.0), horizon=100, scale=1000.0)
    budget = ReturnBudget(raw=5.0, reward_scale=500.0)
    with pytest.raises(ValueError, match="Scale mismatch"):
        capper.step(budget, state=np.zeros(4), step=0)


