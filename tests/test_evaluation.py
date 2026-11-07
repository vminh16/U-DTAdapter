import numpy as np
import pytest

from udt.return_cap import GlobalReturnCap, ReturnCapper
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



def test_evaluate_with_budget_inflated_target_clipped_b3():
    env = MockEnvironment(step_reward=10.0, max_steps=20)
    # Cap set to 1.0 (equivalent to 1000.0 unscaled return)
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

