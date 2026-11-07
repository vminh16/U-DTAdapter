import numpy as np
import pytest

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


def test_evaluation_observation_noise_applied_at_all_steps():
    env = MockEnvironment(step_reward=1.0, max_steps=5)
    received_obs = []

    def recording_policy(tokens, obs, step):
        received_obs.append(obs.copy())
        return np.zeros(1)

    state_std = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    result = evaluate_with_budget(
        env=env,
        policy_fn=recording_policy,
        initial_target=100.0,
        horizon=5,
        obs_noise_scale=0.1,  # alpha = 0.1
        state_std=state_std,
        rng=42,
    )

    assert result.length == 5
    assert len(received_obs) == 5
    assert len(result.clean_observations) == 5

    # Received observations should differ from clean observations by non-zero noise
    for idx, (clean_o, noisy_o) in enumerate(zip(result.clean_observations, received_obs)):
        diff = noisy_o - clean_o
        assert not np.allclose(diff, 0.0)
        # Check that result.observations contains the noisy observations seen by policy
        assert np.allclose(noisy_o, result.observations[idx])


def test_evaluation_actuator_noise_decoupled_from_commanded_action():
    env = MockEnvironment(step_reward=1.0, max_steps=5)

    # Policy commands a constant action 0.5
    def constant_policy(tokens, obs, step):
        return np.array([0.5], dtype=np.float32)

    action_low = np.array([-1.0], dtype=np.float32)
    action_high = np.array([1.0], dtype=np.float32)

    result = evaluate_with_budget(
        env=env,
        policy_fn=constant_policy,
        initial_target=100.0,
        horizon=5,
        act_noise_scale=0.2,  # kappa = 0.2
        action_low=action_low,
        action_high=action_high,
        rng=123,
    )

    assert len(result.actions) == 5
    assert len(result.executed_actions) == 5

    # Commanded action must remain strictly 0.5
    for a_cmd in result.actions:
        assert np.isclose(a_cmd[0], 0.5)

    # Executed action must differ due to noise, but strictly respect bounds
    for a_exec in result.executed_actions:
        assert -1.0 <= a_exec[0] <= 1.0
        assert not np.isclose(a_exec[0], 0.5)


def test_evaluation_tracks_terminated_vs_truncated():
    # Environment that finishes at step 3 when horizon is 10 (early unhealthy termination)
    env_early = MockEnvironment(max_steps=3)
    res_early = evaluate_with_budget(
        env=env_early,
        policy_fn=lambda t, o, s: np.zeros(1),
        horizon=10,
    )
    assert res_early.length == 3
    assert res_early.terminated is True
    assert res_early.truncated is False

    # Environment that reaches full horizon 5
    env_full = MockEnvironment(max_steps=10)
    res_full = evaluate_with_budget(
        env=env_full,
        policy_fn=lambda t, o, s: np.zeros(1),
        horizon=5,
    )
    assert res_full.length == 5
    assert res_full.truncated is True
