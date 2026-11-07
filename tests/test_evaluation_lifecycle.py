"""Episode isolation, RNG stream separation, and telemetry-copy tests."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from udt.dt_adapter import DTAdapter
from udt.evaluation import evaluate_with_budget


class ReusingBufferEnv:
    """Gym-like env that reuses one observation buffer like real wrappers may."""

    def __init__(self, max_steps=4, state_dim=2):
        self.max_steps = max_steps
        self.state_dim = state_dim
        self.step_count = 0
        self.buffer = np.zeros(state_dim, dtype=np.float32)
        self.reset_seeds = []
        self.action_space = type("ActionSpace", (), {"shape": (1,)})()

    def reset(self, seed=None):
        if seed is not None:
            self.reset_seeds.append(int(seed))
        self.step_count = 0
        self.buffer[:] = 1.0
        return self.buffer

    def step(self, action):
        self.step_count += 1
        self.buffer[:] = float(self.step_count)
        done = self.step_count >= self.max_steps
        return self.buffer, 1.0, done, {}


class LegacySeedEnv:
    def __init__(self, max_steps=2):
        self.max_steps = max_steps
        self.step_count = 0
        self.seeded = []

    def seed(self, seed=None):
        self.seeded.append(int(seed))
        return [seed]

    def reset(self):
        self.step_count = 0
        return np.ones(2, dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        return np.ones(2, dtype=np.float32), 1.0, self.step_count >= self.max_steps, {}


class UnseedableEnv:
    def __init__(self, max_steps=2):
        self.max_steps = max_steps
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return np.ones(2, dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        return np.ones(2, dtype=np.float32), 1.0, self.step_count >= self.max_steps, {}


class MockDT(nn.Module):
    def __init__(self, act_dim=1):
        super().__init__()
        self.act_dim = act_dim

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        return torch.zeros(self.act_dim)


def constant_policy(tokens, obs, step):
    return np.array([0.25], dtype=np.float32)


def test_two_consecutive_episodes_do_not_leak_dt_history():
    env = ReusingBufferEnv(max_steps=4, state_dim=2)
    adapter = DTAdapter(MockDT(), state_dim=2, act_dim=1, max_length=10)

    first = evaluate_with_budget(
        env=env, policy_fn=adapter.policy_fn, horizon=4, seed=11,
    )
    assert first.length == 4
    assert len(adapter.history_states) == 4

    second = evaluate_with_budget(
        env=env, policy_fn=adapter.policy_fn, horizon=4, seed=12,
    )
    assert second.length == 4
    # History was cleared, not appended to the previous episode.
    assert len(adapter.history_states) == 4
    assert len(adapter.history_actions) == 4
    np.testing.assert_allclose(adapter.history_actions[0], np.zeros(1))


def test_telemetry_copies_reused_environment_buffers():
    env = ReusingBufferEnv(max_steps=4, state_dim=2)
    result = evaluate_with_budget(
        env=env, policy_fn=constant_policy, horizon=4, seed=3,
    )
    # Mutate the environment-owned buffer after the rollout.
    env.buffer[:] = 999.0
    for obs in result.clean_observations:
        assert not np.any(obs == 999.0)
    np.testing.assert_allclose([o[0] for o in result.clean_observations], [1, 1, 2, 3])


def test_independent_streams_keep_actuator_noise_identical_when_obs_noise_changes():
    action_low = np.array([-1.0], dtype=np.float32)
    action_high = np.array([1.0], dtype=np.float32)

    clean = evaluate_with_budget(
        env=ReusingBufferEnv(max_steps=6, state_dim=2),
        policy_fn=constant_policy,
        horizon=6,
        act_noise_scale=0.2,
        action_low=action_low,
        action_high=action_high,
        seed=77,
    )
    noisy = evaluate_with_budget(
        env=ReusingBufferEnv(max_steps=6, state_dim=2),
        policy_fn=constant_policy,
        horizon=6,
        obs_noise_scale=0.3,
        state_std=np.ones(2, dtype=np.float32),
        act_noise_scale=0.2,
        action_low=action_low,
        action_high=action_high,
        seed=77,
    )

    assert len(clean.executed_actions) == len(noisy.executed_actions)
    for a, b in zip(clean.executed_actions, noisy.executed_actions):
        np.testing.assert_allclose(a, b, rtol=0, atol=0)
    # And observation noise nevertheless differed.
    assert not np.allclose(clean.observations[0], noisy.observations[0])


def test_seed_mapping_and_gymnasium_env_seeding_are_recorded():
    env = ReusingBufferEnv(max_steps=2, state_dim=2)
    result = evaluate_with_budget(env=env, policy_fn=constant_policy, horizon=2, seed=123)
    assert result.env_seeding == "reset_seed"
    assert env.reset_seeds == [result.seed_mapping["streams"]["environment"]["seed"]]
    assert set(result.seed_mapping["streams"]) == {
        "environment", "observation_noise", "actuator_noise", "candidate_proposal"
    }


def test_legacy_env_seed_and_unsupported_seed_are_reported():
    legacy = LegacySeedEnv()
    result = evaluate_with_budget(env=legacy, policy_fn=constant_policy, horizon=2, seed=5)
    assert result.env_seeding == "legacy_seed"
    assert legacy.seeded == [result.seed_mapping["streams"]["environment"]["seed"]]

    unseedable = UnseedableEnv()
    result = evaluate_with_budget(env=unseedable, policy_fn=constant_policy, horizon=2, seed=5)
    assert result.env_seeding == "unsupported"
    assert any("not controlled" in w for w in result.warnings)
    with pytest.raises(RuntimeError, match="not controlled"):
        evaluate_with_budget(
            env=UnseedableEnv(), policy_fn=constant_policy, horizon=2,
            seed=5, strict_env_seed=True,
        )


def test_policy_action_shape_bounds_and_finiteness_are_validated():
    with pytest.raises(ValueError, match="outside environment bounds"):
        evaluate_with_budget(
            env=ReusingBufferEnv(max_steps=1, state_dim=2),
            policy_fn=lambda t, o, s: np.array([2.0], dtype=np.float32),
            horizon=1,
            action_low=np.array([-1.0]),
            action_high=np.array([1.0]),
        )
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_with_budget(
            env=ReusingBufferEnv(max_steps=1, state_dim=2),
            policy_fn=lambda t, o, s: np.array([np.nan], dtype=np.float32),
            horizon=1,
        )
    with pytest.raises(ValueError, match="does not match environment action shape"):
        evaluate_with_budget(
            env=ReusingBufferEnv(max_steps=1, state_dim=2),
            policy_fn=lambda t, o, s: np.array([0.0, 0.0], dtype=np.float32),
            horizon=1,
        )
