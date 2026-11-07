import numpy as np
import pytest
import torch
import torch.nn as nn

from udt.dt_adapter import DTAdapter
from udt.dynamics import DynamicsEnsemble, ModuleUController
from udt.evaluation import evaluate_with_budget
from udt.experiment import run_variant
from udt.return_cap import GlobalReturnCap, ReturnCapper


class DummyDT(nn.Module):
    def __init__(self, act_dim: int = 1):
        super().__init__()
        self.act_dim = act_dim

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        return torch.full((self.act_dim,), 0.2, dtype=torch.float32)


class StepEnv:
    def __init__(self, max_steps: int = 4, state_dim: int = 2):
        self.max_steps = max_steps
        self.state_dim = state_dim
        self.step_count = 0

    def reset(self, seed=None):
        self.step_count = 0
        return np.ones(self.state_dim, dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        obs = np.ones(self.state_dim, dtype=np.float32) * float(self.step_count + 1)
        reward = 1.0
        done = self.step_count >= self.max_steps
        return obs, reward, done, {}


def _create_components(model_seed: int = 42):
    torch.manual_seed(model_seed)
    state_dim = 2
    act_dim = 1
    adapter = DTAdapter(DummyDT(act_dim=act_dim), state_dim=state_dim, act_dim=act_dim)
    ensemble = DynamicsEnsemble(
        state_dim=state_dim,
        act_dim=act_dim,
        num_models=2,
        hidden_dim=16,
        reward_scale=1000.0,
        horizon=10,
    )
    return_model = ReturnCapper(
        model=GlobalReturnCap(cap_value=1.0),
        horizon=10,
        scale=1000.0,
    )
    capper = ReturnCapper(
        model=GlobalReturnCap(cap_value=1.0),
        horizon=10,
        scale=1000.0,
    )
    return adapter, ensemble, return_model, capper


def test_b4_reproducibility_same_seed():
    """Two B4 runs with same seed/config/model/env produce identical candidate sequences and actions."""
    adapter1, ensemble1, return_model1, _ = _create_components()
    adapter2, ensemble2, return_model2, _ = _create_components()

    run1 = run_variant(
        "B4",
        StepEnv(max_steps=3),
        adapter1,
        ensemble=ensemble1,
        return_model=return_model1,
        horizon=10,
        num_candidates=5,
        seed=42,
        action_low=np.array([-1.0]),
        action_high=np.array([1.0]),
    )
    run2 = run_variant(
        "B4",
        StepEnv(max_steps=3),
        adapter2,
        ensemble=ensemble2,
        return_model=return_model2,
        horizon=10,
        num_candidates=5,
        seed=42,
        action_low=np.array([-1.0]),
        action_high=np.array([1.0]),
    )

    c1 = [info["candidate_actions"] for info in run1.rollout.policy_infos]
    c2 = [info["candidate_actions"] for info in run2.rollout.policy_infos]
    assert len(c1) == len(c2) == 3
    for s1, s2 in zip(c1, c2):
        np.testing.assert_allclose(s1, s2, rtol=0, atol=0)

    for a1, a2 in zip(run1.rollout.actions, run2.rollout.actions):
        np.testing.assert_allclose(a1, a2, rtol=0, atol=0)
    for a1, a2 in zip(run1.rollout.executed_actions, run2.rollout.executed_actions):
        np.testing.assert_allclose(a1, a2, rtol=0, atol=0)


def test_b5_reproducibility_same_seed():
    """Two B5 runs with same seed/config/model/env produce identical candidate sequences and actions."""
    adapter1, ensemble1, return_model1, capper1 = _create_components()
    adapter2, ensemble2, return_model2, capper2 = _create_components()

    run1 = run_variant(
        "B5",
        StepEnv(max_steps=3),
        adapter1,
        capper=capper1,
        ensemble=ensemble1,
        return_model=return_model1,
        horizon=10,
        num_candidates=5,
        seed=42,
        action_low=np.array([-1.0]),
        action_high=np.array([1.0]),
    )
    run2 = run_variant(
        "B5",
        StepEnv(max_steps=3),
        adapter2,
        capper=capper2,
        ensemble=ensemble2,
        return_model=return_model2,
        horizon=10,
        num_candidates=5,
        seed=42,
        action_low=np.array([-1.0]),
        action_high=np.array([1.0]),
    )

    c1 = [info["candidate_actions"] for info in run1.rollout.policy_infos]
    c2 = [info["candidate_actions"] for info in run2.rollout.policy_infos]
    assert len(c1) == len(c2) == 3
    for s1, s2 in zip(c1, c2):
        np.testing.assert_allclose(s1, s2, rtol=0, atol=0)

    for a1, a2 in zip(run1.rollout.actions, run2.rollout.actions):
        np.testing.assert_allclose(a1, a2, rtol=0, atol=0)
    for a1, a2 in zip(run1.rollout.executed_actions, run2.rollout.executed_actions):
        np.testing.assert_allclose(a1, a2, rtol=0, atol=0)


def test_different_seeds_change_proposals():
    """Different seeds produce different candidate proposals."""
    adapter1, ensemble1, return_model1, _ = _create_components()
    adapter2, ensemble2, return_model2, _ = _create_components()

    run1 = run_variant(
        "B4",
        StepEnv(max_steps=2),
        adapter1,
        ensemble=ensemble1,
        return_model=return_model1,
        horizon=10,
        num_candidates=5,
        seed=42,
        action_low=np.array([-1.0]),
        action_high=np.array([1.0]),
    )
    run2 = run_variant(
        "B4",
        StepEnv(max_steps=2),
        adapter2,
        ensemble=ensemble2,
        return_model=return_model2,
        horizon=10,
        num_candidates=5,
        seed=999,
        action_low=np.array([-1.0]),
        action_high=np.array([1.0]),
    )

    c1 = run1.rollout.policy_infos[0]["candidate_actions"]
    c2 = run2.rollout.policy_infos[0]["candidate_actions"]
    assert not np.allclose(c1, c2)


def test_changing_obs_noise_does_not_affect_actuator_noise_stream():
    """Independent streams: changing obs noise scale does not change actuator noise sequence."""
    action_low = np.array([-1.0], dtype=np.float32)
    action_high = np.array([1.0], dtype=np.float32)

    def dummy_policy(tokens, obs, step):
        return np.array([0.5], dtype=np.float32)

    clean = evaluate_with_budget(
        env=StepEnv(max_steps=4),
        policy_fn=dummy_policy,
        horizon=4,
        act_noise_scale=0.2,
        action_low=action_low,
        action_high=action_high,
        seed=777,
    )
    noisy = evaluate_with_budget(
        env=StepEnv(max_steps=4),
        policy_fn=dummy_policy,
        horizon=4,
        obs_noise_scale=0.3,
        state_std=np.ones(2, dtype=np.float32),
        act_noise_scale=0.2,
        action_low=action_low,
        action_high=action_high,
        seed=777,
    )

    for a1, a2 in zip(clean.executed_actions, noisy.executed_actions):
        np.testing.assert_allclose(a1, a2, rtol=0, atol=0)


def test_consecutive_episodes_no_leak_controller():
    """Two consecutive episodes with the same controller do not leak history or RNG state."""
    adapter, ensemble, return_model, _ = _create_components()
    controller = ModuleUController(
        dt_adapter=adapter,
        ensemble=ensemble,
        return_model=return_model,
        num_candidates=5,
        horizon=10,
        action_low=np.array([-1.0]),
        action_high=np.array([1.0]),
    )

    run1 = evaluate_with_budget(
        env=StepEnv(max_steps=3),
        policy_fn=controller.policy_fn,
        policy_reset_fn=controller.reset,
        policy_set_rng_fn=controller.set_rng,
        horizon=10,
        seed=101,
        action_low=np.array([-1.0]),
        action_high=np.array([1.0]),
    )

    run2 = evaluate_with_budget(
        env=StepEnv(max_steps=3),
        policy_fn=controller.policy_fn,
        policy_reset_fn=controller.reset,
        policy_set_rng_fn=controller.set_rng,
        horizon=10,
        seed=101,
        action_low=np.array([-1.0]),
        action_high=np.array([1.0]),
    )

    c1 = [info["candidate_actions"] for info in run1.policy_infos]
    c2 = [info["candidate_actions"] for info in run2.policy_infos]
    for s1, s2 in zip(c1, c2):
        np.testing.assert_allclose(s1, s2, rtol=0, atol=0)


def test_unseeded_proposal_fails_explicitly():
    """If policy requires deterministic proposals but no RNG injection is possible, fail explicitly."""
    def blind_policy(tokens, obs, step):
        return np.array([0.0], dtype=np.float32)

    with pytest.raises(RuntimeError, match="Candidate-proposal RNG"):
        evaluate_with_budget(
            env=StepEnv(max_steps=2),
            policy_fn=blind_policy,
            horizon=5,
            seed=42,
            requires_proposal_rng=True,
        )
