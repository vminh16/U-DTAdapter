"""Tests for Minimal Adaptive RTG Intervention (Phase 1: Constant Alpha).

Covers:
- Part A: Mathematical invariants of the adaptive RTG formula:
    B_t^{DT} = B_t^{raw} - alpha * max(0, B_t^{raw} - C_t)
- Part B: Regression tests and backward compatibility (Tests 1-4).
- Part C: Integration tests (seed reproducibility, episode reset, candidate RNG, diagnostics).
"""

import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from udt.intervention import ReturnBudget
from udt.return_cap import GlobalReturnCap, ReturnCapMLP, ReturnCapper
from udt.evaluation import evaluate_with_budget, RolloutResult
from udt.dt_adapter import DTAdapter
from udt.dynamics import DynamicsEnsemble
from udt.experiment import run_variant, evaluate_variant, build_evaluation_report


# ==============================================================================
# Mocks and Fixtures
# ==============================================================================

class DeterministicEnv:
    """Deterministic environment for regression and invariance testing."""

    def __init__(self, step_reward: float = 10.0, max_steps: int = 5, state_dim: int = 2):
        self.step_reward = step_reward
        self.max_steps = max_steps
        self.state_dim = state_dim
        self.step_count = 0
        self.action_space = type(
            "ActionSpace", (),
            {
                "shape": (1,),
                "low": np.array([-1.0], dtype=np.float32),
                "high": np.array([1.0], dtype=np.float32),
            }
        )()

    def reset(self, seed=None):
        self.step_count = 0
        return np.ones(self.state_dim, dtype=np.float32) * 0.5

    def step(self, action):
        self.step_count += 1
        obs = np.ones(self.state_dim, dtype=np.float32) * float(self.step_count)
        reward = self.step_reward
        done = self.step_count >= self.max_steps
        return obs, reward, done, {}


class PredictableDT(nn.Module):
    """Deterministic policy network that reacts to the latest RTG token."""

    def __init__(self, state_dim: int = 2, act_dim: int = 1):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.linear = nn.Linear(state_dim, act_dim, bias=False)

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # Action is deterministically scaled by the latest RTG token
        latest_rtg = returns_to_go[:, -1].item()
        act = np.clip(np.array([latest_rtg * 0.1], dtype=np.float32), -1.0, 1.0)
        return torch.from_numpy(act)


def _build_test_rig(seed: int = 42, horizon: int = 5):
    torch.manual_seed(seed)
    dt_model = PredictableDT()
    adapter = DTAdapter(dt_model, state_dim=2, act_dim=1, max_length=horizon)
    ensemble = DynamicsEnsemble(
        state_dim=2, act_dim=1, num_models=2, hidden_dim=8,
        state_mean=np.zeros(2, dtype=np.float32),
        state_std=np.ones(2, dtype=np.float32),
        horizon=horizon, reward_scale=1000.0,
    )
    for m in ensemble.models:
        for p in m.parameters():
            p.data.zero_()
    mlp = ReturnCapMLP(state_dim=2, hidden_dim=8, num_layers=2, mode="state_time")
    for p in mlp.parameters():
        p.data.zero_()
    list(mlp.parameters())[-1].data.fill_(0.5)
    return_model = ReturnCapper(mlp, horizon=horizon, scale=1000.0)
    capper = ReturnCapper(GlobalReturnCap(0.5), horizon=horizon, scale=1000.0)
    return adapter, ensemble, return_model, capper


# ==============================================================================
# Part A: Mathematical Invariants
# ==============================================================================

class TestMathematicalInvariants:
    """Verify B_t^{DT} = B_t^{raw} - alpha * max(0, B_t^{raw} - C_t) invariants."""

    def test_alpha_zero_returns_raw_budget(self):
        """alpha=0 must return raw budget regardless of cap value."""
        budget = ReturnBudget(raw=5.0, reward_scale=1000.0)
        # Cap is lower than raw (would normally clip)
        assert budget.token(cap=2.0, alpha=0.0) == 5.0
        # Cap with margin
        assert budget.token(cap=2.0, margin=0.5, alpha=0.0) == 5.0
        # Cap higher than raw
        assert budget.token(cap=10.0, alpha=0.0) == 5.0

    def test_alpha_one_returns_hard_cap(self):
        """alpha=1 must return min(raw, cap + margin) matching hard cap."""
        budget = ReturnBudget(raw=5.0, reward_scale=1000.0)
        assert budget.token(cap=2.0, alpha=1.0) == 2.0
        assert budget.token(cap=2.0, margin=0.5, alpha=1.0) == 2.5
        assert budget.token(cap=10.0, alpha=1.0) == 5.0

    def test_alpha_half_returns_midpoint(self):
        """alpha=0.5 returns exact midpoint when cap is active."""
        budget = ReturnBudget(raw=6.0, reward_scale=1000.0)
        # cap = 2.0, ceiling = 2.0. Midpoint of 6.0 and 2.0 is 4.0
        assert budget.token(cap=2.0, alpha=0.5) == 4.0
        # With margin: cap = 2.0, margin = 1.0 -> ceiling = 3.0. Midpoint is 4.5
        assert budget.token(cap=2.0, margin=1.0, alpha=0.5) == 4.5

    def test_no_intervention_when_raw_le_cap(self):
        """When B_raw <= C, token must always equal B_raw for all alpha."""
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            budget = ReturnBudget(raw=3.0, reward_scale=1000.0)
            # Exactly equal
            assert budget.token(cap=3.0, alpha=alpha) == 3.0
            # Raw strictly less than cap
            assert budget.token(cap=5.0, alpha=alpha) == 3.0
            # Raw less than cap + margin
            assert budget.token(cap=2.0, margin=2.0, alpha=alpha) == 3.0

    def test_token_bounded_between_cap_and_raw(self):
        """Token must always lie in [min(raw, ceiling), raw]."""
        raw = 8.0
        cap = 3.0
        margin = 1.0
        ceiling = cap + margin
        budget = ReturnBudget(raw=raw, reward_scale=1000.0)
        for alpha in np.linspace(0.0, 1.0, 50):
            tok = budget.token(cap=cap, margin=margin, alpha=float(alpha))
            assert min(raw, ceiling) <= tok <= raw

    def test_token_monotonic_nonincreasing_in_alpha(self):
        """For fixed context with raw > ceiling, token does not increase as alpha increases."""
        budget = ReturnBudget(raw=10.0, reward_scale=1000.0)
        alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        tokens = [budget.token(cap=4.0, margin=0.5, alpha=a) for a in alphas]
        for i in range(len(tokens) - 1):
            assert tokens[i] >= tokens[i + 1]

    def test_negative_values_handling(self):
        """Invariants hold when reward, budget, or predicted cap are negative."""
        # Negative predicted cap: cap = -2.0, raw = 1.0
        budget = ReturnBudget(raw=1.0, reward_scale=1000.0)
        assert budget.token(cap=-2.0, alpha=0.0) == 1.0
        assert budget.token(cap=-2.0, alpha=1.0) == -2.0
        assert budget.token(cap=-2.0, alpha=0.5) == -0.5

        # Negative raw budget: raw = -3.0, cap = -5.0 (raw > cap)
        neg_budget = ReturnBudget(raw=-3.0, reward_scale=1000.0)
        assert neg_budget.token(cap=-5.0, alpha=0.0) == -3.0
        assert neg_budget.token(cap=-5.0, alpha=1.0) == -5.0
        assert neg_budget.token(cap=-5.0, alpha=0.5) == -4.0

        # Negative raw budget <= cap: raw = -5.0, cap = -3.0
        assert neg_budget.token(cap=-1.0, alpha=0.5) == -3.0

        # Negative reward recurrence
        neg_budget.observe_reward(-100.0)  # raw becomes -3.0 - (-100/1000) = -2.9
        assert np.isclose(neg_budget.raw, -2.9)

    def test_boundary_b_raw_equals_c(self):
        """B_raw == C produces token == B_raw for any alpha."""
        budget = ReturnBudget(raw=4.2, reward_scale=1000.0)
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert budget.token(cap=4.2, alpha=alpha) == 4.2
            assert budget.token(cap=3.2, margin=1.0, alpha=alpha) == 4.2

    def test_invalid_alpha_rejected(self):
        """Alpha outside [0, 1], NaN, and Inf must be rejected."""
        budget = ReturnBudget(raw=5.0, reward_scale=1000.0)
        with pytest.raises(ValueError, match="alpha must be finite and in \\[0, 1\\]"):
            budget.token(cap=2.0, alpha=-0.01)
        with pytest.raises(ValueError, match="alpha must be finite and in \\[0, 1\\]"):
            budget.token(cap=2.0, alpha=1.01)
        with pytest.raises(ValueError, match="alpha must be finite and in \\[0, 1\\]"):
            budget.token(cap=2.0, alpha=float("nan"))
        with pytest.raises(ValueError, match="alpha must be finite and in \\[0, 1\\]"):
            budget.token(cap=2.0, alpha=float("inf"))
        with pytest.raises(ValueError, match="alpha must be finite and in \\[0, 1\\]"):
            budget.token(cap=2.0, alpha=float("-inf"))

    def test_raw_recurrence_unaffected_by_alpha(self):
        """Raw budget recurrence is strictly B_{t+1}^raw = B_t^raw - r_t/c regardless of alpha."""
        b_alpha0 = ReturnBudget(raw=5.0, reward_scale=100.0)
        b_alpha1 = ReturnBudget(raw=5.0, reward_scale=100.0)
        b_alpha_half = ReturnBudget(raw=5.0, reward_scale=100.0)

        rewards = [10.0, -5.0, 25.0, 0.0]
        caps = [3.0, 2.0, 1.0, 0.5]

        for r, cap in zip(rewards, caps):
            _ = b_alpha0.token(cap=cap, alpha=0.0)
            _ = b_alpha1.token(cap=cap, alpha=1.0)
            _ = b_alpha_half.token(cap=cap, alpha=0.5)

            b_alpha0.observe_reward(r)
            b_alpha1.observe_reward(r)
            b_alpha_half.observe_reward(r)

            assert np.isclose(b_alpha0.raw, b_alpha1.raw)
            assert np.isclose(b_alpha0.raw, b_alpha_half.raw)

    def test_return_capper_step_consistency(self):
        """ReturnCapper.step() must yield identical result to ReturnBudget.token()."""
        capper = ReturnCapper(GlobalReturnCap(2.0), horizon=10, margin=0.5, scale=1000.0)
        budget = ReturnBudget(raw=5.0, reward_scale=1000.0)
        dummy_state = np.zeros(2)

        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            step_token = capper.step(budget, dummy_state, step=0, alpha=alpha)
            direct_token = budget.token(cap=2.0, margin=0.5, alpha=alpha)
            assert np.isclose(step_token, direct_token)


# ==============================================================================
# Part B: Regression Tests (Backward Compatibility)
# ==============================================================================

class TestBackwardCompatibilityRegression:
    """Rigorous equivalence tests between new alpha parameterization and existing baselines."""

    def test_1_b3_alpha_one_matches_legacy_hard_cap(self):
        """Test 1: B3 + alpha=1 matches hard-cap B3 across all tokens, actions, and rewards."""
        env1 = DeterministicEnv(step_reward=15.0, max_steps=4)
        env2 = DeterministicEnv(step_reward=15.0, max_steps=4)
        adapter1, _, _, capper1 = _build_test_rig(seed=10, horizon=4)
        adapter2, _, _, capper2 = _build_test_rig(seed=10, horizon=4)

        # Legacy run: default rtg_alpha = 1.0
        run_default = run_variant(
            "B3", env1, adapter1, capper=capper1, horizon=4,
            initial_target=2000.0, reward_scale=1000.0, seed=123,
        )
        # Explicit alpha = 1.0
        run_explicit = run_variant(
            "B3", env2, adapter2, capper=capper2, horizon=4,
            initial_target=2000.0, reward_scale=1000.0, seed=123,
            rtg_alpha=1.0,
        )

        r_def = run_default.rollout
        r_exp = run_explicit.rollout

        assert r_def.tokens_used == r_exp.tokens_used
        assert r_def.raw_budgets == r_exp.raw_budgets
        assert r_def.total_reward == r_exp.total_reward
        assert r_def.clipping_frequency == r_exp.clipping_frequency
        for a1, a2 in zip(r_def.actions, r_exp.actions):
            np.testing.assert_allclose(a1, a2)

    def test_2_b3_alpha_zero_matches_b0(self):
        """Test 2: B3 + alpha=0 matches B0 in tokens, actions, and rewards."""
        env_b0 = DeterministicEnv(step_reward=12.0, max_steps=4)
        env_b3 = DeterministicEnv(step_reward=12.0, max_steps=4)
        adapter_b0, _, _, _ = _build_test_rig(seed=20, horizon=4)
        adapter_b3, _, _, capper_b3 = _build_test_rig(seed=20, horizon=4)

        run_b0 = run_variant(
            "B0", env_b0, adapter_b0, horizon=4,
            initial_target=2000.0, reward_scale=1000.0, seed=456,
        )
        run_b3 = run_variant(
            "B3", env_b3, adapter_b3, capper=capper_b3, horizon=4,
            initial_target=2000.0, reward_scale=1000.0, seed=456,
            rtg_alpha=0.0,
        )

        r_b0 = run_b0.rollout
        r_b3 = run_b3.rollout

        # Tokens must match unclipped raw budgets
        assert r_b3.tokens_used == r_b0.tokens_used
        assert r_b3.tokens_used == r_b3.raw_budgets
        assert r_b3.total_reward == r_b0.total_reward
        for a_b0, a_b3 in zip(r_b0.actions, r_b3.actions):
            np.testing.assert_allclose(a_b0, a_b3)

        # Effective intervention frequency must be 0 for alpha=0
        assert r_b3.effective_intervention_frequency == 0.0
        assert r_b3.mean_rtg_adjustment == 0.0
        # But cap exceedance (clipping_frequency) can still be > 0 because raw > cap
        assert r_b3.clipping_frequency > 0.0

    def test_3_b5_alpha_zero_matches_b4(self):
        """Test 3: B5 + alpha=0 matches B4 in tokens, actions, candidate selection, and rewards."""
        env_b4 = DeterministicEnv(step_reward=10.0, max_steps=3)
        env_b5 = DeterministicEnv(step_reward=10.0, max_steps=3)
        adapter_b4, ens_b4, rm_b4, _ = _build_test_rig(seed=30, horizon=3)
        adapter_b5, ens_b5, rm_b5, capper_b5 = _build_test_rig(seed=30, horizon=3)

        run_b4 = run_variant(
            "B4", env_b4, adapter_b4, ensemble=ens_b4, return_model=rm_b4,
            horizon=3, initial_target=2000.0, reward_scale=1000.0,
            num_candidates=4, seed=789,
        )
        run_b5 = run_variant(
            "B5", env_b5, adapter_b5, capper=capper_b5, ensemble=ens_b5,
            return_model=rm_b5, horizon=3, initial_target=2000.0,
            reward_scale=1000.0, num_candidates=4, seed=789,
            rtg_alpha=0.0,
        )

        r_b4 = run_b4.rollout
        r_b5 = run_b5.rollout

        assert r_b5.tokens_used == r_b4.tokens_used
        assert r_b5.total_reward == r_b4.total_reward
        for a_b4, a_b5 in zip(r_b4.actions, r_b5.actions):
            np.testing.assert_allclose(a_b4, a_b5)

        # Selected candidates and original actions match
        d_b4 = run_b4.diagnostics()
        d_b5 = run_b5.diagnostics()
        assert d_b5["selected_indices"] == d_b4["selected_indices"]
        for o4, o5 in zip(d_b4["original_actions"], d_b5["original_actions"]):
            np.testing.assert_allclose(o4, o5)

    def test_4_omitting_rtg_alpha_preserves_legacy_behavior(self):
        """Test 4: Omitting rtg_alpha yields exactly the legacy hard-cap rollout."""
        env1 = DeterministicEnv(step_reward=5.0, max_steps=3)
        env2 = DeterministicEnv(step_reward=5.0, max_steps=3)
        adapter1, _, _, capper1 = _build_test_rig(seed=40, horizon=3)
        adapter2, _, _, capper2 = _build_test_rig(seed=40, horizon=3)

        r1 = evaluate_with_budget(
            env=env1, policy_fn=adapter1.policy_fn, capper=capper1,
            initial_target=2000.0, horizon=3, reward_scale=1000.0, seed=1,
        )
        r2 = evaluate_with_budget(
            env=env2, policy_fn=adapter2.policy_fn, capper=capper2,
            initial_target=2000.0, horizon=3, reward_scale=1000.0, seed=1,
            rtg_alpha=1.0,
        )

        assert r1.tokens_used == r2.tokens_used
        assert r1.clipping_frequency == r2.clipping_frequency
        assert r1.effective_intervention_frequency == r2.effective_intervention_frequency
        assert r1.mean_rtg_adjustment == r2.mean_rtg_adjustment


# ==============================================================================
# Part C: Integration and System Tests
# ==============================================================================

class TestAdaptiveRTGIntegration:
    """Verify system-level reproducibility, state isolation, and CLI options."""

    def test_seed_reproducibility_across_runs(self):
        """Identical seeds and alpha must yield identical trajectory telemetry."""
        adapter1, _, _, capper1 = _build_test_rig(seed=50, horizon=4)
        adapter2, _, _, capper2 = _build_test_rig(seed=50, horizon=4)
        env1 = DeterministicEnv(step_reward=10.0, max_steps=4)
        env2 = DeterministicEnv(step_reward=10.0, max_steps=4)

        r1 = evaluate_with_budget(
            env=env1, policy_fn=adapter1.policy_fn, capper=capper1,
            horizon=4, seed=99, rtg_alpha=0.75,
        )
        r2 = evaluate_with_budget(
            env=env2, policy_fn=adapter2.policy_fn, capper=capper2,
            horizon=4, seed=99, rtg_alpha=0.75,
        )

        assert r1.tokens_used == r2.tokens_used
        assert r1.total_reward == r2.total_reward
        assert r1.effective_intervention_frequency == r2.effective_intervention_frequency

    def test_episode_reset_does_not_leak_rtg_history(self):
        """Consecutive episodes reset budget cleanly without leakage."""
        adapter, _, _, capper = _build_test_rig(seed=60, horizon=3)
        env = DeterministicEnv(step_reward=10.0, max_steps=3)

        first = evaluate_with_budget(
            env=env, policy_fn=adapter.policy_fn, capper=capper,
            horizon=3, initial_target=1000.0, rtg_alpha=0.5,
        )
        second = evaluate_with_budget(
            env=env, policy_fn=adapter.policy_fn, capper=capper,
            horizon=3, initial_target=1000.0, rtg_alpha=0.5,
        )

        assert first.tokens_used == second.tokens_used
        assert first.raw_budgets == second.raw_budgets

    def test_candidate_rng_stream_independent_of_alpha(self):
        """Candidate proposal stream in Module U must not be shifted by alpha."""
        adapter1, ens1, rm1, capper1 = _build_test_rig(seed=70, horizon=3)
        adapter2, ens2, rm2, capper2 = _build_test_rig(seed=70, horizon=3)

        # Run with alpha = 0.25 vs alpha = 0.75
        run1 = run_variant(
            "B5", DeterministicEnv(step_reward=10.0, max_steps=3), adapter1,
            capper=capper1, ensemble=ens1, return_model=rm1, horizon=3,
            seed=100, num_candidates=3, rtg_alpha=0.25,
        )
        run2 = run_variant(
            "B5", DeterministicEnv(step_reward=10.0, max_steps=3), adapter2,
            capper=capper2, ensemble=ens2, return_model=rm2, horizon=3,
            seed=100, num_candidates=3, rtg_alpha=0.75,
        )

        # Both runs have identical candidate-proposal stream seeds
        s1 = run1.rollout.seed_mapping["streams"]["candidate_proposal"]["seed"]
        s2 = run2.rollout.seed_mapping["streams"]["candidate_proposal"]["seed"]
        assert s1 == s2

    def test_evaluate_variant_records_alpha_and_diagnostics(self):
        """evaluate_variant summary and records correctly store rtg_alpha and intervention metrics."""
        adapter, _, _, capper = _build_test_rig(seed=80, horizon=3)
        env = DeterministicEnv(step_reward=10.0, max_steps=3)

        summary = evaluate_variant(
            "B3", env, adapter, episodes=2, seed=1, capper=capper,
            horizon=3, initial_target=2000.0, reward_scale=1000.0,
            rtg_alpha=0.5, collect_diagnostics=True,
        )

        assert summary["rtg_alpha"] == 0.5
        assert summary["mean_effective_intervention_frequency"] > 0
        assert summary["mean_rtg_adjustment"] > 0
        for ep in summary["episodes_detail"]:
            assert ep["rtg_alpha"] == 0.5
            assert "effective_intervention_frequency" in ep
            assert "mean_rtg_adjustment" in ep

    def test_cli_parsing_and_evaluation_report(self, tmp_path, monkeypatch):
        """scripts/evaluate.py accepts --rtg-alpha and persists it in report."""
        import sys
        from scripts.evaluate import main as evaluate_main
        from tests.test_b0_evaluate_script import _train_dummy_checkpoint, DummyEnv

        ckpt = _train_dummy_checkpoint(tmp_path, state_dim=4, act_dim=2, horizon=10)
        results_out = tmp_path / "results_alpha.json"
        parity_out = tmp_path / "parity_alpha.json"

        import gymnasium as gym
        monkeypatch.setattr(gym, "make", lambda env_id: DummyEnv())
        monkeypatch.setattr(
            sys, "argv",
            [
                "evaluate.py",
                "--variant", "B0",
                "--checkpoint", str(ckpt),
                "--device", "cpu",
                "--episodes", "2",
                "--horizon", "5",
                "--rtg-alpha", "0.5",
                "--results-output", str(results_out),
                "--parity-output", str(parity_out),
            ],
        )

        ret = evaluate_main()
        assert ret == 0
        data = json.loads(results_out.read_text())
        assert data["config"]["rtg_alpha"] == 0.5
        assert data["rtg_alpha"] == 0.5

    def test_cli_rejects_out_of_bounds_alpha(self, tmp_path, monkeypatch):
        """scripts/evaluate.py rejects invalid --rtg-alpha values."""
        import sys
        from scripts.evaluate import main as evaluate_main
        from tests.test_b0_evaluate_script import _train_dummy_checkpoint

        ckpt = _train_dummy_checkpoint(tmp_path, state_dim=4, act_dim=2, horizon=10)
        monkeypatch.setattr(
            sys, "argv",
            [
                "evaluate.py",
                "--variant", "B0",
                "--checkpoint", str(ckpt),
                "--rtg-alpha", "1.5",
            ],
        )
        with pytest.raises(ValueError, match="--rtg-alpha must be in \\[0, 1\\]"):
            evaluate_main()
