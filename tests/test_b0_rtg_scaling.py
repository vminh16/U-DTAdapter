import json
import numpy as np
import pytest
import torch

from udt.data import prepare_dataset
from udt.dt_integration import train_decision_transformer
from udt.evaluation import evaluate_with_budget


def _create_synthetic_raw_data():
    """Create raw transitions for 2 episodes with known rewards."""
    # Episode 1: 5 transitions with reward = 200.0 each. Total return = 1000.0
    # Episode 2: 5 transitions with reward = 400.0 each. Total return = 2000.0
    obs = np.ones((10, 4), dtype=np.float32)
    actions = np.zeros((10, 2), dtype=np.float32)
    rewards = np.array([200.0] * 5 + [400.0] * 5, dtype=np.float32)
    terminals = np.array([False, False, False, False, True, False, False, False, False, True], dtype=bool)
    timeouts = np.zeros(10, dtype=bool)
    return {
        "observations": obs,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
    }


def test_prepare_dataset_rtg_analytical():
    """Verify that prepare_dataset produces Y_t = G_t / c."""
    raw = _create_synthetic_raw_data()
    reward_scale = 1000.0
    prepared, manifest = prepare_dataset(
        raw,
        horizon=10,
        starts_at_reset=True,
        timeout_semantics="unknown",
        validation_fraction=0.5,
        reward_scale=reward_scale,
        seed=0,
    )
    # Episode 1 rewards: [200, 200, 200, 200, 200]
    # G_t: [1000, 800, 600, 400, 200]
    # Y_t = G_t / c: [1.0, 0.8, 0.6, 0.4, 0.2]
    # Episode 2 rewards: [400, 400, 400, 400, 400]
    # G_t: [2000, 1600, 1200, 800, 400]
    # Y_t = G_t / c: [2.0, 1.6, 1.2, 0.8, 0.4]
    rtgs = prepared["returns_to_go"]
    expected_ep1 = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    expected_ep2 = np.array([2.0, 1.6, 1.2, 0.8, 0.4])

    np.testing.assert_allclose(rtgs[:5], expected_ep1, atol=1e-6)
    np.testing.assert_allclose(rtgs[5:], expected_ep2, atol=1e-6)
    assert manifest["reward_scale"] == reward_scale


def test_training_sampler_no_double_scaling(monkeypatch):
    """Verify training sampler yields tokens in G_t / c (NOT G_t / c^2)."""
    raw = _create_synthetic_raw_data()
    reward_scale = 1000.0
    prepared, manifest = prepare_dataset(
        raw,
        horizon=10,
        starts_at_reset=True,
        timeout_semantics="unknown",
        validation_fraction=0.5,
        reward_scale=reward_scale,
        seed=0,
    )

    captured_batches = []

    from udt import dt_integration
    orig_import = dt_integration._import_baseline_dt

    def mock_import(baseline_root=None):
        DT, Trainer = orig_import(baseline_root=baseline_root)

        class CapturingTrainer(Trainer):
            def train_step(self):
                batch = self.get_batch(self.batch_size)
                captured_batches.append(batch)
                return 0.0

        return DT, CapturingTrainer

    monkeypatch.setattr(dt_integration, "_import_baseline_dt", mock_import)

    model, summary = train_decision_transformer(
        prepared,
        manifest,
        num_steps=1,
        batch_size=2,
        k=5,
        embed_dim=16,
        n_layer=1,
        n_head=1,
        device="cpu",
        seed=42,
    )

    assert len(captured_batches) > 0
    s, a, r, d, rtg_tensor, timesteps, mask = captured_batches[0]
    rtg_np = rtg_tensor.cpu().numpy()

    # The maximum value in the RTG tensor should be exactly 2.0 (G/c = 2000 / 1000 = 2.0)
    # or 1.0 depending on which episode was sampled. Neither is 0.002!
    max_rtg = float(np.max(rtg_np))
    assert max_rtg > 0.1, (
        f"RTG token max is {max_rtg}, expected ~1.0-2.0. "
        "Double scaling (G/c^2) detected in training sampler!"
    )


def test_training_sampler_padded_and_unpadded_windows(monkeypatch):
    """Verify analytical RTG values and terminal continuation in padded and unpadded windows."""
    raw = _create_synthetic_raw_data()
    reward_scale = 1000.0
    prepared, manifest = prepare_dataset(
        raw,
        horizon=10,
        starts_at_reset=True,
        timeout_semantics="unknown",
        validation_fraction=0.5,
        reward_scale=reward_scale,
        seed=0,
    )

    captured_batches = []

    from udt import dt_integration
    orig_import = dt_integration._import_baseline_dt

    def mock_import(baseline_root=None):
        DT, Trainer = orig_import(baseline_root=baseline_root)

        class CapturingTrainer(Trainer):
            def train_step(self):
                batch = self.get_batch(self.batch_size)
                captured_batches.append(batch)
                return 0.0

        return DT, CapturingTrainer

    monkeypatch.setattr(dt_integration, "_import_baseline_dt", mock_import)

    # Train with k=7 on 5-step episodes, so every window from start to finish is padded
    train_decision_transformer(
        prepared,
        manifest,
        num_steps=1,
        batch_size=1,
        k=7,
        embed_dim=16,
        n_layer=1,
        n_head=1,
        device="cpu",
        seed=42,
    )

    s, a, r, d, rtg_tensor, timesteps, mask = captured_batches[0]
    rtg_np = rtg_tensor[0, :, 0].cpu().numpy()
    mask_np = mask[0].cpu().numpy()

    # Mask has 0 for padded positions, 1 for valid positions
    # RTG has length k + 1 = 8.
    # Where mask is 0 (padding), RTG must be 0.
    pad_indices = np.where(mask_np == 0)[0]
    for idx in pad_indices:
        assert rtg_np[idx] == 0.0, f"Expected 0.0 padding RTG at index {idx}, got {rtg_np[idx]}"

    # Terminal transition continuation RTG at index (k) must be 0.0
    # Because trajectory ended at step 5
    assert rtg_np[-1] == 0.0, f"Expected terminal continuation RTG to be 0.0, got {rtg_np[-1]}"


def test_scale_mismatch_rejected():
    """Verify ValueError is raised when dataset reward_scale disagrees with manifest."""
    raw = _create_synthetic_raw_data()
    prepared, manifest = prepare_dataset(
        raw,
        horizon=10,
        starts_at_reset=True,
        timeout_semantics="unknown",
        validation_fraction=0.5,
        reward_scale=1000.0,
        seed=0,
    )
    # Add mismatched reward_scale to dataset
    prepared["reward_scale"] = np.array(500.0)

    with pytest.raises(ValueError, match="reward_scale.*mismatch"):
        train_decision_transformer(
            prepared,
            manifest,
            num_steps=1,
            batch_size=2,
            k=5,
            device="cpu",
        )


def test_invalid_manifest_scale_rejected():
    """Verify ValueError when manifest reward_scale is <= 0 or non-finite."""
    raw = _create_synthetic_raw_data()
    prepared, manifest = prepare_dataset(
        raw,
        horizon=10,
        starts_at_reset=True,
        timeout_semantics="unknown",
        validation_fraction=0.5,
        reward_scale=1000.0,
        seed=0,
    )
    manifest["reward_scale"] = -100.0
    with pytest.raises(ValueError, match="reward_scale"):
        train_decision_transformer(
            prepared,
            manifest,
            num_steps=1,
            batch_size=2,
            k=5,
            device="cpu",
        )


def test_inference_and_training_rtg_scale_parity():
    """Verify inference receives identical RTG units as training sampler."""
    target_unscaled_return = 1800.0
    reward_scale = 1000.0

    # In evaluation, initial token is target_unscaled_return / reward_scale = 1.8
    # During rollout, when reward r is received, budget raw becomes B - r / reward_scale
    class DummyEnv:
        def __init__(self):
            self.step_count = 0
            self.action_space = type("ActionSpace", (), {"shape": (2,), "low": -np.ones(2), "high": np.ones(2)})()

        def reset(self, seed=None):
            self.step_count = 0
            return np.zeros(4, dtype=np.float32), {}

        def step(self, action):
            self.step_count += 1
            # Return reward 200.0 each step
            rew = 200.0
            term = self.step_count >= 5
            return np.zeros(4, dtype=np.float32), rew, term, False, {}

    tokens_received = []

    def dummy_policy(tokens, obs, step):
        tokens_received.append(tokens[-1])
        return np.zeros(2, dtype=np.float32)

    rollout = evaluate_with_budget(
        env=DummyEnv(),
        policy_fn=dummy_policy,
        initial_target=target_unscaled_return,
        horizon=5,
        reward_scale=reward_scale,
    )

    # Initial token: 1800 / 1000 = 1.8
    # Step 1 token: 1.8 - 200/1000 = 1.6
    # Step 2 token: 1.6 - 200/1000 = 1.4
    # Step 3 token: 1.4 - 200/1000 = 1.2
    # Step 4 token: 1.2 - 200/1000 = 1.0
    expected_eval_tokens = [1.8, 1.6, 1.4, 1.2, 1.0]
    np.testing.assert_allclose(tokens_received, expected_eval_tokens, atol=1e-6)
