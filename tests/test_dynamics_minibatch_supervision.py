import numpy as np
import pytest

from udt.dynamics import train_dynamics_ensemble


def _make_base_dataset(n_samples: int = 20):
    obs = np.ones((n_samples, 2), dtype=np.float32)
    next_obs = np.ones((n_samples, 2), dtype=np.float32) * 1.5
    actions = np.zeros((n_samples, 1), dtype=np.float32)
    rewards = np.ones(n_samples, dtype=np.float32)
    terminals = np.zeros(n_samples, dtype=bool)
    timesteps = np.zeros(n_samples, dtype=np.int64)
    episode_ids = np.zeros(n_samples, dtype=np.int64)
    episode_ids[n_samples // 2:] = 1
    # 2 episodes: ep 0 is train (10 samples), ep 1 is val (10 samples)
    is_val = episode_ids == 1
    return {
        "observations": obs,
        "next_observations": next_obs,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timesteps": timesteps,
        "episode_ids": episode_ids,
        "is_validation": is_val,
    }


def test_minibatch_with_all_masks_false_does_not_crash_backward():
    """A minibatch (e.g. batch_size=1) with all masks false is skipped without backward failure."""
    ds = _make_base_dataset(n_samples=4)
    # ep 0 has 2 samples, ep 1 has 2 samples
    # In ep 0: sample 0 has valid masks, sample 1 has ALL masks False
    # In ep 1 (val): all samples have valid masks
    valid_next = np.array([True, False, True, True])
    valid_rewards = np.array([True, False, True, True])
    valid_terminals = np.array([True, False, True, True])
    ds["next_observations_valid"] = valid_next
    ds["rewards_valid"] = valid_rewards
    ds["terminals_valid"] = valid_terminals

    # With batch_size=1, sample 1 will form an entirely unmasked minibatch.
    # In old code: torch.zeros(()).backward() crashes with RuntimeError (no grad_fn).
    ensemble, metrics = train_dynamics_ensemble(
        dataset=ds,
        horizon=10,
        num_models=1,
        hidden_dim=8,
        num_layers=2,
        epochs=1,
        batch_size=1,
        seed=42,
    )

    m = metrics["models"][0]
    assert "skipped_minibatches" in m
    assert m["skipped_minibatches"] >= 1


def test_mixed_valid_and_invalid_labels_in_minibatch():
    """Minibatch containing partial valid masks computes exact denominator per output."""
    ds = _make_base_dataset(n_samples=6)
    # ep 0 (train) has 3 samples, ep 1 (val) has 3 samples
    # sample 0: delta only
    # sample 1: reward only
    # sample 2: terminal only
    valid_next = np.array([True, False, False, True, True, True])
    valid_rewards = np.array([False, True, False, True, True, True])
    valid_terminals = np.array([False, False, True, True, True, True])
    ds["valid_next_states"] = valid_next
    ds["valid_rewards"] = valid_rewards
    ds["valid_terminals"] = valid_terminals

    # batch_size=3: all 3 train samples in one minibatch
    ensemble, metrics = train_dynamics_ensemble(
        dataset=ds,
        horizon=10,
        num_models=1,
        hidden_dim=8,
        num_layers=2,
        epochs=1,
        batch_size=3,
        seed=42,
    )
    assert metrics["models"][0]["skipped_minibatches"] == 0
    assert len(metrics["models"][0]["train_loss"]) == 1


def test_entire_head_lacking_delta_state_fails_explicitly():
    """If an entire head or training split completely lacks delta-state labels, fail loudly."""
    ds = _make_base_dataset(n_samples=4)
    # No delta-state labels anywhere
    ds["next_observations_valid"] = np.zeros(4, dtype=bool)

    with pytest.raises((RuntimeError, ValueError), match=r"[Dd]elta.*supervision|valid.*next-state"):
        train_dynamics_ensemble(
            dataset=ds,
            horizon=10,
            num_models=1,
            hidden_dim=8,
            num_layers=2,
            epochs=1,
            batch_size=2,
            seed=42,
        )
