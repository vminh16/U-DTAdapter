import numpy as np
import pytest

from udt.dynamics import train_dynamics_ensemble


def _make_dataset_with_sparse_delta(bootstrap_seed_draws_invalid: bool = False):
    # 4 training episodes: eid 0, 1, 2, 3. 1 validation episode: eid 4.
    # eid 0 has valid next-state.
    # eid 1, 2, 3 have INVALID next-states (valid_next_states=False).
    # eid 4 (val) has valid next-state.
    n_per_ep = 5
    n_total = 5 * n_per_ep
    obs = np.ones((n_total, 2), dtype=np.float32)
    next_obs = np.ones((n_total, 2), dtype=np.float32)
    actions = np.zeros((n_total, 1), dtype=np.float32)
    rewards = np.ones(n_total, dtype=np.float32)
    terminals = np.zeros(n_total, dtype=bool)
    timesteps = np.tile(np.arange(n_per_ep), 5).astype(np.int64)
    episode_ids = np.repeat(np.arange(5), n_per_ep).astype(np.int64)
    is_val = episode_ids == 4

    valid_next_states = np.zeros(n_total, dtype=bool)
    # Only episode 0 and 4 have valid next-states
    valid_next_states[episode_ids == 0] = True
    valid_next_states[episode_ids == 4] = True

    return {
        "observations": obs,
        "next_observations": next_obs,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timesteps": timesteps,
        "episode_ids": episode_ids,
        "is_validation": is_val,
        "valid_next_states": valid_next_states,
    }


def test_bootstrap_head_lacking_delta_supervision_fails_explicitly():
    """If a bootstrap head samples only episodes lacking delta supervision, it fails explicitly."""
    ds = _make_dataset_with_sparse_delta()
    # Seed 0 draws [3, 2, 2, 1], none of which have valid delta states.
    rng = np.random.default_rng(0)
    sampled = rng.choice([0, 1, 2, 3], size=4, replace=True)
    assert 0 not in sampled

    with pytest.raises(RuntimeError, match=r"[Bb]ootstrap.*delta.*supervision|valid delta-state"):
        train_dynamics_ensemble(
            dataset=ds,
            horizon=10,
            num_models=1,
            hidden_dim=8,
            num_layers=2,
            epochs=1,
            batch_size=16,
            seed=0,
        )


def test_head_with_sufficient_supervision_passes_and_records_metadata():
    """Head with sufficient delta supervision passes and metadata reflects exact counts."""
    # Create dataset where all episodes have valid next states
    n_per_ep = 5
    n_total = 4 * n_per_ep
    obs = np.ones((n_total, 2), dtype=np.float32)
    next_obs = np.ones((n_total, 2), dtype=np.float32) * 1.5
    actions = np.zeros((n_total, 1), dtype=np.float32)
    rewards = np.ones(n_total, dtype=np.float32)
    terminals = np.zeros(n_total, dtype=bool)
    timesteps = np.tile(np.arange(n_per_ep), 4).astype(np.int64)
    episode_ids = np.repeat(np.arange(4), n_per_ep).astype(np.int64)
    is_val = episode_ids == 3
    valid_next = np.ones(n_total, dtype=bool)

    ds = {
        "observations": obs,
        "next_observations": next_obs,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timesteps": timesteps,
        "episode_ids": episode_ids,
        "is_validation": is_val,
        "valid_next_states": valid_next,
    }

    ensemble, metrics = train_dynamics_ensemble(
        dataset=ds,
        horizon=10,
        num_models=2,
        hidden_dim=8,
        num_layers=2,
        epochs=1,
        batch_size=16,
        seed=42,
    )

    assert len(metrics["models"]) == 2
    for m_meta in metrics["models"]:
        boot = m_meta["bootstrap"]
        assert "valid_delta_transitions" in boot
        assert boot["valid_delta_transitions"] == 15  # 3 train episodes * 5 transitions
        assert boot["bootstrap_transitions"] == 15
        assert set(boot["sampled_episode_ids"]).issubset({0, 1, 2})
        assert 3 not in boot["sampled_episode_ids"]  # No validation leakage!
