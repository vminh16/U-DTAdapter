import json
import numpy as np
import pytest

from udt.dt_integration import train_decision_transformer


def _make_base_dataset():
    obs = np.ones((10, 4), dtype=np.float32)
    actions = np.zeros((10, 2), dtype=np.float32)
    rewards = np.ones(10, dtype=np.float32)
    rtgs = np.ones(10, dtype=np.float32)
    timesteps = np.tile(np.arange(5), 2)
    terminals = np.array([False, False, False, False, True, False, False, False, False, True])
    eids = np.array([0] * 5 + [1] * 5, dtype=np.int64)
    is_val = np.array([False] * 5 + [True] * 5, dtype=bool)

    arrays = {
        "observations": obs,
        "actions": actions,
        "rewards": rewards,
        "returns_to_go": rtgs,
        "timesteps": timesteps,
        "terminals": terminals,
        "episode_ids": eids,
        "is_validation": is_val,
    }
    manifest = {
        "state_mean": [1.0, 1.0, 1.0, 1.0],
        "state_std": [1.0, 1.0, 1.0, 1.0],
        "reward_scale": 1000.0,
        "horizon": 10,
        "train_episode_ids": [0],
        "validation_episode_ids": [1],
    }
    return arrays, manifest


def test_valid_trajectory_split():
    """Valid split with 1 train and 1 val episode succeeds."""
    arrays, manifest = _make_base_dataset()
    model, summary = train_decision_transformer(
        arrays,
        manifest,
        num_steps=1,
        batch_size=1,
        k=3,
        embed_dim=16,
        n_layer=1,
        n_head=1,
        eval_interval=1,
        eval_batches=1,
        device="cpu",
    )
    assert summary["num_train_trajectories"] == 1
    assert summary["num_val_trajectories"] == 1


def test_mixed_split_episode_rejected():
    """Episode containing both train and validation rows must raise ValueError."""
    arrays, manifest = _make_base_dataset()
    # Make episode 0 have mixed flags (3 False, 2 True)
    arrays["is_validation"] = np.array([False, False, False, True, True, True, True, True, True, True])

    with pytest.raises(ValueError, match="mixed.*validation|trajectory split"):
        train_decision_transformer(
            arrays,
            manifest,
            num_steps=1,
            batch_size=1,
            k=3,
            device="cpu",
        )


def test_empty_train_split_rejected():
    """Empty training split must raise ValueError without falling back to val."""
    arrays, manifest = _make_base_dataset()
    arrays["is_validation"] = np.ones(10, dtype=bool)
    manifest["train_episode_ids"] = []
    manifest["validation_episode_ids"] = [0, 1]

    with pytest.raises(ValueError, match="[Tt]rain.*empty"):
        train_decision_transformer(
            arrays,
            manifest,
            num_steps=1,
            batch_size=1,
            k=3,
            device="cpu",
        )


def test_empty_val_split_rejected():
    """Empty validation split must raise ValueError in B0 protocol."""
    arrays, manifest = _make_base_dataset()
    arrays["is_validation"] = np.zeros(10, dtype=bool)
    manifest["train_episode_ids"] = [0, 1]
    manifest["validation_episode_ids"] = []

    with pytest.raises(ValueError, match="[Vv]alidation.*empty"):
        train_decision_transformer(
            arrays,
            manifest,
            num_steps=1,
            batch_size=1,
            k=3,
            device="cpu",
        )


def test_missing_is_validation_rejected():
    """Missing is_validation field must raise ValueError."""
    arrays, manifest = _make_base_dataset()
    del arrays["is_validation"]

    with pytest.raises(ValueError, match="is_validation"):
        train_decision_transformer(
            arrays,
            manifest,
            num_steps=1,
            batch_size=1,
            k=3,
            device="cpu",
        )


def test_malformed_is_validation_rejected():
    """Malformed is_validation (e.g. wrong shape or non-boolean) must raise ValueError."""
    arrays, manifest = _make_base_dataset()
    arrays["is_validation"] = np.array([0, 1, 2, 3])  # wrong length

    with pytest.raises(ValueError, match="is_validation"):
        train_decision_transformer(
            arrays,
            manifest,
            num_steps=1,
            batch_size=1,
            k=3,
            device="cpu",
        )


def test_manifest_dataset_split_mismatch_rejected():
    """Inconsistency between manifest episode IDs and dataset is_validation must raise ValueError."""
    arrays, manifest = _make_base_dataset()
    # Swap manifest train and val episode lists
    manifest["train_episode_ids"] = [1]
    manifest["validation_episode_ids"] = [0]

    with pytest.raises(ValueError, match="[Ss]plit mismatch|episode_ids"):
        train_decision_transformer(
            arrays,
            manifest,
            num_steps=1,
            batch_size=1,
            k=3,
            device="cpu",
        )
