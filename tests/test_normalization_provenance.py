import numpy as np
import pytest
import torch

from udt.dynamics import (
    DynamicsEnsemble,
    load_dynamics_checkpoint,
    save_dynamics_checkpoint,
    train_dynamics_ensemble,
)
from udt.readiness import check_pilot_readiness


def _make_dummy_dataset(n: int = 20):
    obs = np.ones((n, 2), dtype=np.float32)
    next_obs = np.ones((n, 2), dtype=np.float32) * 1.5
    actions = np.zeros((n, 1), dtype=np.float32)
    rewards = np.ones(n, dtype=np.float32)
    terminals = np.zeros(n, dtype=bool)
    timesteps = np.zeros(n, dtype=np.int64)
    episode_ids = np.zeros(n, dtype=np.int64)
    episode_ids[n // 2:] = 1
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


def test_internal_train_only_normalization(tmp_path):
    """Internal computation on training split records dynamics_training_split and verified=True."""
    ds = _make_dummy_dataset()
    ensemble, metrics = train_dynamics_ensemble(
        dataset=ds,
        horizon=10,
        num_models=1,
        hidden_dim=8,
        num_layers=2,
        epochs=1,
        batch_size=8,
        seed=1,
    )

    assert hasattr(ensemble, "normalization_provenance")
    assert ensemble.normalization_provenance["source"] == "dynamics_training_split"
    assert ensemble.normalization_provenance["verified"] is True

    ckpt_path = tmp_path / "ensemble_internal.pt"
    save_dynamics_checkpoint(ckpt_path, ensemble, metrics=metrics)

    loaded_ensemble, payload = load_dynamics_checkpoint(ckpt_path)
    assert payload["normalization"]["source"] == "dynamics_training_split"
    assert payload["normalization"]["verified"] is True
    assert loaded_ensemble.normalization_provenance["source"] == "dynamics_training_split"
    assert loaded_ensemble.normalization_provenance["verified"] is True


def test_external_verified_normalization(tmp_path):
    """External statistics with verified provenance are recorded and preserved."""
    ds = _make_dummy_dataset()
    mean = np.array([1.0, 1.0], dtype=np.float32)
    std = np.array([2.0, 2.0], dtype=np.float32)
    prov = {"source": "external_verified", "verified": True, "details": "d4rl_train_slice"}

    ensemble, metrics = train_dynamics_ensemble(
        dataset=ds,
        horizon=10,
        num_models=1,
        hidden_dim=8,
        num_layers=2,
        epochs=1,
        batch_size=8,
        seed=1,
        state_mean=mean,
        state_std=std,
        normalization_provenance=prov,
    )

    assert ensemble.normalization_provenance["source"] == "external_verified"
    assert ensemble.normalization_provenance["verified"] is True

    ckpt_path = tmp_path / "ensemble_ext_verified.pt"
    save_dynamics_checkpoint(ckpt_path, ensemble, metrics=metrics)

    loaded_ensemble, payload = load_dynamics_checkpoint(ckpt_path)
    assert payload["normalization"]["source"] == "external_verified"
    assert payload["normalization"]["verified"] is True
    assert loaded_ensemble.normalization_provenance["source"] == "external_verified"
    assert loaded_ensemble.normalization_provenance["verified"] is True


def test_external_unverified_normalization_cannot_claim_training_split(tmp_path):
    """Providing external statistics without verified provenance MUST NOT record dynamics_training_split."""
    ds = _make_dummy_dataset()
    mean = np.array([1.0, 1.0], dtype=np.float32)
    std = np.array([2.0, 2.0], dtype=np.float32)

    ensemble, metrics = train_dynamics_ensemble(
        dataset=ds,
        horizon=10,
        num_models=1,
        hidden_dim=8,
        num_layers=2,
        epochs=1,
        batch_size=8,
        seed=1,
        state_mean=mean,
        state_std=std,
    )

    assert ensemble.normalization_provenance["source"] == "external_unverified"
    assert ensemble.normalization_provenance["verified"] is False

    ckpt_path = tmp_path / "ensemble_ext_unverified.pt"
    save_dynamics_checkpoint(ckpt_path, ensemble, metrics=metrics)

    loaded_ensemble, payload = load_dynamics_checkpoint(ckpt_path)
    assert payload["normalization"]["source"] == "external_unverified"
    assert payload["normalization"]["verified"] is False


def test_readiness_blocks_unverified_normalization(tmp_path):
    """Readiness must block or fail when normalization statistics are unverified."""
    payload = {
        "normalization": {
            "state_mean": [0.0, 0.0],
            "state_std": [1.0, 1.0],
            "coordinate_system": "raw_observation",
            "source": "external_unverified",
            "verified": False,
        }
    }
    report = check_pilot_readiness(dynamics_payload=payload)
    assert report["checks"]["normalization"]["status"] in ("fail", "blocked")
    assert "normalization" in report["blockers"]
    assert not report["ready"]
