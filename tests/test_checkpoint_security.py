import warnings
import numpy as np
import pytest
import torch

from udt.dynamics import (
    load_dynamics_checkpoint,
    save_dynamics_checkpoint,
    train_dynamics_ensemble,
)
from udt.readiness import check_pilot_readiness


class _CustomObject:
    def __init__(self, val: int):
        self.val = val


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


def test_valid_current_checkpoint_loads_with_weights_only_default(tmp_path):
    """Valid checkpoint loads cleanly using default weights_only=True."""
    ds = _make_dummy_dataset()
    ensemble, metrics = train_dynamics_ensemble(
        ds, horizon=10, num_models=1, hidden_dim=8, num_layers=2, epochs=1, batch_size=8, seed=1
    )
    ckpt_path = tmp_path / "valid.pt"
    save_dynamics_checkpoint(ckpt_path, ensemble, metrics=metrics)

    loaded, payload = load_dynamics_checkpoint(ckpt_path)
    assert loaded.state_dim == 2
    assert payload["format_version"] == 1


def test_invalid_format_version_rejected(tmp_path):
    """Unsupported format_version must be rejected."""
    ds = _make_dummy_dataset()
    ensemble, metrics = train_dynamics_ensemble(
        ds, horizon=10, num_models=1, hidden_dim=8, num_layers=2, epochs=1, batch_size=8, seed=1
    )
    ckpt_path = tmp_path / "invalid_ver.pt"
    save_dynamics_checkpoint(ckpt_path, ensemble, metrics=metrics)

    # Tamper with format version
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    payload["format_version"] = 99
    torch.save(payload, ckpt_path)

    with pytest.raises(ValueError, match="format_version"):
        load_dynamics_checkpoint(ckpt_path)


def test_malformed_checkpoint_rejected(tmp_path):
    """Malformed checkpoint missing architecture keys must be rejected."""
    ds = _make_dummy_dataset()
    ensemble, metrics = train_dynamics_ensemble(
        ds, horizon=10, num_models=1, hidden_dim=8, num_layers=2, epochs=1, batch_size=8, seed=1
    )
    ckpt_path = tmp_path / "malformed.pt"
    save_dynamics_checkpoint(ckpt_path, ensemble, metrics=metrics)

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    payload["architecture"].pop("hidden_dim")
    torch.save(payload, ckpt_path)

    with pytest.raises(ValueError, match="missing fields"):
        load_dynamics_checkpoint(ckpt_path)


def test_incompatible_configuration_rejected(tmp_path):
    """Configuration mismatch against expected dims/horizon/scale is rejected."""
    ds = _make_dummy_dataset()
    ensemble, metrics = train_dynamics_ensemble(
        ds, horizon=10, num_models=1, hidden_dim=8, num_layers=2, epochs=1, batch_size=8, seed=1
    )
    ckpt_path = tmp_path / "incompatible.pt"
    save_dynamics_checkpoint(ckpt_path, ensemble, metrics=metrics)

    with pytest.raises(ValueError, match="state_dim"):
        load_dynamics_checkpoint(ckpt_path, expected_state_dim=5)
    with pytest.raises(ValueError, match="horizon"):
        load_dynamics_checkpoint(ckpt_path, expected_horizon=50)


def test_unsafe_object_rejected_by_default(tmp_path):
    """Pickled custom non-primitive object fails under default weights_only=True."""
    ckpt_path = tmp_path / "unsafe.pt"
    torch.save({"custom_obj": _CustomObject(42)}, ckpt_path)

    with pytest.raises(Exception):
        load_dynamics_checkpoint(ckpt_path)


def test_unsafe_weights_only_false_requires_explicit_opt_in(tmp_path):
    """weights_only=False raises error unless allow_unsafe_legacy=True is set."""
    ckpt_path = tmp_path / "test.pt"
    torch.save({"kind": "dummy"}, ckpt_path)

    with pytest.raises(ValueError, match=r"allow_unsafe_legacy"):
        load_dynamics_checkpoint(ckpt_path, weights_only=False, allow_unsafe_legacy=False)


def test_legacy_opt_in_emits_warning(tmp_path):
    """allow_unsafe_legacy=True permits loading but issues a UserWarning."""
    ds = _make_dummy_dataset()
    ensemble, metrics = train_dynamics_ensemble(
        ds, horizon=10, num_models=1, hidden_dim=8, num_layers=2, epochs=1, batch_size=8, seed=1
    )
    ckpt_path = tmp_path / "legacy.pt"
    save_dynamics_checkpoint(ckpt_path, ensemble, metrics=metrics)

    with pytest.warns(UserWarning, match=r"[Uu]nsafe.*deserialization|weights_only=False"):
        load_dynamics_checkpoint(ckpt_path, allow_unsafe_legacy=True)
