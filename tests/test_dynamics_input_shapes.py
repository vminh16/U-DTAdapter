import numpy as np
import pytest

from udt.dynamics import train_dynamics_ensemble


def _make_valid_dataset(n: int = 20):
    obs = np.ones((n, 3), dtype=np.float32)
    next_obs = np.ones((n, 3), dtype=np.float32) * 1.5
    actions = np.zeros((n, 2), dtype=np.float32)
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


def _train(ds):
    return train_dynamics_ensemble(
        dataset=ds,
        horizon=10,
        num_models=1,
        hidden_dim=8,
        num_layers=2,
        epochs=1,
        batch_size=8,
        seed=1,
    )


@pytest.mark.parametrize(
    "field,bad_shape",
    [
        ("rewards", (20, 1)),
        ("rewards", (20, 2)),
        ("terminals", (20, 1)),
        ("terminals", (20, 2)),
        ("timesteps", (20, 1)),
        ("episode_ids", (20, 1)),
        ("is_validation", (20, 1)),
        ("next_observations_valid", (20, 1)),
        ("valid_next_states", (20, 1)),
    ],
)
def test_scalar_fields_reject_2d_shapes(field, bad_shape):
    """(N, 1) or (N, 2) shapes for scalar targets and masks must be strictly rejected."""
    ds = _make_valid_dataset()
    if field in ("next_observations_valid", "valid_next_states"):
        ds[field] = np.ones(bad_shape, dtype=bool)
    elif field == "is_validation":
        ds[field] = np.zeros(bad_shape, dtype=bool)
        ds[field][10:] = True
    elif field == "terminals":
        ds[field] = np.zeros(bad_shape, dtype=bool)
    elif field in ("timesteps", "episode_ids"):
        ds[field] = np.zeros(bad_shape, dtype=np.int64)
    else:
        ds[field] = np.ones(bad_shape, dtype=np.float32)

    with pytest.raises(ValueError, match=r"shape"):
        _train(ds)


def test_invalid_dtypes_rejected():
    """Non-numeric or non-boolean dtypes must be rejected."""
    ds = _make_valid_dataset()
    ds["rewards"] = np.array(["1.0"] * 20)
    with pytest.raises(ValueError, match=r"dtype|numeric"):
        _train(ds)

    ds2 = _make_valid_dataset()
    ds2["is_validation"] = np.ones(20, dtype=np.int32)
    with pytest.raises(ValueError, match=r"boolean"):
        _train(ds2)


def test_nan_and_inf_rejected():
    """NaN and Inf values in observations, actions, rewards, timesteps must be rejected."""
    ds = _make_valid_dataset()
    ds["rewards"][0] = np.nan
    with pytest.raises(ValueError, match=r"finite|non-finite"):
        _train(ds)

    ds2 = _make_valid_dataset()
    ds2["observations"][0, 0] = np.inf
    with pytest.raises(ValueError, match=r"finite|non-finite"):
        _train(ds2)


def test_length_mismatch_rejected():
    """Any array with length mismatch against n_samples must be rejected."""
    ds = _make_valid_dataset()
    ds["rewards"] = np.ones(19, dtype=np.float32)
    with pytest.raises(ValueError, match=r"length|shape|mismatch"):
        _train(ds)
