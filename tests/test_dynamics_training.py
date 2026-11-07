"""Regression tests for supervised dynamics training data handling."""

import numpy as np
import pytest
import torch

from udt.dynamics import train_dynamics_ensemble


def make_dataset(*, n_per_episode=4, n_episodes=4, state_dim=2, act_dim=1,
                 next_obs_valid=None):
    """Build a small prepared-style dataset with two trajectory splits."""
    n = n_per_episode * n_episodes
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(n, state_dim)).astype(np.float32)
    next_obs = obs + 0.1
    actions = rng.normal(size=(n, act_dim)).astype(np.float32)
    rewards = rng.normal(size=n).astype(np.float32)
    terminals = np.zeros(n, dtype=bool)
    timesteps = np.tile(np.arange(n_per_episode), n_episodes).astype(np.int64)
    episode_ids = np.repeat(np.arange(100, 100 + n_episodes), n_per_episode)
    is_val = np.repeat([False, True], n // 2)
    if next_obs_valid is None:
        next_obs_valid = np.ones(n, dtype=bool)
    next_obs_valid = np.asarray(next_obs_valid, dtype=bool).copy()
    # Every episode ends in a terminal transition.
    terminals[n_per_episode - 1::n_per_episode] = True
    return {
        "observations": obs,
        "next_observations": next_obs.astype(np.float32),
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timesteps": timesteps,
        "episode_ids": episode_ids,
        "is_validation": is_val,
        "next_observations_valid": next_obs_valid,
    }


def train(dataset, **kwargs):
    defaults = dict(
        horizon=10,
        num_models=2,
        hidden_dim=16,
        num_layers=2,
        epochs=2,
        batch_size=8,
        seed=7,
    )
    defaults.update(kwargs)
    return train_dynamics_ensemble(dataset=dataset, **defaults)


def test_invalid_next_state_targets_do_not_affect_delta_loss():
    """Changing a placeholder value at invalid transitions must be a no-op."""
    base = make_dataset()
    invalid = np.array([2, 5, 6, 13], dtype=int)
    valid_mask = np.ones(len(base["observations"]), dtype=bool)
    valid_mask[invalid] = False

    clean = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    clean["next_observations_valid"] = valid_mask
    clean["next_observations"][invalid] = 0.0

    extreme = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    extreme["next_observations_valid"] = valid_mask
    extreme["next_observations"][invalid] = 1e9

    _, clean_metrics = train(clean)
    _, extreme_metrics = train(extreme)

    for m_clean, m_extreme in zip(clean_metrics["models"], extreme_metrics["models"]):
        np.testing.assert_allclose(m_clean["train_loss"], m_extreme["train_loss"])
        np.testing.assert_allclose(m_clean["val_loss"], m_extreme["val_loss"])
        np.testing.assert_allclose(m_clean["best_val_loss"], m_extreme["best_val_loss"])


def test_invalid_next_state_transition_still_supervises_reward_and_terminal():
    dataset = make_dataset()
    invalid = np.array([3, 7, 11, 15], dtype=int)  # includes terminal rows 3, 7, 11, 15
    dataset["next_observations_valid"][invalid] = False

    _, metrics = train(dataset)
    summary = metrics["data_summary"]
    n_train = int(np.sum(~dataset["is_validation"]))

    assert summary["train_transitions"] == n_train
    assert summary["train_delta_state_samples"] == n_train - int(
        np.sum(~dataset["is_validation"] & ~dataset["next_observations_valid"]))
    assert summary["train_reward_samples"] == n_train
    assert summary["train_terminal_samples"] == n_train
    assert summary["train_invalid_next_state_transitions"] == int(
        np.sum(~dataset["is_validation"] & ~dataset["next_observations_valid"]))


def test_no_valid_next_state_training_targets_fails_clearly():
    dataset = make_dataset()
    dataset["next_observations_valid"][~dataset["is_validation"]] = False

    with pytest.raises(ValueError, match="valid next-state.*delta-state|delta-state.*valid next-state"):
        train(dataset)


def test_all_valid_mask_is_equivalent_to_implicit_all_valid():
    dataset = make_dataset()
    implicit = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in dataset.items()}
    implicit.pop("next_observations_valid")

    _, explicit_metrics = train(dataset)
    _, implicit_metrics = train(implicit)

    for a, b in zip(explicit_metrics["models"], implicit_metrics["models"]):
        np.testing.assert_allclose(a["train_loss"], b["train_loss"])
        np.testing.assert_allclose(a["val_loss"], b["val_loss"])


def test_bootstrap_samples_whole_trajectories_and_preserves_multiplicity():
    dataset = make_dataset(n_per_episode=4, n_episodes=8)
    _, metrics = train(dataset, num_models=2)
    train_episodes = set(metrics["data_summary"]["train_episode_ids"])
    val_episodes = set(metrics["data_summary"]["validation_episode_ids"])
    episode_lengths = {
        int(eid): int(np.sum(dataset["episode_ids"] == eid))
        for eid in np.unique(dataset["episode_ids"][~dataset["is_validation"]])
    }

    for model_metrics in metrics["models"]:
        info = model_metrics["bootstrap"]
        assert set(info["sampled_episode_ids"]).issubset(train_episodes)
        assert not (set(info["sampled_episode_ids"]) & val_episodes)
        assert len(info["sampled_episode_ids"]) == len(train_episodes)
        assert info["unique_episodes"] == len(info["episode_counts"])
        # All transitions of a selected episode are kept; repeated draws keep
        # the repeated multiplicity.
        expected = sum(count * episode_lengths[eid]
                       for eid, count in info["episode_counts"].items())
        assert info["bootstrap_transitions"] == expected
        assert info["init_seed"] == info["bootstrap_seed"] == info["shuffle_seed"]


def test_heads_have_independent_bootstrap_draws_and_initializations():
    dataset = make_dataset(n_per_episode=4, n_episodes=8)
    ensemble, metrics = train(dataset, num_models=3)
    samples = [tuple(m["bootstrap"]["sampled_episode_ids"]) for m in metrics["models"]]
    assert len(set(samples)) == 3

    weights = [torch.cat([p.detach().flatten() for p in model.parameters()])
               for model in ensemble.models]
    for i in range(len(weights)):
        for j in range(i + 1, len(weights)):
            assert not torch.allclose(weights[i], weights[j])


def test_same_seed_is_invariant_to_global_rng_state():
    dataset = make_dataset(n_per_episode=4, n_episodes=4)

    torch.manual_seed(1)
    np.random.seed(2)
    ensemble_a, metrics_a = train(dataset)

    # Perturb process-global RNGs; scoped seeding must absorb the difference.
    torch.manual_seed(999)
    np.random.seed(999)
    torch.randn(100)
    np.random.rand(100)
    ensemble_b, metrics_b = train(dataset)

    for model_a, model_b in zip(ensemble_a.models, ensemble_b.models):
        for (name_a, param_a), (name_b, param_b) in zip(
                model_a.state_dict().items(), model_b.state_dict().items()):
            assert name_a == name_b
            torch.testing.assert_close(param_a, param_b, rtol=0, atol=0)

    for a, b in zip(metrics_a["models"], metrics_b["models"]):
        assert a["bootstrap"] == b["bootstrap"]
        np.testing.assert_allclose(a["train_loss"], b["train_loss"], rtol=0, atol=0)
        np.testing.assert_allclose(a["val_loss"], b["val_loss"], rtol=0, atol=0)


def test_different_seed_changes_initialization_and_bootstrap():
    dataset = make_dataset(n_per_episode=4, n_episodes=8)
    ensemble_a, metrics_a = train(dataset, seed=7, num_models=2)
    ensemble_b, metrics_b = train(dataset, seed=8, num_models=2)

    assert metrics_a["models"][0]["bootstrap"] != metrics_b["models"][0]["bootstrap"]
    weights_a = torch.cat([p.detach().flatten() for p in ensemble_a.models[0].parameters()])
    weights_b = torch.cat([p.detach().flatten() for p in ensemble_b.models[0].parameters()])
    assert not torch.allclose(weights_a, weights_b)


def test_dynamics_input_validation_reports_invalid_field():
    base = make_dataset()

    with pytest.raises(ValueError, match="horizon must be positive"):
        train(base, horizon=0)
    with pytest.raises(ValueError, match="scale must be positive"):
        train(base, scale=0.0)

    bad = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    bad["observations"][0, 0] = np.nan
    with pytest.raises(ValueError, match="observations contains non-finite"):
        train(bad)

    bad = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    bad["actions"][0, 0] = np.inf
    with pytest.raises(ValueError, match="actions contains non-finite"):
        train(bad)

    bad = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    bad["rewards"][0] = np.nan
    with pytest.raises(ValueError, match="rewards contains non-finite"):
        train(bad)

    bad = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    bad["next_observations"][0, 0] = np.nan
    with pytest.raises(ValueError, match="next_observations contains non-finite"):
        train(bad)

    bad = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    bad["timesteps"][0] = 99
    with pytest.raises(ValueError, match="timesteps must lie"):
        train(bad)

    bad = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    bad["episode_ids"] = bad["episode_ids"].astype(np.float32)
    with pytest.raises(ValueError, match="episode_ids must be integer"):
        train(bad)

    bad = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    del bad["episode_ids"]
    with pytest.raises(ValueError, match="episode_ids"):
        train(bad)


def test_train_validation_episode_leakage_is_rejected():
    dataset = make_dataset()
    # Put one row of a training episode into the validation split.
    val_idx = int(np.flatnonzero(dataset["is_validation"])[0])
    dataset["episode_ids"][val_idx] = dataset["episode_ids"][0]
    with pytest.raises(ValueError, match="leakage"):
        train(dataset)


def test_terminal_transition_with_invalid_next_state_is_masked_not_dropped():
    dataset = make_dataset(n_per_episode=2, n_episodes=4)
    terminal_rows = np.flatnonzero(dataset["terminals"])
    assert len(terminal_rows) > 0
    dataset["next_observations_valid"][terminal_rows] = False

    _, metrics = train(dataset)
    # Terminal rows are still valid supervision for reward/terminal even though
    # their post-terminal state is unknown.
    assert metrics["data_summary"]["train_terminal_samples"] > 0
    assert metrics["data_summary"]["train_reward_samples"] > metrics[
        "data_summary"]["train_delta_state_samples"]
