import h5py
import numpy as np
import pytest

from udt.data import audit_dataset, load_dataset, prepare_dataset


def transitions():
    # Two complete 2-step episodes, one interrupted suffix.
    return {
        'observations': np.array([[0.], [1.], [10.], [11.], [20.]]),
        'next_observations': np.array([[1.], [2.], [11.], [12.], [21.]]),
        'actions': np.zeros((5, 1)),
        'rewards': np.array([1., 2., 3., 4., 99.]),
        'terminals': np.array([0, 1, 0, 0, 0]),
        'timeouts': np.array([0, 0, 0, 1, 0]),
    }


def test_audit_distinguishes_horizon_from_censored_tail():
    report = audit_dataset(transitions(), horizon=2, starts_at_reset=True,
                           timeout_semantics='horizon')
    assert report['rtg_eligible_transitions'] == 4
    assert [e['end_reason'] for e in report['segments']] == [
        'terminal', 'horizon', 'unclosed_tail']
    assert report['segments'][-1]['rtg_eligible'] is False


def test_prepare_keeps_episodes_together_and_excludes_censored_labels():
    arrays, manifest = prepare_dataset(transitions(), horizon=2,
        starts_at_reset=True, timeout_semantics='horizon', seed=7,
        validation_fraction=0.5, reward_scale=1.)
    np.testing.assert_array_equal(arrays['returns_to_go'], [3., 2., 7., 4.])
    np.testing.assert_array_equal(arrays['timesteps'], [0, 1, 0, 1])
    for eid in np.unique(arrays['episode_ids']):
        assert len(np.unique(arrays['is_validation'][arrays['episode_ids'] == eid])) == 1
    train = arrays['observations'][~arrays['is_validation']]
    np.testing.assert_allclose(manifest['state_mean'], train.mean(axis=0))
    assert manifest['dropped_transitions'] == 1


def test_unknown_timeout_and_discontinuity_are_not_silently_completed():
    source = transitions()
    report = audit_dataset(source, horizon=2, starts_at_reset=True)
    assert report['rtg_eligible_transitions'] == 2
    source['next_observations'][0] = 900
    report = audit_dataset(source, horizon=2, starts_at_reset=True,
                           timeout_semantics='horizon')
    assert report['segments'][0]['end_reason'] == 'discontinuity'
    assert report['segments'][1]['known_start'] is False
    assert report['segments'][1]['rtg_eligible'] is False


def test_first_unknown_start_stays_excluded_even_if_terminal_is_present():
    report = audit_dataset(transitions(), horizon=2, timeout_semantics='horizon')
    assert report['rtg_eligible_transitions'] == 2
    assert report['segments'][0]['rtg_eligible'] is False


@pytest.mark.parametrize('field,value', [('rewards', np.nan), ('terminals', 2)])
def test_corrupt_values_fail_before_label_generation(field, value):
    source = transitions()
    source[field] = source[field].astype(float)
    source[field][0] = value
    with pytest.raises(ValueError):
        audit_dataset(source, horizon=2)


def test_data_without_two_complete_episodes_cannot_be_split():
    with pytest.raises(ValueError, match='two verified episodes'):
        prepare_dataset(transitions(), horizon=2)


def test_load_dataset_npz(tmp_path):
    d = transitions()
    npz_path = tmp_path / "test.npz"
    np.savez(npz_path, **d)

    loaded = load_dataset(npz_path)
    for k in d:
        np.testing.assert_array_equal(loaded[k], d[k])


def test_load_dataset_hdf5(tmp_path):
    d = transitions()
    h5_path = tmp_path / "test.hdf5"
    with h5py.File(h5_path, "w") as f:
        for k, v in d.items():
            f.create_dataset(k, data=v)

    loaded = load_dataset(h5_path)
    for k in d:
        np.testing.assert_array_equal(loaded[k], d[k])


def test_load_dataset_rejects_pickle(tmp_path):
    pkl_path = tmp_path / "dataset.pkl"
    pkl_path.write_bytes(b"dummy pickle content")

    with pytest.raises(ValueError, match="NPZ or HDF5"):
        load_dataset(pkl_path)

    pickle_path = tmp_path / "dataset.pickle"
    pickle_path.write_bytes(b"dummy pickle content")

    with pytest.raises(ValueError, match="NPZ or HDF5"):
        load_dataset(pickle_path)


def test_load_dataset_ignores_groups(tmp_path):
    d = transitions()
    h5_path = tmp_path / "groups_test.hdf5"
    with h5py.File(h5_path, "w") as f:
        for k, v in d.items():
            f.create_dataset(k, data=v)
        # Create an HDF5 group (like 'infos' or 'metadata' in D4RL)
        group = f.create_group("infos")
        group.create_dataset("qpos", data=np.ones((5, 2)))

    loaded = load_dataset(h5_path)
    # Group should not be loaded as a string array or corrupt top-level keys
    assert "infos" not in loaded
    for k in d:
        np.testing.assert_array_equal(loaded[k], d[k])


def test_two_consecutive_episodes_missing_boundary_terminal_detected():
    # Two 2-step episodes, but terminal is missing at index 1 (boundary)
    # Episode 1: steps 0-1 (obs [0.], [1.] -> next_obs [1.], [2.])
    # Episode 2: steps 2-3 (obs [10.], [11.] -> next_obs [11.], [12.], terminal at step 3)
    data = {
        'observations': np.array([[0.], [1.], [10.], [11.]]),
        'next_observations': np.array([[1.], [2.], [11.], [12.]]),
        'actions': np.zeros((4, 1)),
        'rewards': np.array([1., 1., 1., 1.]),
        'terminals': np.array([0, 0, 0, 1]),
        'timeouts': np.array([0, 0, 0, 0]),
    }
    report = audit_dataset(data, horizon=5, starts_at_reset=True)
    # At index 1, next_obs is [2.] but obs[2] is [10.] -> discontinuity
    assert report['segments'][0]['end_reason'] == 'discontinuity'
    assert report['segments'][0]['rtg_eligible'] is False
    assert report['segments'][1]['known_start'] is False
    assert report['segments'][1]['rtg_eligible'] is False
    assert report['rtg_eligible_transitions'] == 0


def test_missing_timeouts_provenance_preserved_and_cannot_certify_horizon():
    # Dataset where timeouts field is completely absent
    data = {
        'observations': np.zeros((10, 1)),
        'actions': np.zeros((10, 1)),
        'rewards': np.ones(10),
        'terminals': np.zeros(10),
    }
    # Length == horizon, starts_at_reset=True, timeout_semantics='horizon'
    # BUT timeouts was missing from data, so missing timeout must NOT become trusted horizon timeout!
    report = audit_dataset(data, horizon=10, starts_at_reset=True, timeout_semantics='horizon')
    assert report['has_original_timeouts'] is False
    assert report['rtg_eligible_transitions'] == 0
    assert report['segments'][0]['rtg_eligible'] is False
    assert report['segments'][0]['end_reason'] in ('unverified_timeout', 'unclosed_tail', 'censored')
    assert any('Missing timeouts' in w for w in report['warnings'])


def test_missing_next_observations_provenance_and_safe_reconstruction():
    # Two verified episodes with explicit terminals, but missing next_observations
    data = {
        'observations': np.array([[0.], [1.], [10.], [11.]]),
        'actions': np.zeros((4, 1)),
        'rewards': np.array([1., 2., 3., 4.]),
        'terminals': np.array([0, 1, 0, 1]),
        'timeouts': np.array([0, 0, 0, 0]),
    }
    report = audit_dataset(data, horizon=2, starts_at_reset=True)
    assert report['has_original_next_observations'] is False
    assert report['rtg_eligible_episodes'] == 2
    assert any('Missing' in w and 'next_observations' in w for w in report['warnings'])

    arrays, manifest = prepare_dataset(data, horizon=2, starts_at_reset=True,
                                       seed=0, validation_fraction=0.5, reward_scale=1.0)
    assert manifest['has_original_next_observations'] is False
    assert manifest['next_observations_source'] == 'inferred'
    assert 'next_observations_valid' in arrays
    # Inferred next_obs should be valid within episode (index 0, index 2),
    # but INVALID at terminal transitions (index 1, index 3) because post-terminal state is unknown!
    # And transitions must NEVER be stitched across episode boundary (index 1 next_obs != obs 2)!
    valid_mask = arrays['next_observations_valid']
    np.testing.assert_array_equal(valid_mask, [True, False, True, False])
    # Next observation at step 0 within episode 1 is obs 1
    assert arrays['next_observations'][0, 0] == 1.0
    # Next observation at step 2 within episode 2 is obs 3
    assert arrays['next_observations'][2, 0] == 11.0


def test_censored_segment_cut_midway_rejected():
    # Episode cut at step 2 when horizon=5 without terminal or timeout
    data = {
        'observations': np.zeros((2, 1)),
        'next_observations': np.zeros((2, 1)),
        'actions': np.zeros((2, 1)),
        'rewards': np.ones(2),
        'terminals': np.zeros(2),
        'timeouts': np.zeros(2),
    }
    report = audit_dataset(data, horizon=5, starts_at_reset=True)
    assert report['segments'][0]['end_reason'] == 'unclosed_tail'
    assert report['segments'][0]['rtg_eligible'] is False
    assert report['rtg_eligible_transitions'] == 0


def test_insufficient_information_to_certify_complete_rtg():
    # Dataset where starts_at_reset is False and no verified episode exists
    data = {
        'observations': np.zeros((5, 1)),
        'next_observations': np.zeros((5, 1)),
        'actions': np.zeros((5, 1)),
        'rewards': np.ones(5),
        'terminals': np.array([0, 0, 1, 0, 0]),
        'timeouts': np.zeros(5),
    }
    report = audit_dataset(data, horizon=5, starts_at_reset=False)
    # First segment (indices 0..2) has known_start=False
    assert report['segments'][0]['rtg_eligible'] is False
    # Second segment (indices 3..4) is unclosed tail
    assert report['segments'][1]['rtg_eligible'] is False
    assert report['rtg_eligible_transitions'] == 0


def test_load_dataset_preserves_missing_provenance(tmp_path):
    # Dataset missing next_observations and timeouts
    raw_data = {
        "observations": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "actions": np.zeros((3, 1)),
        "rewards": np.array([1.0, 2.0, 3.0]),
        "terminals": np.array([0, 0, 1]),
    }
    npz_path = tmp_path / "missing_fields.npz"
    np.savez(npz_path, **raw_data)

    loaded = load_dataset(npz_path)
    # Must NOT fabricate next_observations blindly
    assert "next_observations" not in loaded
    # Must NOT invent timeouts as zeros without recording that it was missing
    assert "timeouts" not in loaded or loaded.get("timeouts_source") == "missing"


def test_load_dataset_missing_file(tmp_path):
    missing_path = tmp_path / "non_existent.hdf5"
    with pytest.raises(FileNotFoundError):
        load_dataset(missing_path)
