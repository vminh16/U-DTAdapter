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


def test_load_dataset_missing_file(tmp_path):
    missing_path = tmp_path / "non_existent.hdf5"
    with pytest.raises(FileNotFoundError):
        load_dataset(missing_path)
