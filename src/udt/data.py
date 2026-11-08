"""Audit ordered transitions before producing state-time RTG labels.

Boundary declarations are caller-supplied provenance, not inferred from filenames.
Unknown starts, collection cuts and discontinuities never become complete episodes.
"""

from collections import Counter
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from udt.provenance import compute_file_sha256


MANDATORY_FIELDS = ('observations', 'actions', 'rewards', 'terminals')
OPTIONAL_FIELDS = ('next_observations', 'timeouts')


def validate_dataset(data: dict[str, Any]) -> dict[str, Any]:
    """Validate a flat, ordered dataset preserving field provenance."""
    missing = set(MANDATORY_FIELDS) - data.keys()
    if missing:
        raise ValueError(f'Missing dataset fields: {sorted(missing)}')

    arrays: dict[str, Any] = {}
    for key in MANDATORY_FIELDS:
        arrays[key] = np.asarray(data[key])

    n = len(arrays['rewards'])
    if not n:
        raise ValueError('Dataset is empty')

    # Validate 1D mandatory fields
    for key in ('rewards', 'terminals'):
        val = arrays[key]
        if val.ndim == 0 or len(val) != n or val.shape != (n,):
            raise ValueError(f'{key}: inconsistent transition count or shape')
        if not np.issubdtype(val.dtype, np.number) and val.dtype != np.bool_:
            raise ValueError(f'{key}: expected numeric data')
        if not np.isfinite(val).all():
            raise ValueError(f'{key}: non-finite data')

    if not np.isin(arrays['terminals'], [0, 1]).all():
        raise ValueError('terminals: flags must be 0 or 1')
    arrays['terminals'] = arrays['terminals'].astype(bool)

    # Validate 2D mandatory fields
    for key in ('observations', 'actions'):
        val = arrays[key]
        if val.ndim != 2 or len(val) != n or val.shape[1] == 0:
            raise ValueError(f'{key}: expected shape (N, dimension)')
        if not np.issubdtype(val.dtype, np.number) and val.dtype != np.bool_:
            raise ValueError(f'{key}: expected numeric data')
        if not np.isfinite(val).all():
            raise ValueError(f'{key}: non-finite data')

    # Provenance for timeouts
    has_original_timeouts = (
        'timeouts' in data and data.get('timeouts_source') != 'missing'
    )
    if has_original_timeouts:
        timeouts_arr = np.asarray(data['timeouts'])
        if timeouts_arr.ndim == 0 or len(timeouts_arr) != n or timeouts_arr.shape != (n,):
            raise ValueError('timeouts: inconsistent transition count or shape')
        if not np.issubdtype(timeouts_arr.dtype, np.number) and timeouts_arr.dtype != np.bool_:
            raise ValueError('timeouts: expected numeric data')
        if not np.isfinite(timeouts_arr).all():
            raise ValueError('timeouts: non-finite data')
        if not np.isin(timeouts_arr, [0, 1]).all():
            raise ValueError('timeouts: flags must be 0 or 1')
        arrays['timeouts'] = timeouts_arr.astype(bool)
        arrays['timeouts_source'] = 'original'
        arrays['has_original_timeouts'] = True
    else:
        arrays['timeouts'] = np.zeros(n, dtype=bool)
        arrays['timeouts_source'] = 'missing'
        arrays['has_original_timeouts'] = False

    # Provenance for next_observations
    has_original_next_obs = (
        'next_observations' in data and data.get('next_observations_source') != 'missing'
    )
    if has_original_next_obs:
        next_obs = np.asarray(data['next_observations'])
        if next_obs.ndim != 2 or len(next_obs) != n or next_obs.shape[1] == 0:
            raise ValueError('next_observations: expected shape (N, dimension)')
        if next_obs.shape != arrays['observations'].shape:
            raise ValueError('Observation and next-observation shapes differ')
        if not np.issubdtype(next_obs.dtype, np.number) and next_obs.dtype != np.bool_:
            raise ValueError('next_observations: expected numeric data')
        if not np.isfinite(next_obs).all():
            raise ValueError('next_observations: non-finite data')
        arrays['next_observations'] = next_obs
        arrays['next_observations_source'] = 'original'
        arrays['has_original_next_observations'] = True
    else:
        arrays['next_observations_source'] = 'missing'
        arrays['has_original_next_observations'] = False

    return arrays


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Load a dataset from an HDF5 or NPZ file safely without pickle.

    Args:
        path: Path to .hdf5, .h5, or .npz file.

    Returns:
        Dictionary mapping field names to NumPy arrays.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file is a pickle or an unsupported format.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    suffix = p.suffix.lower()
    if suffix in (".pkl", ".pickle"):
        raise ValueError("Pickle format (.pkl/.pickle) is not supported; use NPZ or HDF5.")

    data: dict[str, np.ndarray] = {}
    if suffix in (".hdf5", ".h5"):
        with h5py.File(p, "r") as f:
            for key in f.keys():
                # Only extract datasets, ignoring sub-groups such as 'infos' and 'metadata'
                if isinstance(f[key], h5py.Dataset):
                    data[key] = np.asarray(f[key])
    elif suffix == ".npz":
        with np.load(p, allow_pickle=False) as npz:
            for key in npz.files:
                data[key] = np.asarray(npz[key])
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}. Expected .hdf5, .h5, or .npz")

    return data


def load_prepared_dataset(
    path: str | Path,
    *,
    horizon: int | None = None,
    reward_scale: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load prepared arrays plus the sibling manifest and check shared settings."""
    data_path = Path(path)
    dataset = load_dataset(data_path)
    manifest_path = data_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}

    if horizon is not None and manifest.get("horizon") is not None:
        if int(manifest["horizon"]) != int(horizon):
            raise ValueError(
                f"Horizon mismatch: manifest horizon ({manifest['horizon']}) != requested horizon ({horizon})"
            )
    if reward_scale is not None and manifest.get("reward_scale") is not None:
        if not np.isclose(float(manifest["reward_scale"]), float(reward_scale)):
            raise ValueError(
                "Reward scale mismatch: manifest reward_scale "
                f"({manifest['reward_scale']}) != requested reward_scale ({reward_scale})"
            )
    return dataset, manifest


def verify_manifest_normalization(
    dataset: dict[str, np.ndarray],
    manifest: dict[str, Any],
    *,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Prove that manifest normalization statistics came from the training split."""
    mean = manifest.get("state_mean")
    std = manifest.get("state_std")
    if mean is None or std is None:
        raise ValueError("Manifest is missing state_mean/state_std")
    if "is_validation" not in dataset or "observations" not in dataset:
        raise ValueError("Dataset must contain observations and is_validation")

    train_obs = np.asarray(dataset["observations"])[~np.asarray(dataset["is_validation"])]
    if len(train_obs) == 0:
        raise ValueError("Training split is empty; cannot verify normalization")
    expected_mean = train_obs.mean(axis=0, dtype=np.float64)
    expected_std = np.maximum(train_obs.std(axis=0, dtype=np.float64), 1e-6)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if mean.shape != expected_mean.shape or std.shape != expected_std.shape:
        raise ValueError("Manifest normalization shape does not match observations")
    if not np.allclose(mean, expected_mean, rtol=0, atol=atol):
        raise ValueError("Manifest state_mean does not match the training split")
    if not np.allclose(std, expected_std, rtol=0, atol=atol):
        raise ValueError("Manifest state_std does not match the training split")
    return {
        "source": "prepared_manifest_train_split",
        "verified": True,
        "atol": atol,
    }


def audit_dataset_file(
    path: str | Path,
    *,
    horizon: int,
    starts_at_reset: bool = False,
    timeout_semantics: str = "unknown",
    dataset_id: str = "unspecified",
    continuity_atol: float = 1e-6,
) -> dict[str, Any]:
    """Audit a raw dataset file and attach source provenance."""
    data_path = Path(path)
    report = audit_dataset(
        load_dataset(data_path),
        horizon=horizon,
        starts_at_reset=starts_at_reset,
        timeout_semantics=timeout_semantics,
        continuity_atol=continuity_atol,
    )
    report["provenance"] = {
        "dataset_id": dataset_id,
        "source_sha256": compute_file_sha256(data_path),
    }
    return report


def prepare_dataset_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    horizon: int,
    starts_at_reset: bool = False,
    timeout_semantics: str = "unknown",
    seed: int = 0,
    validation_fraction: float = 0.1,
    reward_scale: float = 1000.0,
    dataset_id: str = "unspecified",
    continuity_atol: float = 1e-6,
) -> dict[str, Any]:
    """Prepare, verify, and persist a dataset without overwriting an existing one."""
    source = Path(input_path)
    target = Path(output_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Output directory {target} already exists and is not empty.")

    arrays, manifest = prepare_dataset(
        load_dataset(source),
        horizon=horizon,
        starts_at_reset=starts_at_reset,
        timeout_semantics=timeout_semantics,
        seed=seed,
        validation_fraction=validation_fraction,
        reward_scale=reward_scale,
        continuity_atol=continuity_atol,
    )
    manifest["dataset_id"] = dataset_id
    manifest["provenance"] = {
        "command": ["prepare", str(source)],
        "source_sha256": compute_file_sha256(source),
    }

    target.mkdir(parents=True, exist_ok=True)
    np.savez(target / "dataset.npz", **arrays)
    manifest["provenance"]["dataset_sha256"] = compute_file_sha256(target / "dataset.npz")
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def audit_dataset(data: dict[str, np.ndarray], *, horizon: int,
                  starts_at_reset: bool = False,
                  timeout_semantics: str = 'unknown',
                  continuity_atol: float = 1e-6) -> dict[str, Any]:
    """Return an integrity report, not an automatic research Gate A pass.

    A horizon timeout is trusted only if explicitly declared and observed at H.
    No implicit boundary is invented every H steps. Segment returns are observed
    sums, not counterfactual full returns for censored segments.
    """
    if horizon < 1 or timeout_semantics not in ('unknown', 'horizon'):
        raise ValueError('Require positive horizon and timeout_semantics unknown/horizon')
    if not np.isfinite(continuity_atol) or continuity_atol < 0:
        raise ValueError('continuity_atol must be finite and nonnegative')
    arrays = validate_dataset(data)
    n = len(arrays['rewards'])
    has_timeouts = arrays['has_original_timeouts']
    has_next_obs = arrays['has_original_next_observations']
    segments = []
    start = 0
    known_start = bool(starts_at_reset)
    for i in range(n):
        terminal = bool(arrays['terminals'][i])
        timeout = bool(arrays['timeouts'][i]) if has_timeouts else False
        length = i - start + 1
        next_start_known = False
        reason = None
        if terminal and timeout:
            reason = 'ambiguous_terminal_timeout'
        elif terminal:
            reason = 'terminal'
            next_start_known = True
        elif timeout:
            if timeout_semantics == 'horizon' and known_start and length == horizon and has_timeouts:
                reason = 'horizon'
                next_start_known = True
            else:
                reason = 'unverified_timeout'
        elif has_next_obs and i + 1 < n and not np.allclose(
            arrays['next_observations'][i], arrays['observations'][i + 1],
            rtol=0, atol=continuity_atol,
        ):
            reason = 'discontinuity'
        elif i == n - 1:
            reason = 'unclosed_tail'
        if reason is None:
            continue
        eligible = known_start and length <= horizon and reason in ('terminal', 'horizon')
        segments.append({
            'id': len(segments), 'start': start, 'stop': i + 1,
            'length': length, 'known_start': known_start, 'end_reason': reason,
            'rtg_eligible': eligible,
            'observed_reward_sum': float(arrays['rewards'][start:i + 1].sum()),
        })
        start = i + 1
        known_start = next_start_known
    eligible = [segment for segment in segments if segment['rtg_eligible']]
    retained = sum(segment['length'] for segment in eligible)
    warnings = []
    if not has_timeouts:
        warnings.append('Missing timeouts: horizon endings cannot be verified.')
    if not has_next_obs:
        warnings.append('Missing original next_observations; inter-step continuity cannot be verified from next-state.')
    if not starts_at_reset:
        warnings.append('First segment has unknown episode start/time.')
    if retained < n:
        warnings.append('Filtered labels change the training population; inspect selection bias.')
    if any(segment['length'] > horizon for segment in segments):
        warnings.append('Segments longer than declared horizon: inspect boundary metadata.')
    return {
        'schema_version': 1, 'transitions': n, 'horizon': horizon,
        'timeout_semantics': timeout_semantics, 'starts_at_reset': starts_at_reset,
        'continuity_atol': continuity_atol,
        'has_original_timeouts': has_timeouts,
        'timeouts_source': arrays['timeouts_source'],
        'has_original_next_observations': has_next_obs,
        'next_observations_source': arrays['next_observations_source'],
        'segment_count': len(segments), 'rtg_eligible_episodes': len(eligible),
        'rtg_eligible_transitions': retained, 'excluded_transitions': n - retained,
        'retained_fraction': retained / n,
        'end_reason_counts': dict(Counter(s['end_reason'] for s in segments)),
        'warnings': warnings, 'segments': segments,
        'gate_a': 'REVIEW_REQUIRED',
    }


def prepare_dataset(data: dict[str, np.ndarray], *, horizon: int,
                    starts_at_reset: bool = False,
                    timeout_semantics: str = 'unknown', seed: int = 0,
                    validation_fraction: float = 0.1, reward_scale: float = 1000.,
                    continuity_atol: float = 1e-6) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Prepare complete episodes only; preserve raw inputs and train-only statistics.

    This initial R dataset intentionally excludes uncertain segments from both
    splits. The raw source remains the input to any later dynamics-data pipeline.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must lie between 0 and 1')
    if not np.isfinite(reward_scale) or reward_scale <= 0:
        raise ValueError('reward_scale must be finite and positive')
    report = audit_dataset(data, horizon=horizon, starts_at_reset=starts_at_reset,
                           timeout_semantics=timeout_semantics, continuity_atol=continuity_atol)
    episodes = [s for s in report['segments'] if s['rtg_eligible']]
    if len(episodes) < 2:
        raise ValueError('Need at least two verified episodes for a trajectory split')
    arrays = validate_dataset(data)
    has_next_obs = arrays['has_original_next_observations']
    ids = np.array([e['id'] for e in episodes])
    nval = min(len(ids) - 1, max(1, int(np.ceil(len(ids) * validation_fraction))))
    val_ids = np.random.default_rng(seed).permutation(ids)[:nval]
    indices, rtgs, timesteps, episode_ids = [], [], [], []
    inferred_next_obs = []
    next_obs_valid = []

    for episode in episodes:
        rows = np.arange(episode['start'], episode['stop'])
        rewards = arrays['rewards'][rows].astype(np.float64) / reward_scale
        indices.extend(rows)
        rtgs.extend(np.cumsum(rewards[::-1])[::-1])
        timesteps.extend(range(len(rows)))
        episode_ids.extend([episode['id']] * len(rows))

        if has_next_obs:
            next_obs_valid.extend([True] * len(rows))
        else:
            ep_obs = arrays['observations'][rows]
            L = len(rows)
            for k in range(L):
                if k < L - 1:
                    inferred_next_obs.append(ep_obs[k + 1])
                    next_obs_valid.append(True)
                else:
                    # Terminal transition: post-terminal state is not observed
                    inferred_next_obs.append(ep_obs[k].copy())
                    next_obs_valid.append(False)

    rows = np.asarray(indices, dtype=np.int64)
    result = {
        'observations': arrays['observations'][rows],
        'actions': arrays['actions'][rows],
        'rewards': arrays['rewards'][rows],
        'terminals': arrays['terminals'][rows],
        'timeouts': arrays['timeouts'][rows],
        'source_indices': rows,
        'returns_to_go': np.asarray(rtgs),
        'timesteps': np.asarray(timesteps, dtype=np.int64),
        'episode_ids': np.asarray(episode_ids, dtype=np.int64),
        'scaled_rewards': arrays['rewards'][rows].astype(np.float64) / reward_scale,
    }
    if has_next_obs:
        result['next_observations'] = arrays['next_observations'][rows]
        result['next_observations_valid'] = np.ones(len(rows), dtype=bool)
    else:
        result['next_observations'] = np.asarray(inferred_next_obs)
        result['next_observations_valid'] = np.asarray(next_obs_valid, dtype=bool)

    result['is_validation'] = np.isin(result['episode_ids'], val_ids)
    train_obs = result['observations'][~result['is_validation']].astype(np.float64)
    manifest = {
        'schema_version': 1, 'seed': seed, 'horizon': horizon, 'reward_scale': reward_scale,
        'requested_validation_fraction': validation_fraction,
        'train_episode_ids': sorted(set(ids.tolist()) - set(val_ids.tolist())),
        'validation_episode_ids': sorted(val_ids.tolist()),
        'state_mean': train_obs.mean(axis=0).tolist(),
        'state_std': np.maximum(train_obs.std(axis=0), 1e-6).tolist(),
        'retained_transitions': len(rows), 'dropped_transitions': len(arrays['rewards']) - len(rows),
        'has_original_next_observations': has_next_obs,
        'next_observations_source': 'original' if has_next_obs else 'inferred',
        'valid_next_observations_count': int(result['next_observations_valid'].sum()),
        'has_original_timeouts': arrays['has_original_timeouts'],
        'timeouts_source': arrays['timeouts_source'],
        'label_population': 'verified complete episodes only; selection bias not corrected',
        'audit': report,
    }
    return result, manifest
