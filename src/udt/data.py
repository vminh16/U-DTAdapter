"""Audit ordered transitions before producing state-time RTG labels.

Boundary declarations are caller-supplied provenance, not inferred from filenames.
Unknown starts, collection cuts and discontinuities never become complete episodes.
"""

from collections import Counter
from typing import Any

import numpy as np


FIELDS = ('observations', 'next_observations', 'actions', 'rewards', 'terminals')


def validate_dataset(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Validate a flat, ordered dataset without silently converting bad flags."""
    missing = set(FIELDS) - data.keys()
    if missing:
        raise ValueError(f'Missing dataset fields: {sorted(missing)}')
    arrays = {key: np.asarray(data[key]) for key in FIELDS}
    arrays['timeouts'] = np.asarray(data.get('timeouts', np.zeros(len(arrays['rewards']))))
    n = len(arrays['rewards'])
    if not n:
        raise ValueError('Dataset is empty')
    for key, value in arrays.items():
        if value.ndim == 0 or len(value) != n:
            raise ValueError(f'{key}: inconsistent transition count')
        if not np.issubdtype(value.dtype, np.number) and value.dtype != np.bool_:
            raise ValueError(f'{key}: expected numeric data')
        if not np.isfinite(value).all():
            raise ValueError(f'{key}: non-finite data')
    for key in ('rewards', 'terminals', 'timeouts'):
        if arrays[key].shape != (n,):
            raise ValueError(f'{key}: expected shape (N,)')
    for key in ('observations', 'next_observations', 'actions'):
        if arrays[key].ndim != 2 or arrays[key].shape[1] == 0:
            raise ValueError(f'{key}: expected shape (N, dimension)')
    if arrays['observations'].shape != arrays['next_observations'].shape:
        raise ValueError('Observation and next-observation shapes differ')
    for key in ('terminals', 'timeouts'):
        if not np.isin(arrays[key], [0, 1]).all():
            raise ValueError(f'{key}: flags must be 0 or 1')
        arrays[key] = arrays[key].astype(bool)
    return arrays


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
    segments = []
    start = 0
    known_start = bool(starts_at_reset)
    for i in range(n):
        terminal = bool(arrays['terminals'][i])
        timeout = bool(arrays['timeouts'][i])
        length = i - start + 1
        next_start_known = False
        reason = None
        if terminal and timeout:
            reason = 'ambiguous_terminal_timeout'
        elif terminal:
            reason = 'terminal'
            next_start_known = True
        elif timeout:
            if timeout_semantics == 'horizon' and known_start and length == horizon:
                reason = 'horizon'
                next_start_known = True
            else:
                reason = 'unverified_timeout'
        elif i + 1 < n and not np.allclose(
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
    if 'timeouts' not in data:
        warnings.append('Missing timeouts: horizon endings cannot be verified.')
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
    ids = np.array([e['id'] for e in episodes])
    nval = min(len(ids) - 1, max(1, int(np.ceil(len(ids) * validation_fraction))))
    val_ids = np.random.default_rng(seed).permutation(ids)[:nval]
    indices, rtgs, timesteps, episode_ids = [], [], [], []
    for episode in episodes:
        rows = np.arange(episode['start'], episode['stop'])
        rewards = arrays['rewards'][rows].astype(np.float64) / reward_scale
        indices.extend(rows)
        rtgs.extend(np.cumsum(rewards[::-1])[::-1])
        timesteps.extend(range(len(rows)))
        episode_ids.extend([episode['id']] * len(rows))
    rows = np.asarray(indices, dtype=np.int64)
    result = {k: v[rows] for k, v in arrays.items()}
    result.update({
        'source_indices': rows, 'returns_to_go': np.asarray(rtgs),
        'timesteps': np.asarray(timesteps, dtype=np.int64),
        'episode_ids': np.asarray(episode_ids, dtype=np.int64),
        'scaled_rewards': arrays['rewards'][rows].astype(np.float64) / reward_scale,
    })
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
        'label_population': 'verified complete episodes only; selection bias not corrected',
        'audit': report,
    }
    return result, manifest
