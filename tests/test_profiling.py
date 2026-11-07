"""P3.2: latency profiling and scalar/vectorized return-evaluation parity."""

import numpy as np
import pytest
import torch

from udt.dynamics import DynamicsEnsemble
from udt.profiling import measure_controller_latency, measure_latency
from udt.return_cap import GlobalReturnCap, ReturnCapMLP, ReturnCapper


def _return_capper(mode="state_time", state_mean=None, state_std=None):
    torch.manual_seed(0)
    model = ReturnCapMLP(state_dim=3, hidden_dim=8, num_layers=2, mode=mode)
    return ReturnCapper(
        model,
        horizon=10,
        scale=1000.0,
        state_mean=state_mean,
        state_std=state_std,
    )


def test_scalar_and_vectorized_get_cap_match_for_shared_and_per_sample_steps():
    capper = _return_capper(
        state_mean=np.array([0.5, -1.0, 2.0], dtype=np.float32),
        state_std=np.array([2.0, 0.5, 3.0], dtype=np.float32),
    )
    rng = np.random.default_rng(3)
    states = rng.normal(size=(7, 3)).astype(np.float32)
    steps = np.arange(7)

    shared = capper.get_cap_batch(states, 4)
    scalar_shared = np.array([capper.get_cap(state, 4) for state in states], dtype=np.float32)
    np.testing.assert_allclose(shared, scalar_shared, atol=1e-6, rtol=1e-6)

    per_sample = capper.get_cap_batch(states, steps)
    scalar_per_sample = np.array(
        [capper.get_cap(state, step) for state, step in zip(states, steps)],
        dtype=np.float32,
    )
    np.testing.assert_allclose(per_sample, scalar_per_sample, atol=1e-6, rtol=1e-6)


def test_global_cap_batch_matches_scalar_cap():
    capper = ReturnCapper(GlobalReturnCap(2.5), horizon=10, scale=1000.0)
    states = np.zeros((4, 3), dtype=np.float32)
    batch = capper.get_cap_batch(states, 0)
    scalar = np.array([capper.get_cap(state, 0) for state in states], dtype=np.float32)
    np.testing.assert_allclose(batch, scalar)


def test_controller_vectorized_return_path_equals_per_state_scalar_path():
    torch.manual_seed(1)
    ensemble = DynamicsEnsemble(
        state_dim=3,
        act_dim=2,
        num_models=3,
        hidden_dim=16,
        state_mean=np.array([2.0, 1.0, -1.0], dtype=np.float32),
        state_std=np.array([0.5, 2.0, 3.0], dtype=np.float32),
        horizon=10,
        reward_scale=1000.0,
    )
    capper = _return_capper(
        state_mean=np.array([2.0, 1.0, -1.0], dtype=np.float32),
        state_std=np.array([0.5, 2.0, 3.0], dtype=np.float32),
    )
    norm_obs = np.zeros(3, dtype=np.float32)
    candidates = np.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.0]], dtype=np.float32)
    next_states, _, _ = ensemble.predict(norm_obs, candidates, 0.2)
    raw = ensemble.unnormalize_state(next_states)
    flat = raw.reshape(-1, 3)
    vectorized = capper.get_cap_batch(flat, 3).reshape(next_states.shape[:2])
    scalar = np.array(
        [
            [capper.get_cap(raw[head, candidate], 3) for candidate in range(raw.shape[1])]
            for head in range(raw.shape[0])
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(vectorized, scalar, atol=1e-6, rtol=1e-6)


def test_measure_latency_reports_percentiles_and_environment():
    calls = {"count": 0}

    def work():
        calls["count"] += 1

    stats = measure_latency(work, warmup=1, repetitions=5, synchronize=False)
    assert calls["count"] == 6
    assert stats["warmup"] == 1
    assert stats["repetitions"] == 5
    assert stats["p50_ms"] <= stats["p95_ms"]
    assert stats["min_ms"] <= stats["p50_ms"] <= stats["max_ms"]
    assert stats["environment"]["torch"]
    assert stats["environment"]["device"] == "cpu"


def test_measure_controller_latency_returns_dt_and_overhead():
    torch.manual_seed(2)
    ensemble = DynamicsEnsemble(
        state_dim=2,
        act_dim=1,
        num_models=2,
        hidden_dim=8,
        state_mean=np.zeros(2, dtype=np.float32),
        state_std=np.ones(2, dtype=np.float32),
        horizon=10,
        reward_scale=1000.0,
    )
    capper = ReturnCapper(GlobalReturnCap(1.0), horizon=10, scale=1000.0)
    adapter = _TinyAdapter()
    controller = _TinyController(adapter, ensemble, capper)
    report = measure_controller_latency(
        adapter,
        controller,
        obs=np.zeros(2, dtype=np.float32),
        rtg_token=1.0,
        warmup=1,
        repetitions=4,
        seed=5,
    )
    assert report["controller"]["p95_ms"] >= report["controller"]["p50_ms"]
    assert report["dt"]["p95_ms"] >= report["dt"]["p50_ms"]
    assert report["overhead_ms"]["mean_ms"] == pytest.approx(
        report["controller"]["mean_ms"] - report["dt"]["mean_ms"]
    )
    assert report["seed"] == 5
    assert report["environment"]["device"] == "cpu"


class _TinyAdapter:
    def __init__(self):
        self.calls = 0

    def reset(self):
        self.calls = 0

    def step(self, obs, rtg_token, step):
        self.calls += 1
        return np.zeros(1, dtype=np.float32)


class _TinyController:
    def __init__(self, adapter, ensemble, capper):
        self.adapter = adapter
        self.ensemble = ensemble
        self.capper = capper

    def reset(self):
        self.adapter.reset()

    def set_rng(self, rng):
        self.rng = rng

    def get_action(self, obs, rtg_token, step):
        self.adapter.step(obs, rtg_token, step)
        return np.zeros(1, dtype=np.float32), {"selected_index": 0}
