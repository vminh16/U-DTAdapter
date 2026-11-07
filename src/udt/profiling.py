"""Latency and cost profiling for the U-DT controller.

The RRD/EXPERIMENT protocol asks for measured (not asserted) inference cost:
model latency split from simulator stepping, p50/p95 percentiles, warm-up,
and the hardware/software environment. This module provides small, dependency-
free helpers used by tests and the readiness command.
"""

from __future__ import annotations

import platform
import time
from typing import Any, Callable

import numpy as np
import torch


def _synchronize(device: str | torch.device | None) -> None:
    if device is None:
        return
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


def environment_metadata(device: str | torch.device = "cpu") -> dict[str, Any]:
    """Describe the measurement environment so numbers stay traceable."""
    resolved = torch.device(device)
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(resolved),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if resolved.type == "cuda" and torch.cuda.is_available():
        metadata["cuda_device_name"] = torch.cuda.get_device_name(resolved)
    return metadata


def _summarize(samples_ms: list[float]) -> dict[str, float]:
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "std_ms": float(np.std(values)),
    }


def measure_latency(
    fn: Callable[[], Any],
    *,
    warmup: int = 5,
    repetitions: int = 50,
    device: str | torch.device = "cpu",
    synchronize: bool = True,
) -> dict[str, Any]:
    """Time ``fn`` and report mean/p50/p95 latency in milliseconds.

    Args:
        fn: Callable performing exactly one inference/control step.
        warmup: Untimed calls used to populate caches and CUDA kernels.
        repetitions: Timed calls.
        device: Device used for optional CUDA synchronization.
        synchronize: Synchronize CUDA at measurement boundaries when true.

    Returns:
        Summary statistics plus warm-up/repetition counts and environment.
    """
    if warmup < 0:
        raise ValueError(f"warmup must be nonnegative, got {warmup}")
    if repetitions <= 0:
        raise ValueError(f"repetitions must be positive, got {repetitions}")

    for _ in range(warmup):
        fn()

    samples_ms: list[float] = []
    for _ in range(repetitions):
        if synchronize:
            _synchronize(device)
        start = time.perf_counter_ns()
        fn()
        if synchronize:
            _synchronize(device)
        samples_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)

    summary = _summarize(samples_ms)
    summary.update(
        {
            "warmup": int(warmup),
            "repetitions": int(repetitions),
            "samples_ms": samples_ms,
            "environment": environment_metadata(device),
        }
    )
    return summary


def measure_controller_latency(
    adapter: Any,
    controller: Any,
    *,
    obs: np.ndarray,
    rtg_token: float,
    step: int = 0,
    warmup: int = 5,
    repetitions: int = 50,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> dict[str, Any]:
    """Measure DT-only and full-controller latency and their overhead.

    Each timed call resets the controller/adapter so the measurement covers one
    fresh control step, not a growing context window. Simulator stepping is
    intentionally excluded.
    """
    obs = np.asarray(obs, dtype=np.float32)
    rng = np.random.default_rng(seed)
    setter = getattr(controller, "set_rng", None)
    if callable(setter):
        setter(rng)

    def dt_call() -> Any:
        adapter.reset()
        return adapter.step(obs=obs, rtg_token=rtg_token, step=step)

    def controller_call() -> Any:
        controller.reset()
        return controller.get_action(obs=obs, rtg_token=rtg_token, step=step)

    dt_stats = measure_latency(
        dt_call, warmup=warmup, repetitions=repetitions, device=device
    )
    controller_stats = measure_latency(
        controller_call, warmup=warmup, repetitions=repetitions, device=device
    )

    return {
        "dt": dt_stats,
        "controller": controller_stats,
        "overhead_ms": {
            "mean_ms": controller_stats["mean_ms"] - dt_stats["mean_ms"],
            "p50_ms": controller_stats["p50_ms"] - dt_stats["p50_ms"],
            "p95_ms": controller_stats["p95_ms"] - dt_stats["p95_ms"],
        },
        "obs_shape": list(obs.shape),
        "rtg_token": float(rtg_token),
        "step": int(step),
        "seed": int(seed),
        "environment": controller_stats["environment"],
        "note": "Simulator stepping excluded; each call resets episode context.",
    }
