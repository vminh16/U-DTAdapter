"""Variant-scoped readiness: B0 and B3 must not demand unrelated artifacts."""

import numpy as np
import torch

from udt.readiness import check_pilot_readiness
from udt.provenance import compute_file_sha256


def _dataset(tmp_path):
    path = tmp_path / "dataset.npz"
    np.savez(path, observations=np.zeros((4, 2), dtype=np.float32))
    return path


def _manifest(path):
    return {
        "horizon": 10,
        "reward_scale": 1000.0,
        "seed": 7,
        "train_episode_ids": [0, 1],
        "validation_episode_ids": [2],
        "provenance": {"source_sha256": compute_file_sha256(path)},
    }


PARITY = {
    "passed": True,
    "model_source": "real",
    "max_abs_difference": 0.0,
    "tolerance": 1e-6,
}


def test_b0_does_not_require_return_or_dynamics_artifacts(tmp_path):
    dataset = _dataset(tmp_path)
    report = check_pilot_readiness(
        manifest=_manifest(dataset),
        dataset_path=dataset,
        dt_parity=PARITY,
        required_variants=("B0",),
    )
    assert report["ready"], report
    assert report["checks"]["bootstrap"]["status"] == "pass"
    assert report["checks"]["normalization"]["status"] == "pass"
    assert report["checks"]["traceability"]["status"] == "pass"
    assert "No U variant" in report["checks"]["bootstrap"]["detail"]


def test_b3_requires_return_but_not_dynamics(tmp_path):
    dataset = _dataset(tmp_path)
    return_path = tmp_path / "return.pt"
    torch.save(
        {
            "model_state_dict": {"weight": torch.zeros(1)},
            "mode": "state_time",
            "horizon": 10,
            "scale": 1000.0,
            "margin": 0.0,
            "state_mean": [0.0, 0.0],
            "state_std": [1.0, 1.0],
            "metrics": {
                "best_val_loss": 0.5,
                "best_val_coverage": 0.9,
                "restored_checkpoint_evaluation": {"loss": 0.5, "coverage": 0.9},
            },
        },
        return_path,
    )
    report = check_pilot_readiness(
        manifest=_manifest(dataset),
        dataset_path=dataset,
        return_checkpoint=return_path,
        dt_parity=PARITY,
        required_variants=("B3",),
    )
    assert report["ready"], report
    assert report["checks"]["bootstrap"]["status"] == "pass"
    assert report["checks"]["components"]["status"] == "pass"


def test_b4_still_requires_dynamics(tmp_path):
    dataset = _dataset(tmp_path)
    return_path = tmp_path / "return.pt"
    torch.save({"model_state_dict": {"weight": torch.zeros(1)}}, return_path)
    report = check_pilot_readiness(
        manifest=_manifest(dataset),
        dataset_path=dataset,
        return_checkpoint=return_path,
        dt_parity=PARITY,
        required_variants=("B4",),
    )
    assert report["checks"]["components"]["status"] == "blocked"
    assert "B4" in report["checks"]["components"]["detail"]
