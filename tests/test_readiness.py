"""P3.3: pilot readiness checks."""

import copy
import json

import numpy as np
import torch

from udt.readiness import (
    check_pilot_readiness,
    compute_file_sha256,
    format_readiness_report,
    readiness_json,
)


def _make_artifacts(tmp_path):
    dataset_path = tmp_path / "dataset.npz"
    np.savez(dataset_path, observations=np.zeros((4, 2), dtype=np.float32))
    dataset_sha = compute_file_sha256(dataset_path)

    manifest = {
        "horizon": 10,
        "reward_scale": 1000.0,
        "seed": 7,
        "train_episode_ids": [0, 1],
        "validation_episode_ids": [2],
        "provenance": {"source_sha256": dataset_sha},
    }

    dynamics_metrics = {
        "base_seed": 7,
        "training_config": {"seed": 7, "num_models": 2},
        "data_summary": {
            "train_episode_ids": [0, 1],
            "validation_episode_ids": [2],
        },
        "models": [
            {
                "bootstrap": {
                    "bootstrap_seed": index,
                    "sampled_episode_ids": [0, 1],
                    "bootstrap_transitions": 10,
                }
            }
            for index in range(2)
        ],
    }
    dynamics_payload = {
        "kind": "udt_dynamics_ensemble",
        "format_version": 1,
        "architecture": {"state_dim": 2, "act_dim": 1, "num_models": 2},
        "normalization": {
            "state_mean": [0.0, 0.0],
            "state_std": [1.0, 1.0],
            "coordinate_system": "raw_observation",
            "source": "dynamics_training_split",
            "verified": True,
        },
        "horizon": 10,
        "reward_scale": 1000.0,
        "init_seeds": [0, 1],
        "bootstrap": [{"bootstrap_seed": 0}, {"bootstrap_seed": 1}],
        "dataset_provenance": {"dataset_sha256": dataset_sha},
        "metrics": dynamics_metrics,
        "models_state_dict": [{}, {}],
    }
    return_payload = {
        "model_state_dict": {"weight": torch.zeros(1)},
        "mode": "state_time",
        "horizon": 10,
        "tau": 0.9,
        "margin": 0.0,
        "scale": 1000.0,
        "state_mean": [0.0, 0.0],
        "state_std": [1.0, 1.0],
        "metrics": {
            "best_val_loss": 0.5,
            "best_val_coverage": 0.9,
            "restored_checkpoint_evaluation": {"loss": 0.5, "coverage": 0.9},
        },
    }
    dynamics_checkpoint = tmp_path / "dynamics.pt"
    return_checkpoint = tmp_path / "return.pt"
    torch.save(dynamics_payload, dynamics_checkpoint)
    torch.save(return_payload, return_checkpoint)

    dt_parity = {
        "passed": True,
        "model_source": "real",
        "max_abs_difference": 0.0,
        "tolerance": 1e-6,
    }
    return {
        "dataset_path": dataset_path,
        "manifest": manifest,
        "dynamics_payload": dynamics_payload,
        "return_payload": return_payload,
        "dynamics_checkpoint": dynamics_checkpoint,
        "return_checkpoint": return_checkpoint,
        "dt_parity": dt_parity,
    }


def _run(artifacts, **overrides):
    kwargs = {
        "manifest": artifacts["manifest"],
        "dataset_path": artifacts["dataset_path"],
        "dynamics_checkpoint": artifacts["dynamics_checkpoint"],
        "return_checkpoint": artifacts["return_checkpoint"],
        "dynamics_payload": artifacts["dynamics_payload"],
        "return_payload": artifacts["return_payload"],
        "dt_parity": artifacts["dt_parity"],
    }
    kwargs.update(overrides)
    return check_pilot_readiness(**kwargs)


def test_complete_artifacts_are_ready(tmp_path):
    artifacts = _make_artifacts(tmp_path)
    report = _run(artifacts)
    assert report["ready"], report
    assert report["blockers"] == []
    for name in (
        "provenance",
        "splits",
        "final_test_isolation",
        "frozen_dt_parity",
        "bootstrap",
        "normalization",
        "components",
        "traceability",
        "consistency",
    ):
        assert report["checks"][name]["status"] == "pass", (name, report["checks"][name])


def test_complete_artifacts_load_from_checkpoint_files(tmp_path):
    artifacts = _make_artifacts(tmp_path)
    report = _run(artifacts, dynamics_payload=None, return_payload=None)
    assert report["ready"], report


def test_dataset_hash_mismatch_fails_provenance(tmp_path):
    artifacts = _make_artifacts(tmp_path)
    artifacts["dataset_path"].write_bytes(b"tampered")
    report = _run(artifacts)
    assert not report["ready"]
    assert report["checks"]["provenance"]["status"] == "fail"
    assert "provenance" in report["blockers"]


def test_missing_provenance_fails(tmp_path):
    artifacts = _make_artifacts(tmp_path)
    artifacts["manifest"]["provenance"] = {}
    artifacts["dynamics_payload"]["dataset_provenance"] = {}
    report = _run(artifacts)
    assert report["checks"]["provenance"]["status"] == "fail"


def test_train_validation_overlap_fails(tmp_path):
    artifacts = _make_artifacts(tmp_path)
    artifacts["manifest"]["validation_episode_ids"] = [1, 2]
    report = _run(artifacts)
    assert report["checks"]["splits"]["status"] == "fail"


def test_test_episode_leakage_fails(tmp_path):
    artifacts = _make_artifacts(tmp_path)
    artifacts["manifest"]["test_episode_ids"] = [1, 5]
    report = _run(artifacts)
    assert report["checks"]["final_test_isolation"]["status"] == "fail"


def test_isolated_test_split_passes(tmp_path):
    artifacts = _make_artifacts(tmp_path)
    artifacts["manifest"]["test_episode_ids"] = [5, 6]
    report = _run(artifacts)
    assert report["checks"]["final_test_isolation"]["status"] == "pass"


def test_bootstrap_using_validation_episode_fails(tmp_path):
    artifacts = _make_artifacts(tmp_path)
    artifacts["dynamics_payload"]["metrics"]["models"][0]["bootstrap"][
        "sampled_episode_ids"
    ] = [0, 2]
    report = _run(artifacts)
    assert report["checks"]["bootstrap"]["status"] == "fail"


def test_missing_bootstrap_seed_fails(tmp_path):
    artifacts = _make_artifacts(tmp_path)
    artifacts["dynamics_payload"]["metrics"]["models"][1]["bootstrap"][
        "bootstrap_seed"
    ] = None
    report = _run(artifacts)
    assert report["checks"]["bootstrap"]["status"] == "fail"


def test_invalid_normalization_fails(tmp_path):
    artifacts = _make_artifacts(tmp_path)
    artifacts["dynamics_payload"]["normalization"]["state_std"] = [1.0, 0.0]
    report = _run(artifacts)
    assert report["checks"]["normalization"]["status"] == "fail"


def test_missing_parity_is_blocked_not_passed(tmp_path):
    artifacts = _make_artifacts(tmp_path)
    report = _run(artifacts, dt_parity=None)
    assert not report["ready"]
    assert report["checks"]["frozen_dt_parity"]["status"] == "blocked"


def test_mock_parity_is_blocked_for_real_pilot(tmp_path):
    artifacts = _make_artifacts(tmp_path)
    artifacts["dt_parity"]["model_source"] = "mock"
    report = _run(artifacts)
    assert report["checks"]["frozen_dt_parity"]["status"] == "blocked"


def test_failed_parity_fails(tmp_path):
    artifacts = _make_artifacts(tmp_path)
    artifacts["dt_parity"]["passed"] = False
    report = _run(artifacts)
    assert report["checks"]["frozen_dt_parity"]["status"] == "fail"


def test_missing_return_component_blocks_b3_and_b5(tmp_path):
    artifacts = _make_artifacts(tmp_path)
    report = check_pilot_readiness(
        manifest=artifacts["manifest"],
        dataset_path=artifacts["dataset_path"],
        dynamics_checkpoint=artifacts["dynamics_checkpoint"],
        dt_parity=artifacts["dt_parity"],
        required_variants=("B0", "B3", "B5"),
    )
    assert report["checks"]["components"]["status"] == "blocked"
    assert not report["ready"]


def test_missing_dynamics_blocks_b4(tmp_path):
    artifacts = _make_artifacts(tmp_path)
    report = check_pilot_readiness(
        manifest=artifacts["manifest"],
        dataset_path=artifacts["dataset_path"],
        return_checkpoint=artifacts["return_checkpoint"],
        dt_parity=artifacts["dt_parity"],
        required_variants=("B0", "B4"),
    )
    assert report["checks"]["components"]["status"] == "blocked"


def test_single_dynamics_head_warns_about_disagreement(tmp_path):
    artifacts = _make_artifacts(tmp_path)
    artifacts["dynamics_payload"]["architecture"]["num_models"] = 1
    artifacts["dynamics_payload"]["metrics"]["models"] = (
        artifacts["dynamics_payload"]["metrics"]["models"][:1]
    )
    artifacts["dynamics_payload"]["init_seeds"] = [0]
    artifacts["dynamics_payload"]["bootstrap"] = [{"bootstrap_seed": 0}]
    report = _run(artifacts)
    assert report["checks"]["components"]["status"] == "warn"
    assert report["ready"]


def test_incomplete_return_metrics_fail_traceability(tmp_path):
    artifacts = _make_artifacts(tmp_path)
    artifacts["return_payload"]["metrics"].pop("best_val_coverage")
    report = _run(artifacts)
    assert report["checks"]["traceability"]["status"] == "fail"


def test_horizon_mismatch_fails_consistency(tmp_path):
    artifacts = _make_artifacts(tmp_path)
    artifacts["manifest"]["horizon"] = 11
    report = _run(artifacts)
    assert report["checks"]["consistency"]["status"] == "fail"


def test_missing_checkpoint_blocks_without_crashing(tmp_path):
    artifacts = _make_artifacts(tmp_path)
    report = _run(
        artifacts,
        dynamics_checkpoint=tmp_path / "missing.pt",
        dynamics_payload=None,
    )
    assert report["checks"]["traceability"]["status"] == "blocked"
    problems = report["checks"]["traceability"]["evidence"]["problems"]
    assert any("dynamics" in problem for problem in problems)


def test_report_rendering_and_json(tmp_path):
    artifacts = _make_artifacts(tmp_path)
    report = _run(artifacts)
    text = format_readiness_report(report)
    assert "PILOT READINESS: READY" in text
    payload = json.loads(readiness_json(report))
    assert payload["ready"] is True
