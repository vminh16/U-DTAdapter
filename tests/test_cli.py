import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def invoke(script, *args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
        capture_output=True,
        text=True,
        env=env,
    )


def write_demo_dataset(path):
    obs = np.array(
        [[0.0], [1.0], [2.0], [10.0], [11.0], [12.0], [20.0], [21.0], [22.0],
         [30.0], [31.0], [32.0], [40.0]],
        dtype=np.float32,
    )
    next_obs = np.array(
        [[1.0], [2.0], [3.0], [11.0], [12.0], [13.0], [21.0], [22.0], [23.0],
         [31.0], [32.0], [33.0], [41.0]],
        dtype=np.float32,
    )
    np.savez(
        path,
        observations=obs,
        next_observations=next_obs,
        actions=np.zeros((len(obs), 1), dtype=np.float32),
        rewards=np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 5], dtype=np.float32),
        terminals=np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int32),
        timeouts=np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], dtype=np.int32),
    )


def test_audit_prepare_roundtrip(tmp_path):
    source = tmp_path / "demo.npz"
    write_demo_dataset(source)

    audit_path = tmp_path / "audit.json"
    result = invoke(
        "audit_data.py", source, "--horizon", 3, "--starts-at-reset",
        "--timeout-semantics", "horizon", "--output", audit_path,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(audit_path.read_text())
    assert report["gate_a"] == "REVIEW_REQUIRED"
    assert report["provenance"]["dataset_id"] == "unspecified"
    assert report["rtg_eligible_episodes"] == 4

    out = tmp_path / "prepared"
    result = invoke(
        "prepare_data.py", source, "--horizon", 3, "--starts-at-reset",
        "--timeout-semantics", "horizon", "--output", out,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["provenance"]["source_sha256"]) == 64
    assert manifest["provenance"]["command"][0] == "prepare"
    from udt.provenance import compute_file_sha256

    assert manifest["provenance"]["dataset_sha256"] == compute_file_sha256(
        out / "dataset.npz"
    )
    with np.load(out / "dataset.npz", allow_pickle=False) as data:
        assert len(data["returns_to_go"]) == 12
        assert set(data["episode_ids"][data["is_validation"]]).isdisjoint(
            data["episode_ids"][~data["is_validation"]]
        )

    result = invoke(
        "prepare_data.py", source, "--horizon", 3, "--starts-at-reset",
        "--timeout-semantics", "horizon", "--output", out,
    )
    assert result.returncode != 0


def test_audit_rejects_pickle_with_actionable_message(tmp_path):
    path = tmp_path / "old.pkl"
    path.write_bytes(b"not a pickle")
    result = invoke("audit_data.py", path, "--horizon", 3, "--output", tmp_path / "a.json")
    assert result.returncode != 0
    assert "NPZ or HDF5" in result.stderr
    assert not (tmp_path / "a.json").exists()


def test_check_readiness_reports_blockers_without_artifacts(tmp_path):
    report_path = tmp_path / "readiness.json"
    result = invoke("check_readiness.py", "--json", "--output", report_path)
    assert result.returncode == 1, result.stderr
    report = json.loads(report_path.read_text())
    assert report["ready"] is False
    assert "frozen_dt_parity" in report["blockers"]


def test_train_u_roundtrip(tmp_path):
    source = tmp_path / "demo.npz"
    write_demo_dataset(source)
    prepared_dir = tmp_path / "prep"
    assert invoke(
        "prepare_data.py", source, "--horizon", 3, "--starts-at-reset",
        "--timeout-semantics", "horizon", "--output", prepared_dir,
    ).returncode == 0

    checkpoint_dir = tmp_path / "checkpoints_u"
    result = invoke(
        "train_u.py", "--data", prepared_dir / "dataset.npz", "--horizon", 3,
        "--output", checkpoint_dir, "--num-models", 2, "--epochs", 2, "--seed", 42,
    )
    assert result.returncode == 0, result.stderr
    assert (checkpoint_dir / "dynamics_ensemble.pt").exists()
