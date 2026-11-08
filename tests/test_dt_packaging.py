import os
import sys
from pathlib import Path
import pytest
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from udt.dt_integration import (
    RealDTUnavailable,
    _import_baseline_dt,
    load_decision_transformer,
    train_decision_transformer,
)


def test_pyproject_toml_declares_baseline_dt_optional_dependency():
    """pyproject.toml must declare baseline optional dependencies."""
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert pyproject_path.exists()
    content = tomllib.loads(pyproject_path.read_text())

    opt_deps = content.get("project", {}).get("optional-dependencies", {})
    assert "baseline-dt" in opt_deps or "baseline" in opt_deps or "dt" in opt_deps
    key = "baseline-dt" if "baseline-dt" in opt_deps else ("baseline" if "baseline" in opt_deps else "dt")
    deps = opt_deps[key]
    assert any("transformers" in dep for dep in deps)


def test_import_baseline_dt_accepts_explicit_path():
    """_import_baseline_dt respects an explicit baseline_root parameter."""
    real_baseline = Path(__file__).resolve().parents[1] / "baselines" / "decision_transformer" / "gym"
    dt_cls, trainer_cls = _import_baseline_dt(baseline_root=real_baseline)
    assert dt_cls.__name__ == "DecisionTransformer"
    assert trainer_cls.__name__ == "SequenceTrainer"


def test_import_baseline_dt_rejects_nonexistent_explicit_path(tmp_path):
    """Passing a nonexistent baseline_root raises RealDTUnavailable."""
    nonexistent = tmp_path / "nowhere" / "gym"
    with pytest.raises(RealDTUnavailable, match="not found"):
        _import_baseline_dt(baseline_root=nonexistent)


def test_import_baseline_dt_respects_env_var(monkeypatch, tmp_path):
    """_import_baseline_dt respects UDT_BASELINE_DT_ROOT environment variable."""
    nonexistent = tmp_path / "env_nonexistent"
    monkeypatch.setenv("UDT_BASELINE_DT_ROOT", str(nonexistent))
    with pytest.raises(RealDTUnavailable, match="UDT_BASELINE_DT_ROOT"):
        _import_baseline_dt()


def test_no_sys_path_pollution(tmp_path):
    """_import_baseline_dt must not permanently pollute sys.path."""
    real_baseline = (Path(__file__).resolve().parents[1] / "baselines" / "decision_transformer" / "gym").resolve()
    real_str = str(real_baseline)

    # Ensure it is not in sys.path before call
    if real_str in sys.path:
        sys.path.remove(real_str)

    _import_baseline_dt(baseline_root=real_baseline)
    assert real_str not in sys.path, f"{real_str} was left in sys.path!"


def test_no_sys_path_pollution_on_failure(tmp_path):
    """Failed import must clean up sys.path in finally block."""
    fake_baseline = tmp_path / "empty_dir"
    fake_baseline.mkdir()
    fake_str = str(fake_baseline.resolve())

    if fake_str in sys.path:
        sys.path.remove(fake_str)

    with pytest.raises(RealDTUnavailable):
        _import_baseline_dt(baseline_root=fake_baseline)

    assert fake_str not in sys.path, f"{fake_str} was left in sys.path after failure!"


def test_load_decision_transformer_accepts_baseline_root(tmp_path):
    """load_decision_transformer passes baseline_root through to _import_baseline_dt."""
    ckpt = tmp_path / "empty.pt"
    import torch
    torch.save({"model_state_dict": {}}, ckpt)

    with pytest.raises(RealDTUnavailable, match="not found"):
        load_decision_transformer(
            ckpt,
            state_dim=2,
            act_dim=1,
            hidden_size=8,
            baseline_root=tmp_path / "invalid_root",
        )


def test_train_decision_transformer_accepts_baseline_root(tmp_path):
    """train_decision_transformer passes baseline_root through to _import_baseline_dt."""
    import numpy as np
    arrays = {
        "observations": np.ones((4, 2), dtype=np.float32),
        "actions": np.zeros((4, 1), dtype=np.float32),
        "rewards": np.ones(4, dtype=np.float32),
        "returns_to_go": np.ones(4, dtype=np.float32),
        "timesteps": np.arange(4, dtype=np.int64),
        "episode_ids": np.zeros(4, dtype=np.int64),
        "is_validation": np.zeros(4, dtype=bool),
    }
    manifest = {
        "state_mean": [0.0, 0.0],
        "state_std": [1.0, 1.0],
        "horizon": 4,
        "reward_scale": 100.0,
    }

    with pytest.raises(RealDTUnavailable, match="not found"):
        train_decision_transformer(
            arrays,
            manifest,
            allow_no_validation=True,
            baseline_root=tmp_path / "invalid_root",
        )
