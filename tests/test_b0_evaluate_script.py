import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate import main as evaluate_main
from udt.dt_integration import train_decision_transformer


def _train_dummy_checkpoint(tmp_path, state_dim=4, act_dim=2, horizon=20, reward_scale=100.0):
    obs = np.ones((10, state_dim), dtype=np.float32)
    acts = np.zeros((10, act_dim), dtype=np.float32)
    rews = np.ones(10, dtype=np.float32)
    rtgs = np.ones(10, dtype=np.float32)
    timesteps = np.tile(np.arange(5), 2)
    terminals = np.array([False, False, False, False, True, False, False, False, False, True])
    eids = np.array([0] * 5 + [1] * 5, dtype=np.int64)
    is_val = np.array([False] * 5 + [True] * 5, dtype=bool)

    arrays = {
        "observations": obs,
        "actions": acts,
        "rewards": rews,
        "returns_to_go": rtgs,
        "timesteps": timesteps,
        "terminals": terminals,
        "episode_ids": eids,
        "is_validation": is_val,
    }
    manifest = {
        "state_mean": [0.0] * state_dim,
        "state_std": [1.0] * state_dim,
        "reward_scale": reward_scale,
        "horizon": horizon,
    }
    ckpt_path = tmp_path / "model.pt"
    train_decision_transformer(
        arrays,
        manifest,
        output_path=ckpt_path,
        num_steps=1,
        batch_size=2,
        k=3,
        embed_dim=16,
        n_layer=1,
        n_head=1,
        eval_interval=1,
        eval_batches=1,
        device="cpu",
    )
    return ckpt_path


def _argv(ckpt_path, tmp_path, **overrides):
    values = {
        "--variant": "B0",
        "--checkpoint": str(ckpt_path),
        "--device": "cpu",
        "--parity-output": str(tmp_path / "parity.json"),
        "--results-output": str(tmp_path / "results.json"),
    }
    values.update(overrides)
    args = ["evaluate.py"]
    for key, value in values.items():
        if value != "":
            args.extend([key, str(value)])
        else:
            args.append(key)
    return args


def test_evaluate_rejects_invalid_episodes(tmp_path, monkeypatch):
    ckpt_path = _train_dummy_checkpoint(tmp_path)
    monkeypatch.setattr("sys.argv", _argv(ckpt_path, tmp_path, **{"--episodes": 0}))
    with pytest.raises(ValueError, match="episodes.*positive"):
        evaluate_main()


def test_evaluate_rejects_invalid_horizon(tmp_path, monkeypatch):
    ckpt_path = _train_dummy_checkpoint(tmp_path)
    monkeypatch.setattr("sys.argv", _argv(ckpt_path, tmp_path, **{"--horizon": 0}))
    with pytest.raises(ValueError, match="horizon.*positive"):
        evaluate_main()


def test_evaluate_rejects_horizon_exceeding_checkpoint_max_ep_len(tmp_path, monkeypatch):
    ckpt_path = _train_dummy_checkpoint(tmp_path, horizon=15)
    monkeypatch.setattr("sys.argv", _argv(ckpt_path, tmp_path, **{"--horizon": 20}))
    with pytest.raises(ValueError, match="horizon.*max_ep_len"):
        evaluate_main()


def test_evaluate_rejects_horizon_exceeding_n_positions(tmp_path, monkeypatch):
    ckpt_path = _train_dummy_checkpoint(tmp_path, horizon=2000)
    monkeypatch.setattr("sys.argv", _argv(ckpt_path, tmp_path, **{"--horizon": 1025}))
    with pytest.raises(ValueError, match="capacity|positional embedding|1024"):
        evaluate_main()


def test_evaluate_rejects_missing_metadata(tmp_path, monkeypatch):
    bad_ckpt = tmp_path / "bad_model.pt"
    torch.save({"model_state_dict": {}}, bad_ckpt)
    monkeypatch.setattr("sys.argv", _argv(bad_ckpt, tmp_path))
    with pytest.raises(ValueError, match="missing required metadata|state_dim"):
        evaluate_main()


class DummyEnv:
    def __init__(self, state_dim=4, act_dim=2):
        self.observation_space = type("ObsSpace", (), {"shape": (state_dim,)})()
        self.action_space = type(
            "ActSpace",
            (),
            {
                "shape": (act_dim,),
                "low": -np.ones(act_dim, dtype=np.float32),
                "high": np.ones(act_dim, dtype=np.float32),
            },
        )()

    def reset(self, seed=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 1.0, True, False, {}

    def close(self):
        pass


def test_evaluate_rejects_env_dimension_mismatch(tmp_path, monkeypatch):
    ckpt_path = _train_dummy_checkpoint(tmp_path, state_dim=4, act_dim=2)
    monkeypatch.setattr(
        "sys.argv", _argv(ckpt_path, tmp_path, **{"--env": "DummyEnv", "--horizon": 20})
    )
    import gymnasium as gym

    monkeypatch.setattr(gym, "make", lambda env_id: DummyEnv(state_dim=11, act_dim=3))
    with pytest.raises(ValueError, match="dimension mismatch|state_dim|observation shape"):
        evaluate_main()


def test_evaluate_safe_deserialization_rejects_arbitrary_objects(tmp_path, monkeypatch):
    import os

    unsafe_ckpt = tmp_path / "unsafe.pt"
    torch.save({"model_state_dict": {}, "state_dim": 4, "act_dim": 2, "func": os.system}, unsafe_ckpt)
    monkeypatch.setattr("sys.argv", _argv(unsafe_ckpt, tmp_path))
    with pytest.raises(Exception):
        evaluate_main()


def test_evaluate_allow_unsafe_legacy_flag(tmp_path, monkeypatch):
    import os

    unsafe_ckpt = tmp_path / "unsafe.pt"
    torch.save({"model_state_dict": {}, "func": os.system}, unsafe_ckpt)
    monkeypatch.setattr(
        "sys.argv", _argv(unsafe_ckpt, tmp_path, **{"--allow-unsafe-legacy": ""})
    )
    with pytest.raises(ValueError, match="missing required metadata"):
        evaluate_main()


def test_evaluate_successful_run_writes_provenance_and_telemetry(tmp_path, monkeypatch):
    ckpt_path = _train_dummy_checkpoint(tmp_path, state_dim=4, act_dim=2, horizon=20)
    out_results = tmp_path / "results.json"
    out_parity = tmp_path / "parity.json"

    import gymnasium as gym

    monkeypatch.setattr(gym, "make", lambda env_id: DummyEnv())
    monkeypatch.setattr(
        "sys.argv",
        _argv(
            ckpt_path,
            tmp_path,
            **{"--episodes": 2, "--horizon": 10},
        ),
    )

    assert evaluate_main() == 0
    assert out_results.exists()
    assert out_parity.exists()

    data = json.loads(out_results.read_text())
    assert data["variant"] == "B0"
    assert len(data["provenance"]["checkpoint_sha256"]["decision_transformer"]) == 64
    assert data["provenance"]["git_commit"]
    assert data["environment_metadata"]["d4rl_closed_loop_parity"] == "BLOCKED"
    assert len(data["episodes_detail"]) == 2
    for episode in data["episodes_detail"]:
        for key in ("seed", "total_reward", "length", "final_raw_budget"):
            assert key in episode
    assert data["parity"]["passed"] is True
