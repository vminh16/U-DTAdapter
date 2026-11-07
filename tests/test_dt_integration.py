"""Integration-interface tests for frozen DT parity and closed-loop variants."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from udt.dt_adapter import DTAdapter
from udt.dt_integration import (
    RealDTUnavailable,
    assert_parameters_frozen,
    load_decision_transformer,
    train_decision_transformer,
    verify_b0_parity,
    verify_n1_matches_dt_action,
)
from udt.dynamics import DynamicsEnsemble, ModuleUController
from udt.experiment import run_variant
from udt.return_cap import GlobalReturnCap, ReturnCapMLP, ReturnCapper


class ReferenceDT(nn.Module):
    """Deterministic DT-shaped model used only for interface parity tests."""

    def __init__(self, state_dim=2, act_dim=1, hidden_dim=8):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.state_proj = nn.Linear(state_dim, act_dim, bias=False)
        self.context_proj = nn.Linear(act_dim, act_dim, bias=False)

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        latest = self.state_proj(states[:, -1])
        context = self.context_proj(actions[:, :-1].mean(dim=1)) if actions.shape[1] > 1 else 0.0
        rtg = returns_to_go[:, -1] * 0.001
        return (latest + context + rtg)[0]


class TinyEnv:
    def __init__(self, max_steps=3):
        self.max_steps = max_steps
        self.step_count = 0
        self.action_space = type("ActionSpace", (), {"shape": (1,)})()
        self.reset_seeds = []

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.step_count = 0
        return np.ones(2, dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        return np.ones(2, dtype=np.float32), 1.0, self.step_count >= self.max_steps, {}


def test_b0_parity_against_direct_reference_dt():
    torch.manual_seed(0)
    model = ReferenceDT()
    adapter = DTAdapter(
        model, state_dim=2, act_dim=1, max_length=4,
        state_mean=np.array([0.5, -0.5], dtype=np.float32),
        state_std=np.array([2.0, 3.0], dtype=np.float32),
    )
    observations = [np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.array([3.0, -1.0])]
    rtgs = [5.0, 4.0, 3.0]
    report = verify_b0_parity(model, adapter, observations=observations, rtg_tokens=rtgs)
    assert report["passed"], report
    assert report["steps"] == 3


def test_b0_parity_applies_adapter_action_bounds():
    model = ReferenceDT()
    adapter = DTAdapter(
        model,
        state_dim=2,
        act_dim=1,
        max_length=4,
        action_low=np.array([-0.01], dtype=np.float32),
        action_high=np.array([0.01], dtype=np.float32),
    )
    report = verify_b0_parity(
        model,
        adapter,
        observations=[np.array([1.0, 2.0])],
        rtg_tokens=[5.0],
    )
    assert report["passed"], report


def test_n1_returns_dt_action_for_active_r_branch():
    model = ReferenceDT()
    adapter = DTAdapter(model, state_dim=2, act_dim=1)
    ensemble = DynamicsEnsemble(
        state_dim=2, act_dim=1, num_models=1, hidden_dim=8,
        state_mean=np.zeros(2, dtype=np.float32),
        state_std=np.ones(2, dtype=np.float32),
        horizon=10, reward_scale=1000.0,
    )
    return_model = ReturnCapper(GlobalReturnCap(1.0), horizon=10, scale=1000.0)
    controller = ModuleUController(
        dt_adapter=adapter, ensemble=ensemble, return_model=return_model,
        num_candidates=1, horizon=10,
        action_low=np.array([-10.0]), action_high=np.array([10.0]),
    )
    report = verify_n1_matches_dt_action(
        adapter, controller, obs=np.array([1.0, 2.0]), rtg_token=2.5, step=3,
    )
    assert report["passed"], report


def test_frozen_parameter_check():
    model = ReferenceDT()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    assert_parameters_frozen(model)
    model.train()
    with pytest.raises(AssertionError, match="eval mode"):
        assert_parameters_frozen(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = True
    with pytest.raises(AssertionError, match="not frozen"):
        assert_parameters_frozen(model)


def test_real_dt_loader_fails_loudly_without_optional_dependency(tmp_path):
    checkpoint = tmp_path / "dt.pt"
    torch.save({"model_state_dict": {}}, checkpoint)
    with pytest.raises((RealDTUnavailable, RuntimeError)):
        load_decision_transformer(
            checkpoint, state_dim=2, act_dim=1, hidden_size=8, n_layer=1, n_head=1,
        )


def _build_u_components(seed=0):
    torch.manual_seed(seed)
    model = ReferenceDT()
    adapter = DTAdapter(model, state_dim=2, act_dim=1, max_length=4)
    ensemble = DynamicsEnsemble(
        state_dim=2, act_dim=1, num_models=2, hidden_dim=8,
        state_mean=np.zeros(2, dtype=np.float32),
        state_std=np.ones(2, dtype=np.float32),
        horizon=3, reward_scale=1000.0,
    )
    for m in ensemble.models:
        for p in m.parameters():
            p.data.zero_()
    mlp = ReturnCapMLP(state_dim=2, hidden_dim=8, num_layers=2, mode="state_time")
    for p in mlp.parameters():
        p.data.zero_()
    list(mlp.parameters())[-1].data.fill_(0.5)
    return_model = ReturnCapper(mlp, horizon=3, scale=1000.0)
    capper = ReturnCapper(GlobalReturnCap(0.5), horizon=3, scale=1000.0)
    return adapter, ensemble, return_model, capper


def test_closed_loop_variants_expose_required_diagnostics():
    for variant in ("B0", "B3", "B4", "B5"):
        adapter, ensemble, return_model, capper = _build_u_components()
        run = run_variant(
            variant,
            TinyEnv(max_steps=3),
            adapter,
            capper=capper,
            ensemble=ensemble,
            return_model=return_model,
            num_candidates=4,
            horizon=3,
            initial_target=2000.0,
            reward_scale=1000.0,
            seed=42,
            action_low=np.array([-1.0]),
            action_high=np.array([1.0]),
        )
        diag = run.diagnostics()
        assert diag["variant"] == variant
        assert len(diag["raw_rtg"]) == 3
        assert len(diag["token_rtg"]) == 3
        assert len(diag["selected_actions"]) == 3
        assert diag["seed_mapping"]["streams"]
        if variant in ("B4", "B5"):
            assert len(diag["original_actions"]) == 3
            assert len(diag["disagreements"]) == 3


def test_r_only_and_u_only_branches_use_expected_components():
    adapter, ensemble, return_model, capper = _build_u_components()
    b3 = run_variant(
        "B3", TinyEnv(max_steps=2), adapter, capper=capper, horizon=3,
        initial_target=2000.0, reward_scale=1000.0, seed=1,
        action_low=np.array([-1.0]), action_high=np.array([1.0]),
    )
    assert b3.rollout.clipping_frequency > 0

    adapter, ensemble, return_model, capper = _build_u_components()
    b4 = run_variant(
        "B4", TinyEnv(max_steps=2), adapter, ensemble=ensemble,
        return_model=return_model, horizon=3, initial_target=2000.0,
        reward_scale=1000.0, seed=1, num_candidates=3,
        action_low=np.array([-1.0]), action_high=np.array([1.0]),
    )
    assert b4.rollout.clipping_frequency == 0.0
    assert b4.rollout.caps == [float("inf"), float("inf")]


def test_train_decision_transformer_uses_baseline_sequence_trainer(tmp_path):
    import json

    # Create synthetic trajectory dataset
    # 4 trajectories (eids 0, 1, 2 are train, 3 is validation)
    obs = []
    acts = []
    rews = []
    rtgs = []
    timesteps = []
    terminals = []
    eids = []
    is_val = []

    for eid in range(4):
        val_flag = (eid == 3)
        for t in range(5):
            obs.append([float(eid + 1), float(t)])
            acts.append([float(eid * 0.1 + t * 0.01)])
            rews.append(1.0)
            rtgs.append(5.0 - t)
            timesteps.append(t)
            terminals.append(t == 4)
            eids.append(eid)
            is_val.append(val_flag)

    dataset_path = tmp_path / "dataset.npz"
    np.savez(
        dataset_path,
        observations=np.array(obs, dtype=np.float32),
        actions=np.array(acts, dtype=np.float32),
        rewards=np.array(rews, dtype=np.float32),
        returns_to_go=np.array(rtgs, dtype=np.float32),
        timesteps=np.array(timesteps, dtype=np.int64),
        terminals=np.array(terminals, dtype=bool),
        episode_ids=np.array(eids, dtype=np.int64),
        is_validation=np.array(is_val, dtype=bool),
    )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "state_mean": [1.0, 2.0],
        "state_std": [1.0, 1.0],
        "reward_scale": 1000.0,
        "horizon": 10,
    }))

    output_ckpt = tmp_path / "model.pt"

    model, summary = train_decision_transformer(
        dataset_path,
        manifest_path,
        output_path=output_ckpt,
        num_steps=5,
        batch_size=2,
        k=3,
        embed_dim=16,
        n_layer=1,
        n_head=1,
        eval_interval=5,
        eval_batches=2,
        device="cpu",
        seed=42,
    )

    assert output_ckpt.exists()
    assert summary["num_train_trajectories"] == 3
    assert summary["num_val_trajectories"] == 1
    assert len(summary["metrics"]["train_loss"]) == 5
    assert "val_loss" in summary["metrics"]

    # Load back the saved checkpoint and verify parameter freezing & parity
    loaded_model, payload = load_decision_transformer(
        output_ckpt,
        state_dim=2,
        act_dim=1,
        hidden_size=16,
        max_length=3,
        max_ep_len=10,
        n_layer=1,
        n_head=1,
    )
    assert_parameters_frozen(loaded_model)

    adapter = DTAdapter(
        loaded_model,
        state_dim=2,
        act_dim=1,
        max_length=3,
        state_mean=np.array([1.0, 2.0], dtype=np.float32),
        state_std=np.array([1.0, 1.0], dtype=np.float32),
    )
    report = verify_b0_parity(
        loaded_model,
        adapter,
        observations=[np.array([1.0, 2.0], dtype=np.float32)],
        rtg_tokens=[5.0],
        model_source="real",
    )
    assert report["passed"] is True


def test_train_decision_transformer_in_memory_and_no_output_path():
    # Pass in-memory dicts with no validation split
    arrays = {
        "observations": np.ones((10, 2), dtype=np.float32),
        "actions": np.zeros((10, 1), dtype=np.float32),
        "rewards": np.ones(10, dtype=np.float32),
        "returns_to_go": np.ones(10, dtype=np.float32),
        "timesteps": np.arange(10, dtype=np.int64),
        "episode_ids": np.zeros(10, dtype=np.int64),
    }
    manifest = {
        "state_mean": [0.0, 0.0],
        "state_std": [1.0, 1.0],
        "horizon": 10,
        "reward_scale": 100.0,
    }

    model, summary = train_decision_transformer(
        arrays,
        manifest,
        output_path=None,
        num_steps=3,
        batch_size=2,
        k=2,
        embed_dim=16,
        n_layer=1,
        n_head=1,
        eval_interval=None,
        device="cpu",
    )

    assert isinstance(model, nn.Module)
    assert len(summary["metrics"]["train_loss"]) == 3
    assert summary["num_train_trajectories"] == 1
    assert summary["num_val_trajectories"] == 0
    assert_parameters_frozen(model)


