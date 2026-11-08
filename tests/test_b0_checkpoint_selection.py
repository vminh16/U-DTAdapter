import json
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from udt.data import prepare_dataset
from udt.dt_integration import (
    assert_parameters_frozen,
    load_decision_transformer,
    train_decision_transformer,
)


def _create_synthetic_dataset():
    """Create a minimal 4-episode dataset (3 train, 1 val)."""
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
            rtgs.append((5.0 - t) / 10.0)  # G_t / c
            timesteps.append(t)
            terminals.append(t == 4)
            eids.append(eid)
            is_val.append(val_flag)

    arrays = {
        "observations": np.array(obs, dtype=np.float32),
        "actions": np.array(acts, dtype=np.float32),
        "rewards": np.array(rews, dtype=np.float32),
        "returns_to_go": np.array(rtgs, dtype=np.float32),
        "timesteps": np.array(timesteps, dtype=np.int64),
        "terminals": np.array(terminals, dtype=bool),
        "episode_ids": np.array(eids, dtype=np.int64),
        "is_validation": np.array(is_val, dtype=bool),
    }
    manifest = {
        "state_mean": [1.0, 2.0],
        "state_std": [1.0, 1.0],
        "reward_scale": 10.0,
        "horizon": 10,
    }
    return arrays, manifest


def test_best_validation_checkpoint_restored(tmp_path, monkeypatch):
    """Verify that model weights are restored from best validation step, not last step."""
    arrays, manifest = _create_synthetic_dataset()
    output_ckpt = tmp_path / "model.pt"

    from udt import dt_integration
    orig_import = dt_integration._import_baseline_dt

    step_counter = [0]
    weights_at_step_1 = {}
    weights_at_step_2 = {}

    def mock_import(baseline_root=None):
        DT, Trainer = orig_import(baseline_root=baseline_root)

        class MockTrainer(Trainer):
            def train_step(self):
                step_counter[0] += 1
                # Perturb a parameter to ensure weights change each step
                with torch.no_grad():
                    first_param = next(self.model.parameters())
                    first_param.add_(float(step_counter[0]))
                    if step_counter[0] == 1:
                        weights_at_step_1["val"] = first_param.detach().cpu().clone()
                    elif step_counter[0] == 2:
                        weights_at_step_2["val"] = first_param.detach().cpu().clone()
                return 1.0

        return DT, MockTrainer

    monkeypatch.setattr(dt_integration, "_import_baseline_dt", mock_import)

    # Monkeypatch validation loss to be LOW at step 1 (0.05) and HIGH at step 2 (0.95)
    orig_eval = None
    # We run 2 steps, evaluating at each step: eval_interval=1
    # Step 1 should be chosen as best_step!

    # We patch _evaluate_validation inside dt_integration if possible, or monkeypatch forward
    call_count = [0]
    orig_forward = dt_integration.DecisionTransformer.forward if hasattr(dt_integration, "DecisionTransformer") else None

    # Let's monkeypatch SequenceTrainer to control validation evaluation directly
    # In train_decision_transformer, _evaluate_validation evaluates val_sampler with model.forward
    # Let's patch model.forward so validation MSE is 0.05 at step 1 and 0.95 at step 2
    def mock_import_with_val(baseline_root=None):
        DT, Trainer = orig_import(baseline_root=baseline_root)

        class CustomDT(DT):
            def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
                s_pred, a_pred, r_pred = super().forward(
                    states, actions, rewards, returns_to_go, timesteps, attention_mask=attention_mask
                )
                if not self.training:
                    # In validation mode, offset action predictions based on step
                    offset = 0.05 if step_counter[0] == 1 else 5.0
                    a_pred = actions + offset
                return s_pred, a_pred, r_pred

        class MockTrainer(Trainer):
            def train_step(self):
                step_counter[0] += 1
                with torch.no_grad():
                    first_param = next(self.model.parameters())
                    first_param.fill_(float(step_counter[0]) * 100.0)
                    if step_counter[0] == 1:
                        weights_at_step_1["val"] = first_param.detach().cpu().clone()
                    elif step_counter[0] == 2:
                        weights_at_step_2["val"] = first_param.detach().cpu().clone()
                return 1.0

        return CustomDT, MockTrainer

    monkeypatch.setattr(dt_integration, "_import_baseline_dt", mock_import_with_val)

    model, summary = train_decision_transformer(
        arrays,
        manifest,
        output_path=output_ckpt,
        num_steps=2,
        batch_size=2,
        k=3,
        embed_dim=16,
        n_layer=1,
        n_head=1,
        eval_interval=1,
        eval_batches=1,
        device="cpu",
        seed=42,
    )

    # Best step must be 1, NOT 2
    assert summary["best_step"] == 1, f"Expected best_step=1, got {summary.get('best_step')}"
    assert summary["best_val_loss"] < summary["metrics"]["val_loss"][1]

    # The returned model and saved checkpoint must have weights from step 1, NOT step 2
    first_param_returned = next(model.parameters())
    torch.testing.assert_close(first_param_returned, weights_at_step_1["val"])

    # Load saved checkpoint
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
    first_param_loaded = next(loaded_model.parameters())
    torch.testing.assert_close(first_param_loaded, weights_at_step_1["val"])
    assert payload["best_step"] == 1
    assert payload["best_val_loss"] == summary["best_val_loss"]

    # Must be frozen and in eval mode
    assert not loaded_model.training
    assert_parameters_frozen(loaded_model)


def test_no_validation_produces_no_fake_best_loss(tmp_path):
    """When eval_interval is None, best_val_loss must be None, not 0.0 or inf."""
    arrays, manifest = _create_synthetic_dataset()
    output_ckpt = tmp_path / "model_no_val.pt"

    model, summary = train_decision_transformer(
        arrays,
        manifest,
        output_path=output_ckpt,
        num_steps=2,
        batch_size=2,
        k=3,
        embed_dim=16,
        n_layer=1,
        n_head=1,
        eval_interval=None,
        device="cpu",
        seed=42,
    )

    assert summary["best_val_loss"] is None
    assert summary["best_step"] is None

    payload = torch.load(output_ckpt, weights_only=True)
    assert payload["best_val_loss"] is None
    assert payload["best_step"] is None
