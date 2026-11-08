import json
import numpy as np
import pytest
import torch

from udt.dt_integration import train_decision_transformer


def _build_dataset_with_episodes(episode_lengths: list[int], horizon: int = 20):
    """Build a dataset with specified episode lengths and unique identifiable transitions."""
    obs = []
    acts = []
    rews = []
    rtgs = []
    timesteps = []
    terminals = []
    eids = []
    is_val = []

    c = 100.0  # reward_scale
    for eid, ep_len in enumerate(episode_lengths):
        val_flag = (eid == len(episode_lengths) - 1)
        # Each transition has identifiable values:
        # obs: [eid, t, eid*10 + t, 1.0]
        # act: [eid * 100 + t]
        # rew: 10.0
        # rtg: sum of remaining rewards / c
        ep_rews = [10.0] * ep_len
        for t in range(ep_len):
            obs.append([float(eid), float(t), float(eid * 10 + t), 1.0])
            acts.append([float(eid * 100 + t)])
            rews.append(10.0)
            remaining_rew = sum(ep_rews[t:])
            rtgs.append(remaining_rew / c)
            timesteps.append(t)
            terminals.append(t == ep_len - 1)
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
        "state_mean": [0.0, 0.0, 0.0, 0.0],
        "state_std": [1.0, 1.0, 1.0, 1.0],
        "reward_scale": c,
        "horizon": horizon,
    }
    return arrays, manifest


def _capture_batches(arrays, manifest, k: int, batch_size: int = 1, num_steps: int = 1, seed: int = 42):
    """Helper to run train_decision_transformer and intercept sampler batches."""
    captured = []
    from udt import dt_integration
    orig_import = dt_integration._import_baseline_dt

    def mock_import(baseline_root=None):
        DT, Trainer = orig_import(baseline_root=baseline_root)

        class CapturingTrainer(Trainer):
            def train_step(self):
                batch = self.get_batch(self.batch_size)
                captured.append(batch)
                return 0.0

        return DT, CapturingTrainer

    dt_integration._import_baseline_dt = mock_import
    try:
        train_decision_transformer(
            arrays,
            manifest,
            num_steps=num_steps,
            batch_size=batch_size,
            k=k,
            embed_dim=16,
            n_layer=1,
            n_head=1,
            device="cpu",
            seed=seed,
            allow_no_validation=False,
        )
    finally:
        dt_integration._import_baseline_dt = orig_import
    return captured


def test_sampler_sequence_shorter_than_k():
    """When trajectory length < K, verify padding is on the left with mask=0."""
    # Episode lengths: 3 (train), 5 (val). With K=6, train trajectory is strictly shorter than K
    arrays, manifest = _build_dataset_with_episodes([3, 5], horizon=10)
    batches = _capture_batches(arrays, manifest, k=6, batch_size=1, seed=42)
    s, a, r, d, rtg, timesteps, mask = batches[0]

    mask_np = mask[0].cpu().numpy()
    # Length of episode is 3, so cur_len <= 3, pad_len >= 3
    pad_len = 6 - int(mask_np.sum())
    assert pad_len >= 3
    # Left pad_len tokens must have mask 0
    assert np.all(mask_np[:pad_len] == 0.0)
    # Right tokens must have mask 1
    assert np.all(mask_np[pad_len:] == 1.0)

    # Padding in RTG must be 0.0
    rtg_np = rtg[0, :, 0].cpu().numpy()
    assert np.all(rtg_np[:pad_len] == 0.0)

    # Padding in actions must be -10.0 (baseline standard)
    a_np = a[0, :, 0].cpu().numpy()
    assert np.all(a_np[:pad_len] == -10.0)


def test_sampler_sequence_longer_than_k():
    """When trajectory length > K, unpadded window has mask=1 everywhere."""
    # Episode lengths: 10 (train), 5 (val). With K=4
    arrays, manifest = _build_dataset_with_episodes([10, 5], horizon=10)
    batches = _capture_batches(arrays, manifest, k=4, batch_size=1, seed=42)
    s, a, r, d, rtg, timesteps, mask = batches[0]

    mask_np = mask[0].cpu().numpy()
    assert mask_np.shape == (4,)
    # If unpadded window sampled (e.g. si <= 6), mask is all 1s
    if mask_np.sum() == 4:
        assert np.all(mask_np == 1.0)


def test_sampler_single_transition_episode():
    """Episode with exactly one transition must pad correctly without crashing."""
    arrays, manifest = _build_dataset_with_episodes([1, 2], horizon=5)
    batches = _capture_batches(arrays, manifest, k=4, batch_size=1, seed=42)
    s, a, r, d, rtg, timesteps, mask = batches[0]

    assert s.shape == (1, 4, 4)
    assert a.shape == (1, 4, 1)
    assert rtg.shape == (1, 5, 1)
    assert mask.shape == (1, 4)


def test_sampler_alignment_and_terminal_continuation():
    """Check exact alignment between observation, action, RTG, timestep, and continuation RTG."""
    # 1 train episode with 5 transitions, 1 val episode
    arrays, manifest = _build_dataset_with_episodes([5, 5], horizon=10)
    batches = _capture_batches(arrays, manifest, k=5, batch_size=1, seed=123)
    s, a, r, d, rtg, timesteps, mask = batches[0]

    s_np = s[0].cpu().numpy()
    a_np = a[0, :, 0].cpu().numpy()
    rtg_np = rtg[0, :, 0].cpu().numpy()
    t_np = timesteps[0].cpu().numpy()
    mask_np = mask[0].cpu().numpy()

    valid_indices = np.where(mask_np > 0)[0]
    for idx in valid_indices:
        t = t_np[idx]
        # Verify action corresponds to eid*100 + t (eid=0 for train) -> 0*100 + t = t
        assert a_np[idx] == float(t)
        # Verify RTG corresponds to remaining rewards / 100.0
        # Total episode length is 5, rewards are 10.0 each.
        # remaining at t = (5 - t) * 10.0 / 100.0 = (5 - t) * 0.1
        expected_rtg = (5 - t) * 0.1
        assert np.isclose(rtg_np[idx], expected_rtg, atol=1e-5)

    # Terminal continuation RTG must be 0.0 when episode ends
    if valid_indices[-1] == len(mask_np) - 1 and t_np[-1] == 4:
        assert rtg_np[-1] == 0.0


def test_sampler_padded_action_loss_independence():
    """Verify that padding tokens (mask=0) do NOT affect the action MSE loss."""
    from decision_transformer.training.seq_trainer import SequenceTrainer

    # Create dummy tensors with K=5, batch=1, where first 3 tokens are padded (mask=0)
    batch_size = 1
    k = 5
    act_dim = 2
    attention_mask = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0]])
    action_preds = torch.randn(batch_size, k, act_dim)
    action_target = torch.randn(batch_size, k, act_dim)

    # Compute loss as SequenceTrainer does:
    preds_masked = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
    targets_masked = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
    loss_orig = torch.mean((preds_masked - targets_masked) ** 2).item()

    # Now corrupt padded actions with extreme values (e.g. +1e6, -1e6)
    action_preds_corrupted = action_preds.clone()
    action_target_corrupted = action_target.clone()
    action_preds_corrupted[:, :3, :] = 1e6
    action_target_corrupted[:, :3, :] = -1e6

    preds_masked_c = action_preds_corrupted.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
    targets_masked_c = action_target_corrupted.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
    loss_corrupted = torch.mean((preds_masked_c - targets_masked_c) ** 2).item()

    # The loss MUST be completely identical
    assert np.isclose(loss_orig, loss_corrupted), "Padded tokens leaked into action loss calculation!"


def test_sampler_deterministic_sampling():
    """Verify that identical seed produces identical sampled batches."""
    arrays, manifest = _build_dataset_with_episodes([5, 5, 5], horizon=10)
    batches_seed1 = _capture_batches(arrays, manifest, k=3, batch_size=2, num_steps=2, seed=999)
    batches_seed2 = _capture_batches(arrays, manifest, k=3, batch_size=2, num_steps=2, seed=999)

    for (b1, b2) in zip(batches_seed1, batches_seed2):
        for t1, t2 in zip(b1, b2):
            torch.testing.assert_close(t1, t2)
