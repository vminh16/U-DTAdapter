"""Frozen Decision Transformer integration and parity checks.

The baseline Decision Transformer code lives under
``baselines/decision_transformer`` and depends on the optional ``transformers``
package. When that dependency or a checkpoint is unavailable this module
raises an explicit error instead of silently falling back to a mock model.
"""

import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from udt.dt_adapter import DTAdapter


class RealDTUnavailable(RuntimeError):
    """Raised when real DT inference cannot be constructed."""


def _resolve_baseline_root(baseline_root: str | Path | None = None) -> Path | None:
    """Resolve the directory containing the baseline gym DecisionTransformer codebase."""
    if baseline_root is not None:
        p = Path(baseline_root).resolve()
        if not p.exists():
            raise RealDTUnavailable(f"Specified baseline DT root not found at: {p}")
        return p

    env_root = os.environ.get("UDT_BASELINE_DT_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if not p.exists():
            raise RealDTUnavailable(
                f"UDT_BASELINE_DT_ROOT was set to '{env_root}', but directory does not exist."
            )
        return p

    curr = Path(__file__).resolve().parent
    for _ in range(5):
        candidate = curr / "baselines" / "decision_transformer" / "gym"
        if candidate.exists():
            return candidate.resolve()
        if curr.parent == curr:
            break
        curr = curr.parent

    cwd_candidate = Path.cwd() / "baselines" / "decision_transformer" / "gym"
    if cwd_candidate.exists():
        return cwd_candidate.resolve()

    return None


def _import_baseline_dt(baseline_root: str | Path | None = None) -> tuple[Any, Any]:
    """Import DecisionTransformer model and SequenceTrainer from baseline gym codebase."""
    resolved_root = _resolve_baseline_root(baseline_root)
    path_added = False
    path_str = None

    if resolved_root is not None:
        path_str = str(resolved_root)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            path_added = True

    try:
        if resolved_root is not None and not (resolved_root / "decision_transformer").exists():
            raise RealDTUnavailable(
                f"Baseline DT directory {resolved_root} does not contain 'decision_transformer' package."
            )

        try:
            import transformers.file_utils as _fu

            if hasattr(_fu, "add_code_sample_docstrings"):
                _orig_add = _fu.add_code_sample_docstrings

                def _compat_add(*args: Any, **kwargs: Any) -> Any:
                    kwargs.pop("tokenizer_class", None)
                    return _orig_add(*args, **kwargs)

                _fu.add_code_sample_docstrings = _compat_add

            from decision_transformer.models.decision_transformer import DecisionTransformer
            from decision_transformer.training.seq_trainer import SequenceTrainer

            return DecisionTransformer, SequenceTrainer
        except ImportError as exc:
            if resolved_root is None:
                raise RealDTUnavailable(
                    "Baseline Decision Transformer codebase not found. "
                    "Specify the path using the UDT_BASELINE_DT_ROOT environment variable "
                    "or pass baseline_root=<path>."
                ) from exc
            raise RealDTUnavailable(
                "Real Decision Transformer training/inference requires the optional baseline "
                "dependencies (notably `transformers`). Install them and retry; "
                f"import failed with: {exc}"
            ) from exc
        except Exception as exc:  # pragma: no cover - depends on optional deps
            raise RealDTUnavailable(
                "Real Decision Transformer training/inference requires the optional baseline "
                "dependencies (notably `transformers`). Install them and retry; "
                f"import failed with: {exc}"
            ) from exc
    finally:
        if path_added and path_str in sys.path:
            sys.path.remove(path_str)


def load_frozen_dt_adapter(
    model: nn.Module,
    *,
    state_dim: int,
    act_dim: int,
    max_length: int = 20,
    state_mean: np.ndarray | None = None,
    state_std: np.ndarray | None = None,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
    device: str | torch.device = "cpu",
) -> DTAdapter:
    """Wrap an already-constructed (or loaded) DT model for inference."""
    return DTAdapter(
        model=model,
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=max_length,
        state_mean=state_mean,
        state_std=state_std,
        action_low=action_low,
        action_high=action_high,
        device=device,
    )


def load_decision_transformer(
    checkpoint_path: str | Path | dict[str, Any],
    *,
    state_dim: int,
    act_dim: int,
    hidden_size: int = 128,
    max_length: int = 20,
    max_ep_len: int = 4096,
    n_layer: int = 3,
    n_head: int = 1,
    device: str | torch.device = "cpu",
    weights_only: bool = True,
    allow_unsafe_legacy: bool = False,
    baseline_root: str | Path | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Construct the baseline DT architecture and load a checkpoint.

    Raises:
        RealDTUnavailable: If the baseline dependency tree (``transformers``)
            or the checkpoint cannot be loaded. This is intentionally loud.
    """
    if not isinstance(checkpoint_path, dict):
        if not weights_only and not allow_unsafe_legacy:
            raise ValueError(
                "Loading checkpoints with weights_only=False is unsafe and prohibited by default. "
                "Pass allow_unsafe_legacy=True explicitly to opt into loading trusted legacy checkpoints."
            )

        if allow_unsafe_legacy:
            import warnings
            warnings.warn(
                "Loading checkpoint with unsafe deserialization (weights_only=False). "
                "Only do this for trusted legacy files.",
                UserWarning,
                stacklevel=2,
            )
            actual_weights_only = False
        else:
            actual_weights_only = True
    else:
        actual_weights_only = True

    DecisionTransformer, _ = _import_baseline_dt(baseline_root=baseline_root)

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=hidden_size,
        max_length=max_length,
        max_ep_len=max_ep_len,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=4 * hidden_size,
        activation_function="relu",
        n_positions=1024,
        n_ctx=1024,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
    )
    if isinstance(checkpoint_path, dict):
        payload = checkpoint_path
    else:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=actual_weights_only)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict)
    model.to(torch.device(device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, payload if isinstance(payload, dict) else {"state_dict": payload}


def train_decision_transformer(
    dataset: str | Path | dict[str, np.ndarray],
    manifest: str | Path | dict[str, Any],
    *,
    output_path: str | Path | None = None,
    num_steps: int = 100,
    batch_size: int = 64,
    k: int = 20,
    embed_dim: int = 128,
    n_layer: int = 3,
    n_head: int = 1,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    warmup_steps: int = 50,
    eval_interval: int | None = None,
    eval_batches: int = 10,
    device: str | torch.device = "cpu",
    seed: int = 42,
    allow_no_validation: bool = False,
    baseline_root: str | Path | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train a Decision Transformer using the official baseline SequenceTrainer.

    Reuses ``SequenceTrainer`` from the baseline Decision Transformer codebase,
    adapting our offline trajectory dataset format (with strict train/validation
    separation via ``is_validation == False``) and state normalization.

    Args:
        dataset: Path to prepared ``dataset.npz`` or dictionary of arrays.
        manifest: Path to ``manifest.json`` or parsed dictionary.
        output_path: Optional destination to save frozen model checkpoint.
        num_steps: Total number of training steps (gradient updates).
        batch_size: Batch size of trajectory windows sampled per step.
        k: Context length K (history window length).
        embed_dim: Transformer hidden embedding dimension.
        n_layer: Number of transformer layers.
        n_head: Number of attention heads.
        learning_rate: Peak AdamW learning rate.
        weight_decay: Weight decay regularizer.
        warmup_steps: Number of linear warmup steps.
        eval_interval: Frequency of running validation evaluation (in steps).
        eval_batches: Number of batches sampled during validation evaluation.
        device: Torch compute device ("cpu", "cuda", etc.).
        seed: Random seed for reproducibility.

    Returns:
        A tuple of ``(model, summary)`` where model is in eval mode with frozen parameters.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dev = torch.device(device)

    # 1. Load dataset arrays and manifest
    if isinstance(dataset, (str, Path)):
        with np.load(dataset, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
    else:
        arrays = dataset

    if isinstance(manifest, (str, Path)):
        manifest_data = json.loads(Path(manifest).read_text())
    else:
        manifest_data = manifest

    state_mean = np.array(manifest_data["state_mean"], dtype=np.float32)
    state_std = np.array(manifest_data["state_std"], dtype=np.float32)
    state_std = np.maximum(state_std, 1e-6)
    reward_scale = float(manifest_data.get("reward_scale", 1000.0))
    if not np.isfinite(reward_scale) or reward_scale <= 0:
        raise ValueError(f"reward_scale in manifest must be finite and positive, got {reward_scale}")

    # Check dataset metadata if reward_scale is present in arrays
    if "reward_scale" in arrays:
        ds_scale = float(np.asarray(arrays["reward_scale"]).item())
        if not np.isclose(ds_scale, reward_scale):
            raise ValueError(
                f"Dataset reward_scale ({ds_scale}) mismatch with manifest reward_scale ({reward_scale})"
            )
    horizon = int(manifest_data.get("horizon", 1000))

    state_dim = state_mean.shape[0]
    act_dim = arrays["actions"].shape[1]

    # 2. Extract trajectory-level train and validation splits with strict leakage prevention
    if "is_validation" not in arrays:
        raise ValueError("Dataset is missing mandatory 'is_validation' split field.")
    is_val_arr = np.asarray(arrays["is_validation"])
    num_transitions = len(arrays["observations"])
    if is_val_arr.ndim != 1 or len(is_val_arr) != num_transitions:
        raise ValueError(
            f"is_validation shape {is_val_arr.shape} is malformed; expected ({num_transitions},)"
        )
    if not (np.issubdtype(is_val_arr.dtype, np.bool_) or np.isin(is_val_arr, [0, 1]).all()):
        raise ValueError("is_validation must contain boolean or binary (0, 1) flags.")
    is_val_mask = is_val_arr.astype(bool)

    if "episode_ids" not in arrays:
        raise ValueError("Dataset is missing mandatory 'episode_ids' field.")
    episode_ids = np.asarray(arrays["episode_ids"])
    if episode_ids.ndim != 1 or len(episode_ids) != num_transitions:
        raise ValueError(
            f"episode_ids shape {episode_ids.shape} is malformed; expected ({num_transitions},)"
        )

    unique_eids = np.unique(episode_ids)

    train_trajs: list[dict[str, np.ndarray]] = []
    val_trajs: list[dict[str, np.ndarray]] = []
    train_eids: list[int] = []
    val_eids: list[int] = []

    for eid in unique_eids:
        mask = episode_ids == eid
        ep_val_flags = is_val_mask[mask]
        has_val = bool(np.any(ep_val_flags))
        all_val = bool(np.all(ep_val_flags))

        # Trajectory-level split invariant: no episode may span both train and val splits
        if has_val and not all_val:
            raise ValueError(
                f"Episode {eid} contains mixed train and validation transitions; trajectory split violated."
            )

        terminals = (
            arrays["terminals"][mask]
            if "terminals" in arrays
            else arrays.get("dones", np.zeros(mask.sum(), dtype=bool))[mask]
        )
        traj = {
            "observations": arrays["observations"][mask],
            "actions": arrays["actions"][mask],
            "rewards": arrays["rewards"][mask],
            "returns_to_go": arrays["returns_to_go"][mask],
            "timesteps": arrays["timesteps"][mask],
            "terminals": terminals,
        }
        if all_val:
            val_trajs.append(traj)
            val_eids.append(int(eid))
        else:
            train_trajs.append(traj)
            train_eids.append(int(eid))

    # Cross-check dataset split against manifest if present
    if "train_episode_ids" in manifest_data:
        manifest_train = set(manifest_data["train_episode_ids"])
        if manifest_train and set(train_eids) != manifest_train:
            raise ValueError(
                f"Split mismatch: dataset train episode_ids {sorted(train_eids)} != manifest train_episode_ids {sorted(manifest_train)}"
            )
    if "validation_episode_ids" in manifest_data:
        manifest_val = set(manifest_data["validation_episode_ids"])
        if manifest_val and set(val_eids) != manifest_val:
            raise ValueError(
                f"Split mismatch: dataset val episode_ids {sorted(val_eids)} != manifest validation_episode_ids {sorted(manifest_val)}"
            )

    # Empty split checks: never silently fallback validation to training
    if not train_trajs:
        raise ValueError("Training split is empty; at least one training trajectory required.")
    if not val_trajs and not allow_no_validation:
        raise ValueError("Validation split is empty; at least one validation trajectory required in B0 protocol.")

    # 3. Define batch sampler closure matching baseline SequenceTrainer.get_batch
    def _make_sampler(trajectories: list[dict[str, np.ndarray]]):
        if not trajectories:
            return None
        lengths = np.array([len(t["observations"]) for t in trajectories])
        p_sample = lengths / lengths.sum()

        def sample_batch(batch_sz: int):
            batch_inds = np.random.choice(len(trajectories), size=batch_sz, replace=True, p=p_sample)
            s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
            for idx in batch_inds:
                traj = trajectories[idx]
                tlen = len(traj["observations"])
                si = random.randint(0, tlen - 1)

                traj_s = traj["observations"][si : si + k]
                traj_a = traj["actions"][si : si + k]
                traj_r = traj["rewards"][si : si + k]
                traj_rtg = traj["returns_to_go"][si : si + k + 1]
                traj_t = np.clip(traj["timesteps"][si : si + k], 0, horizon - 1)
                traj_d = traj["terminals"][si : si + k]

                cur_len = len(traj_s)
                pad_len = k - cur_len

                norm_s = (traj_s - state_mean) / state_std
                if len(traj_rtg) <= cur_len:
                    traj_rtg = np.append(traj_rtg, 0.0)

                # Variable semantics:
                # - raw_reward: r_t (unscaled environment reward)
                # - scaled_reward: \rho_t = r_t / c
                # - raw_return: G_t = \sum r_k
                # - scaled_rtg: Y_t = G_t / c (computed once in prepare_dataset)
                # - dt_rtg_token: token provided to DT model during training (Y_t) and inference (B_t)
                # Note: traj["returns_to_go"] is ALREADY scaled by reward_scale (Y_t = G_t / c).
                # Do NOT divide by reward_scale again to prevent double-scaling (G_t / c^2).
                dt_rtg_token = traj_rtg[:, None] if traj_rtg.ndim == 1 else traj_rtg

                if pad_len > 0:
                    pad_s = np.zeros((pad_len, state_dim), dtype=np.float32)
                    pad_a = np.ones((pad_len, act_dim), dtype=np.float32) * -10.0
                    pad_r = np.zeros((pad_len, 1), dtype=np.float32)
                    pad_d = np.ones(pad_len, dtype=np.int64) * 2
                    pad_rtg = np.zeros((pad_len, 1), dtype=np.float32)
                    pad_t = np.zeros(pad_len, dtype=np.int64)
                    pad_m = np.zeros(pad_len, dtype=np.float32)

                    s.append(np.vstack([pad_s, norm_s]))
                    a.append(np.vstack([pad_a, traj_a]))
                    r.append(np.vstack([pad_r, traj_r[:, None] if traj_r.ndim == 1 else traj_r]))
                    d.append(np.concatenate([pad_d, traj_d.astype(np.int64)]))
                    rtg.append(np.vstack([pad_rtg, dt_rtg_token]))
                    timesteps.append(np.concatenate([pad_t, traj_t]))
                    mask.append(np.concatenate([pad_m, np.ones(cur_len, dtype=np.float32)]))
                else:
                    s.append(norm_s)
                    a.append(traj_a)
                    r.append(traj_r[:, None] if traj_r.ndim == 1 else traj_r)
                    d.append(traj_d.astype(np.int64))
                    rtg.append(dt_rtg_token)
                    timesteps.append(traj_t)
                    mask.append(np.ones(cur_len, dtype=np.float32))

            return (
                torch.as_tensor(np.array(s), dtype=torch.float32, device=dev),
                torch.as_tensor(np.array(a), dtype=torch.float32, device=dev),
                torch.as_tensor(np.array(r), dtype=torch.float32, device=dev),
                torch.as_tensor(np.array(d), dtype=torch.long, device=dev),
                torch.as_tensor(np.array(rtg), dtype=torch.float32, device=dev),
                torch.as_tensor(np.array(timesteps), dtype=torch.long, device=dev),
                torch.as_tensor(np.array(mask), dtype=torch.float32, device=dev),
            )

        return sample_batch

    train_sampler = _make_sampler(train_trajs)
    val_sampler = _make_sampler(val_trajs)

    # 4. Construct architecture and baseline SequenceTrainer
    DecisionTransformer, SequenceTrainer = _import_baseline_dt(baseline_root=baseline_root)

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=embed_dim,
        max_length=k,
        max_ep_len=horizon,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=4 * embed_dim,
        activation_function="relu",
        n_positions=1024,
        n_ctx=1024,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model.to(dev)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / max(1, warmup_steps), 1.0),
    )

    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=train_sampler,
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
    )

    # 5. Training and evaluation loop reusing SequenceTrainer.train_step()
    model.train()
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_steps: list[int] = []
    best_val_loss: float | None = None
    best_step: int | None = None
    best_weights: dict[str, torch.Tensor] | None = None

    def _evaluate_validation() -> float:
        if val_sampler is None:
            raise RuntimeError("Validation sampler is not available.")
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for _ in range(eval_batches):
                vs, va, vr, vd, vrtg, vt, vm = val_sampler(batch_size)
                _, vpreds, _ = model.forward(
                    vs, va, vr, vrtg[:, :-1], vt, attention_mask=vm,
                )
                act_d = vpreds.shape[2]
                preds_masked = vpreds.reshape(-1, act_d)[vm.reshape(-1) > 0]
                targets_masked = va.reshape(-1, act_d)[vm.reshape(-1) > 0]
                losses.append(torch.mean((preds_masked - targets_masked) ** 2).item())
        model.train()
        return float(np.mean(losses)) if losses else 0.0

    for step in range(1, num_steps + 1):
        loss = trainer.train_step()
        if scheduler is not None:
            scheduler.step()
        train_losses.append(loss)

        if eval_interval is not None and val_sampler is not None and (step % eval_interval == 0 or step == num_steps):
            vloss = _evaluate_validation()
            val_losses.append(vloss)
            val_steps.append(step)
            if best_val_loss is None or vloss < best_val_loss:
                best_val_loss = vloss
                best_step = step
                best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # 6. Restore best validation checkpoint weights if validation was performed
    if best_weights is not None:
        model.load_state_dict(best_weights)
        model.to(dev)

    # 7. Freeze model parameters strictly and set to eval mode
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    metrics = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_steps": val_steps,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "state_dim": state_dim,
                "act_dim": act_dim,
                "hidden_size": embed_dim,
                "max_length": k,
                "max_ep_len": horizon,
                "n_layer": n_layer,
                "n_head": n_head,
                "state_mean": np.asarray(state_mean, dtype=np.float32).tolist(),
                "state_std": np.asarray(state_std, dtype=np.float32).tolist(),
                "reward_scale": reward_scale,
                "best_val_loss": best_val_loss,
                "best_step": best_step,
                "metrics": metrics,
            },
            out,
        )

    summary = {
        "metrics": metrics,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "num_train_trajectories": len(train_trajs),
        "num_val_trajectories": len(val_trajs),
    }
    return model, summary


def vanilla_dt_action(
    model: nn.Module,
    adapter: DTAdapter,
    obs: np.ndarray,
    rtg_token: float,
    step: int,
) -> np.ndarray:
    """Evaluate one vanilla-DT action without adapter-side intervention."""
    normalized = adapter.normalize_obs(obs)
    states = torch.as_tensor(normalized, dtype=torch.float32, device=adapter.device)
    states = states.reshape(1, 1, -1)
    actions = torch.zeros(1, 1, adapter.act_dim, dtype=torch.float32, device=adapter.device)
    rtgs = torch.tensor([[[float(rtg_token)]]], dtype=torch.float32, device=adapter.device)
    timesteps = torch.tensor([[int(step)]], dtype=torch.long, device=adapter.device)
    with torch.no_grad():
        action = model.get_action(states, actions, None, rtgs, timesteps)
    return action.detach().cpu().numpy().astype(np.float32).reshape(-1)


def verify_b0_parity(
    model: nn.Module,
    adapter: DTAdapter,
    *,
    observations: list[np.ndarray],
    rtg_tokens: list[float],
    atol: float = 1e-6,
    model_source: str = "real",
) -> dict[str, Any]:
    """Compare a fresh adapter's B0 action with direct vanilla DT inference."""
    adapter.reset()
    history_states: list[np.ndarray] = []
    history_actions: list[np.ndarray] = []
    history_rtgs: list[float] = []
    history_timesteps: list[int] = []
    k = adapter.max_length

    def direct_action(obs, rtg, step):
        history_states.append(adapter.normalize_obs(obs))
        history_actions.append(np.zeros(adapter.act_dim, dtype=np.float32))
        history_rtgs.append(float(rtg))
        history_timesteps.append(int(step))

        states = np.array(history_states[-k:], dtype=np.float32)[None]
        actions = np.array(history_actions[-k:], dtype=np.float32)[None]
        rtgs = np.array(history_rtgs[-k:], dtype=np.float32)[None, :, None]
        timesteps = np.array(history_timesteps[-k:], dtype=np.int64)[None]
        with torch.no_grad():
            action = model.get_action(
                torch.as_tensor(states, device=adapter.device),
                torch.as_tensor(actions, device=adapter.device),
                None,
                torch.as_tensor(rtgs, device=adapter.device),
                torch.as_tensor(timesteps, device=adapter.device),
            )
        action = action.detach().cpu().numpy().astype(np.float32).reshape(-1)
        if adapter.action_low is not None and adapter.action_high is not None:
            action = np.clip(action, adapter.action_low, adapter.action_high)
        history_actions[-1] = action.copy()
        return action

    differences = []
    for step, (obs, rtg) in enumerate(zip(observations, rtg_tokens)):
        direct = direct_action(obs, rtg, step)
        # B0 is exactly adapter.step with no R cap or U reranking.
        via_adapter = adapter.step(obs=obs, rtg_token=rtg, step=step)
        differences.append(float(np.max(np.abs(direct - via_adapter))))
    max_diff = max(differences) if differences else 0.0
    passed = bool(max_diff <= atol)
    return {
        "max_abs_difference": max_diff,
        "tolerance": float(atol),
        "passed": passed,
        "status": "pass" if passed else "fail",
        "model_source": model_source,
        "steps": len(differences),
    }


def assert_parameters_frozen(model: nn.Module) -> None:
    """Raise when any DT parameter is trainable or the model is in train mode."""
    if model.training:
        raise AssertionError("Decision Transformer must be in eval mode")
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if trainable:
        raise AssertionError(f"Decision Transformer parameters are not frozen: {trainable}")


def verify_n1_matches_dt_action(
    adapter: DTAdapter,
    controller: Any,
    *,
    obs: np.ndarray,
    rtg_token: float,
    step: int,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Verify N=1 returns the DT action of the corresponding R branch."""
    adapter.reset()
    expected = adapter.step(obs=obs, rtg_token=rtg_token, step=step)
    # The controller must see the same fresh episode context.
    adapter.reset()
    selected, info = controller.get_action(obs=obs, rtg_token=rtg_token, step=step)
    difference = float(np.max(np.abs(selected - expected)))
    return {
        "max_abs_difference": difference,
        "tolerance": float(atol),
        "passed": bool(difference <= atol),
        "selected_index": info.get("selected_index"),
    }
