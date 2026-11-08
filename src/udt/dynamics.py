"""Module U: Ensemble Dynamics and Disagreement-Penalized Action Reranking.

Implements RRD Sections 2.2, 2.3, and 2.4:
- M independent MLP dynamics models predicting delta state, reward, and terminal probability.
- Ensemble disagreement calculation.
- Local candidate generation with original action inclusion.
- Action reranking with disagreement penalty and horizon-aware continuation.
"""

import copy
from pathlib import Path
from typing import Any, Literal
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from udt.checkpoint import load_checkpoint_payload
from udt.dt_adapter import DTAdapter
from udt.intervention import ensemble_scores, propose_actions
from udt.return_cap import ReturnCapper


class DynamicsMLP(nn.Module):
    """Single dynamics transition head f_phi(z_t, a_t, t/H).

    Predicts:
    1. Normalized state difference delta_z = z_{t+1} - z_t in R^{d_s}.
    2. Scaled reward rho in R.
    3. Natural termination probability p^{term} in [0, 1].
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        activation: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        if state_dim <= 0 or act_dim <= 0:
            raise ValueError(f"state_dim ({state_dim}) and act_dim ({act_dim}) must be positive")
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.in_dim = state_dim + act_dim + 1

        layers: list[nn.Module] = [nn.Linear(self.in_dim, hidden_dim), activation()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation()])

        self.backbone = nn.Sequential(*layers)
        self.head_delta_z = nn.Linear(hidden_dim, state_dim)
        self.head_reward = nn.Linear(hidden_dim, 1)
        self.head_terminal = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        t_rel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass predicting transition statistics.

        Args:
            z: Normalized state tensor of shape (batch, state_dim).
            a: Action tensor of shape (batch, act_dim).
            t_rel: Relative timestep tensor of shape (batch, 1).

        Returns:
            delta_z: Predicted normalized state difference, shape (batch, state_dim).
            reward: Predicted scaled reward, shape (batch, 1).
            p_term: Predicted termination probability in [0, 1], shape (batch, 1).
        """
        x = torch.cat([z, a, t_rel], dim=-1)
        feat = self.backbone(x)
        delta_z = self.head_delta_z(feat)
        reward = self.head_reward(feat)
        p_term = torch.sigmoid(self.head_terminal(feat))
        return delta_z, reward, p_term


class DynamicsEnsemble:
    """Ensemble of M independent dynamics heads."""

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        num_models: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 3,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        state_mean: np.ndarray | None = None,
        state_std: np.ndarray | None = None,
        horizon: int | None = None,
        reward_scale: float | None = None,
        normalization_provenance: dict[str, Any] | None = None,
    ):
        if num_models < 1:
            raise ValueError(f"num_models must be >= 1, got {num_models}")

        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)
        self.num_models = num_models
        self.device = torch.device(device)
        self.state_mean = self._normalization_array(state_mean, "state_mean")
        self.state_std = self._normalization_array(state_std, "state_std")
        if (self.state_mean is None) != (self.state_std is None):
            raise ValueError("state_mean and state_std must be provided together")
        if self.state_std is not None and np.any(self.state_std <= 0):
            raise ValueError("state_std must be strictly positive")
        if normalization_provenance is not None:
            self.normalization_provenance = dict(normalization_provenance)
        elif self.state_mean is not None:
            self.normalization_provenance = {
                "source": "external_unverified",
                "verified": False,
            }
        else:
            self.normalization_provenance = {}
        self.horizon = None if horizon is None else int(horizon)
        if self.horizon is not None and self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        self.reward_scale = None if reward_scale is None else float(reward_scale)
        if self.reward_scale is not None and (
                not np.isfinite(self.reward_scale) or self.reward_scale <= 0):
            raise ValueError(f"reward_scale must be positive and finite, got {reward_scale}")

        # Initialization must be reproducible without leaking global RNG state.
        # Each head gets a scoped seed derived from the base seed.
        self.init_seeds: list[int | None] = []
        self.models: list[DynamicsMLP] = []
        for m_idx in range(num_models):
            head_seed = None if seed is None else int(seed) + m_idx * 1000
            self.init_seeds.append(head_seed)
            if head_seed is None:
                model = DynamicsMLP(
                    state_dim=state_dim,
                    act_dim=act_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                )
            else:
                # fork_rng restores global CPU RNG state after initialization.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(head_seed)
                    model = DynamicsMLP(
                        state_dim=state_dim,
                        act_dim=act_dim,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                    )
            self.models.append(model.to(self.device))
        for m in self.models:
            m.eval()

    def _normalization_array(self, value: np.ndarray | None, name: str) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (self.state_dim,):
            raise ValueError(f"{name} must have shape ({self.state_dim},), got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite")
        return arr

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize raw observations with dynamics-training statistics."""
        arr = np.asarray(obs, dtype=np.float32)
        if arr.shape[-1:] != (self.state_dim,):
            raise ValueError(
                f"observation must have trailing dimension {self.state_dim}, got {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError("observation contains non-finite values")
        if self.state_mean is None or self.state_std is None:
            return arr
        return (arr - self.state_mean) / self.state_std

    def unnormalize_state(self, normalized_state: np.ndarray) -> np.ndarray:
        """Map dynamics-normalized states back to raw observation units."""
        arr = np.asarray(normalized_state, dtype=np.float32)
        if arr.shape[-1:] != (self.state_dim,):
            raise ValueError(
                f"normalized state must have trailing dimension {self.state_dim}, got {arr.shape}"
            )
        if self.state_mean is None or self.state_std is None:
            return arr
        return arr * self.state_std + self.state_mean

    def to(self, device: str | torch.device) -> "DynamicsEnsemble":
        self.device = torch.device(device)
        for m in self.models:
            m.to(self.device)
        return self

    def eval(self) -> "DynamicsEnsemble":
        for m in self.models:
            m.eval()
        return self

    def predict(
        self,
        z: np.ndarray,
        actions: np.ndarray,
        t_rel: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict next states, rewards, and termination probabilities for N candidate actions.

        Args:
            z: Single normalized state vector of shape (state_dim,).
            actions: Candidate action array of shape (N, act_dim).
            t_rel: Relative timestep t/H clamped to [0, 1].

        Returns:
            next_states: Array of shape (M, N, state_dim).
            rewards: Array of shape (M, N).
            terminals: Array of shape (M, N).
        """
        z_arr = np.asarray(z, dtype=np.float32)
        act_arr = np.asarray(actions, dtype=np.float32)
        n_candidates = act_arr.shape[0]

        # Broadcast z and t_rel across all N candidates
        z_batch = np.broadcast_to(z_arr, (n_candidates, self.state_dim)).copy()
        t_batch = np.full((n_candidates, 1), float(t_rel), dtype=np.float32)

        z_tensor = torch.as_tensor(z_batch, dtype=torch.float32, device=self.device)
        a_tensor = torch.as_tensor(act_arr, dtype=torch.float32, device=self.device)
        t_tensor = torch.as_tensor(t_batch, dtype=torch.float32, device=self.device)

        m_next_states = []
        m_rewards = []
        m_terminals = []

        with torch.no_grad():
            for model in self.models:
                delta_z, rew, p_term = model(z_tensor, a_tensor, t_tensor)
                # Next state: z_{t+1} = z_t + delta_z
                next_z = z_tensor + delta_z
                m_next_states.append(next_z.cpu().numpy())
                m_rewards.append(rew.squeeze(-1).cpu().numpy())
                m_terminals.append(p_term.squeeze(-1).cpu().numpy())

        return (
            np.stack(m_next_states, axis=0),  # (M, N, state_dim)
            np.stack(m_rewards, axis=0),      # (M, N)
            np.stack(m_terminals, axis=0),    # (M, N)
        )


def _masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor | None, int]:
    """Mean squared error with the valid denominator, or None if empty."""
    valid = float(mask.sum().item())
    if valid == 0:
        return None, 0
    squared = (prediction - target) ** 2
    numerator = (squared * mask).sum()
    denominator = valid * float(target.shape[-1])
    return numerator / denominator, int(valid)


def _masked_bce(
    probability: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor | None, int]:
    """Binary cross-entropy with the valid denominator, or None if empty."""
    valid = float(mask.sum().item())
    if valid == 0:
        return None, 0
    bce = nn.functional.binary_cross_entropy(probability, target, reduction="none")
    return (bce * mask).sum() / valid, int(valid)


class _LossAccumulator:
    """Exact epoch-level per-output losses with masked denominators."""

    def __init__(self, state_dim: int):
        self.state_dim = int(state_dim)
        self.delta_num = 0.0
        self.delta_den = 0
        self.reward_num = 0.0
        self.reward_den = 0
        self.terminal_num = 0.0
        self.terminal_den = 0

    def add(
        self,
        pred_dz: torch.Tensor,
        target_dz: torch.Tensor,
        mask_dz: torch.Tensor,
        pred_r: torch.Tensor,
        target_r: torch.Tensor,
        mask_r: torch.Tensor,
        pred_term: torch.Tensor,
        target_term: torch.Tensor,
        mask_term: torch.Tensor,
    ) -> None:
        if mask_dz.sum().item() > 0:
            self.delta_num += float((((pred_dz - target_dz) ** 2) * mask_dz).sum().item())
            self.delta_den += int(mask_dz.sum().item()) * self.state_dim
        if mask_r.sum().item() > 0:
            self.reward_num += float((((pred_r - target_r) ** 2) * mask_r).sum().item())
            self.reward_den += int(mask_r.sum().item())
        if mask_term.sum().item() > 0:
            bce = nn.functional.binary_cross_entropy(pred_term, target_term, reduction="none")
            self.terminal_num += float((bce * mask_term).sum().item())
            self.terminal_den += int(mask_term.sum().item())

    @property
    def delta_loss(self) -> float:
        return self.delta_num / self.delta_den if self.delta_den else 0.0

    @property
    def reward_loss(self) -> float:
        return self.reward_num / self.reward_den if self.reward_den else 0.0

    @property
    def terminal_loss(self) -> float:
        return self.terminal_num / self.terminal_den if self.terminal_den else 0.0

    def weighted_loss(self, w_reward: float, w_terminal: float) -> float:
        return self.delta_loss + w_reward * self.reward_loss + w_terminal * self.terminal_loss


def _sample_mask(dataset: dict[str, np.ndarray], key: str, n_samples: int) -> np.ndarray:
    """Return a boolean per-transition supervision mask.

    Missing masks mean the field was observed for every transition (the
    prepared-data path supplies explicit masks when next states are inferred).
    """
    if key not in dataset:
        return np.ones(n_samples, dtype=bool)
    mask = np.asarray(dataset[key])
    if mask.shape != (n_samples,):
        raise ValueError(f"{key} must have shape ({n_samples},), got {mask.shape}")
    if not np.issubdtype(mask.dtype, np.bool_):
        raise ValueError(f"{key} must be a boolean mask, got dtype {mask.dtype}")
    return mask


def train_dynamics_ensemble(
    dataset: dict[str, np.ndarray],
    horizon: int,
    num_models: int = 3,
    state_mean: np.ndarray | None = None,
    state_std: np.ndarray | None = None,
    scale: float = 1000.0,
    hidden_dim: int = 128,
    num_layers: int = 3,
    w_reward: float = 1.0,
    w_terminal: float = 1.0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 15,
    batch_size: int = 256,
    device: str = "cpu",
    patience: int = 5,
    seed: int | None = None,
    normalization_provenance: dict[str, Any] | None = None,
) -> tuple[DynamicsEnsemble, dict[str, Any]]:
    """Train M independent dynamics transition models on offline dataset.

    Args:
        dataset: Prepared dataset containing 'observations', 'next_observations',
                 'actions', 'rewards', 'terminals', 'timesteps', 'is_validation'.
        horizon: Maximum episode horizon H.
        num_models: Number of independent ensemble heads M.
        state_mean: Observation normalization mean.
        state_std: Observation normalization std.
        scale: Reward scaling constant c.
        hidden_dim: Hidden unit dimension.
        num_layers: Number of MLP layers per model.
        w_reward: Weight for reward prediction loss.
        w_terminal: Weight for terminal binary cross-entropy loss.
        lr: Adam learning rate.
        weight_decay: L2 regularization coefficient.
        epochs: Training epochs.
        batch_size: Batch size.
        device: Device string.
        patience: Early stopping patience.
        seed: Base random seed.

    Returns:
        (DynamicsEnsemble, metrics_dict).
    """
    for req in ("observations", "next_observations", "actions", "rewards", "terminals",
                "timesteps", "episode_ids", "is_validation"):
        if req not in dataset:
            raise ValueError(f"Dataset missing required key: {req}")

    if not np.isfinite(horizon) or int(horizon) <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    horizon = int(horizon)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be positive and finite, got {scale}")
    if not np.isfinite(w_reward) or w_reward < 0:
        raise ValueError(f"w_reward must be finite and nonnegative, got {w_reward}")
    if not np.isfinite(w_terminal) or w_terminal < 0:
        raise ValueError(f"w_terminal must be finite and nonnegative, got {w_terminal}")
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError(f"lr must be positive and finite, got {lr}")
    if not isinstance(epochs, (int, np.integer)) or epochs <= 0:
        raise ValueError(f"epochs must be a positive integer, got {epochs}")
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

    obs = np.asarray(dataset["observations"])
    next_obs = np.asarray(dataset["next_observations"])
    actions = np.asarray(dataset["actions"])
    rewards = np.asarray(dataset["rewards"])
    terminals_arr = np.asarray(dataset["terminals"])
    timesteps_arr = np.asarray(dataset["timesteps"])

    if obs.ndim != 2 or obs.shape[1] == 0:
        raise ValueError(f"observations must have shape (N, state_dim), got {obs.shape}")
    n_samples = len(obs)
    if n_samples == 0:
        raise ValueError("Dataset is empty")

    if next_obs.shape != obs.shape:
        raise ValueError(
            f"next_observations shape {next_obs.shape} must match observations {obs.shape}"
        )
    if actions.ndim != 2 or actions.shape[1] == 0 or len(actions) != n_samples:
        raise ValueError(f"actions must have shape ({n_samples}, act_dim), got {actions.shape}")
    if rewards.shape != (n_samples,):
        raise ValueError(f"rewards must have shape ({n_samples},), got {rewards.shape}")
    if terminals_arr.shape != (n_samples,):
        raise ValueError(f"terminals must have shape ({n_samples},), got {terminals_arr.shape}")
    if timesteps_arr.shape != (n_samples,):
        raise ValueError(f"timesteps must have shape ({n_samples},), got {timesteps_arr.shape}")

    for name, arr in (("observations", obs), ("actions", actions),
                      ("rewards", rewards), ("terminals", terminals_arr),
                      ("timesteps", timesteps_arr)):
        if len(arr) != n_samples:
            raise ValueError(f"{name} length mismatch: {len(arr)} != {n_samples}")
    if not np.issubdtype(obs.dtype, np.number):
        raise ValueError(f"observations must be numeric, got dtype {obs.dtype}")
    if not np.issubdtype(actions.dtype, np.number):
        raise ValueError(f"actions must be numeric, got dtype {actions.dtype}")
    if not np.issubdtype(rewards.dtype, np.number):
        raise ValueError(f"rewards must be numeric, got dtype {rewards.dtype}")
    if not np.isfinite(obs).all():
        raise ValueError("observations contains non-finite values")
    if not np.isfinite(actions).all():
        raise ValueError("actions contains non-finite values")
    if not np.isfinite(rewards).all():
        raise ValueError("rewards contains non-finite values")
    if not np.isfinite(timesteps_arr).all():
        raise ValueError("timesteps contains non-finite values")
    if not np.issubdtype(timesteps_arr.dtype, np.integer):
        raise ValueError(f"timesteps must be integer-valued, got dtype {timesteps_arr.dtype}")
    if np.any(timesteps_arr < 0) or np.any(timesteps_arr >= horizon):
        raise ValueError(
            f"timesteps must lie in [0, {horizon - 1}], got "
            f"[{int(np.min(timesteps_arr))}, {int(np.max(timesteps_arr))}]"
        )
    if terminals_arr.dtype != bool:
        if not np.issubdtype(terminals_arr.dtype, np.number) or not np.isin(terminals_arr, [0, 1]).all():
            raise ValueError("terminals must be boolean or 0/1 flags")

    episode_arr = np.asarray(dataset["episode_ids"])
    if episode_arr.shape != (n_samples,):
        raise ValueError(f"episode_ids must have shape ({n_samples},), got {episode_arr.shape}")
    if not np.issubdtype(episode_arr.dtype, np.integer):
        raise ValueError(f"episode_ids must be integer-valued, got dtype {episode_arr.dtype}")

    if "horizon" in dataset:
        ds_horizon = int(np.asarray(dataset["horizon"]).item())
        if ds_horizon != horizon:
            raise ValueError(
                f"Horizon mismatch: dataset horizon ({ds_horizon}) != training horizon ({horizon})"
            )
    if "reward_scale" in dataset:
        ds_scale = float(np.asarray(dataset["reward_scale"]).item())
        if not np.isclose(ds_scale, scale):
            raise ValueError(
                f"Reward scale mismatch: dataset reward_scale ({ds_scale}) != training scale ({scale})"
            )

    is_val = dataset["is_validation"]
    if not (isinstance(is_val, np.ndarray) and np.issubdtype(is_val.dtype, np.bool_)):
        raise ValueError("is_validation must be boolean numpy array")
    if is_val.shape != (n_samples,):
        raise ValueError(f"is_validation must have shape ({n_samples},), got {is_val.shape}")

    episode_ids = np.asarray(dataset["episode_ids"])
    if episode_ids.shape != (n_samples,):
        raise ValueError(f"episode_ids must have shape ({n_samples},), got {episode_ids.shape}")

    train_mask = ~is_val
    val_mask = is_val
    if np.sum(train_mask) == 0 or np.sum(val_mask) == 0:
        raise ValueError("Dataset must contain both training and validation samples")

    # Trajectory splits must not leak: every episode belongs wholly to one side.
    train_episode_ids = np.unique(episode_ids[train_mask])
    validation_episode_ids = np.unique(episode_ids[val_mask])
    overlap = np.intersect1d(train_episode_ids, validation_episode_ids)
    if overlap.size:
        raise ValueError(
            f"Train/validation episode leakage detected for episode IDs: {overlap.tolist()}"
        )

    state_dim = dataset["observations"].shape[1]
    act_dim = dataset["actions"].shape[1]

    # Supervised targets have separate validity masks. An invalid post-terminal
    # next observation must not leak a fabricated delta target, but the same
    # transition can still supervise reward and natural-terminal prediction.
    delta_valid = _sample_mask(dataset, "next_observations_valid", n_samples)
    reward_valid = _sample_mask(dataset, "rewards_valid", n_samples)
    terminal_valid = _sample_mask(dataset, "terminals_valid", n_samples)

    if "valid_next_states" in dataset:
        delta_valid = delta_valid & _sample_mask(dataset, "valid_next_states", n_samples)
    if "valid_rewards" in dataset:
        reward_valid = reward_valid & _sample_mask(dataset, "valid_rewards", n_samples)
    if "valid_terminals" in dataset:
        terminal_valid = terminal_valid & _sample_mask(dataset, "valid_terminals", n_samples)

    if not np.isfinite(next_obs[delta_valid]).all():
        raise ValueError(
            "next_observations contains non-finite values at transitions marked "
            "next_observations_valid=True"
        )

    if np.sum(delta_valid & train_mask) == 0:
        raise ValueError(
            "Training split has no valid next-state transitions for delta-state supervision; "
            "provide at least one transition with next_observations_valid=True"
        )

    if np.sum(delta_valid & val_mask) == 0:
        raise ValueError(
            "Validation split has no valid next-state transitions for delta-state supervision; "
            "provide at least one transition with next_observations_valid=True"
        )

    data_summary = {
        "train_transitions": int(np.sum(train_mask)),
        "validation_transitions": int(np.sum(val_mask)),
        "train_delta_state_samples": int(np.sum(train_mask & delta_valid)),
        "train_reward_samples": int(np.sum(train_mask & reward_valid)),
        "train_terminal_samples": int(np.sum(train_mask & terminal_valid)),
        "train_invalid_next_state_transitions": int(np.sum(train_mask & ~delta_valid)),
        "validation_delta_state_samples": int(np.sum(val_mask & delta_valid)),
        "validation_reward_samples": int(np.sum(val_mask & reward_valid)),
        "validation_terminal_samples": int(np.sum(val_mask & terminal_valid)),
    }

    if state_mean is None or state_std is None:
        train_obs_raw = dataset["observations"][train_mask]
        state_mean = np.mean(train_obs_raw, axis=0, dtype=np.float32)
        state_std = np.maximum(np.std(train_obs_raw, axis=0, dtype=np.float32), 1e-6)
        norm_provenance = {
            "source": "dynamics_training_split",
            "verified": True,
        }
    else:
        state_mean = np.asarray(state_mean, dtype=np.float32)
        state_std = np.asarray(state_std, dtype=np.float32)
        if state_mean.shape != (state_dim,) or state_std.shape != (state_dim,):
            raise ValueError(
                f"state_mean/state_std must have shape ({state_dim},), got "
                f"{state_mean.shape}/{state_std.shape}"
            )
        if not np.isfinite(state_mean).all() or not np.isfinite(state_std).all():
            raise ValueError("state_mean/state_std must be finite")
        if np.any(state_std <= 0):
            raise ValueError("state_std must be strictly positive")
        if normalization_provenance is not None and isinstance(normalization_provenance, dict):
            norm_provenance = dict(normalization_provenance)
            if norm_provenance.get("source") == "dynamics_training_split":
                raise ValueError("Externally supplied normalization statistics cannot claim source='dynamics_training_split'")
        else:
            norm_provenance = {
                "source": "external_unverified",
                "verified": False,
            }

    # Normalize observations and compute delta targets. Invalid next states are
    # replaced before normalization so an extreme placeholder cannot overflow or
    # influence the loss; the mask below is the source of truth.
    next_obs_raw = np.array(dataset["next_observations"], dtype=np.float32, copy=True)
    next_obs_raw[~delta_valid] = dataset["observations"][~delta_valid]
    norm_obs = ((dataset["observations"] - state_mean) / state_std).astype(np.float32)
    norm_next_obs = ((next_obs_raw - state_mean) / state_std).astype(np.float32)
    delta_z = (norm_next_obs - norm_obs).astype(np.float32)

    actions = dataset["actions"].astype(np.float32)
    scaled_rewards = (dataset["rewards"] / scale).astype(np.float32)[:, None]
    terminals = dataset["terminals"].astype(np.float32)[:, None]
    t_rel = np.clip(dataset["timesteps"].astype(np.float32) / float(horizon), 0.0, 1.0)[:, None]
    delta_mask_f = delta_valid.astype(np.float32)[:, None]
    reward_mask_f = reward_valid.astype(np.float32)[:, None]
    terminal_mask_f = terminal_valid.astype(np.float32)[:, None]

    torch_device = torch.device(device)
    ensemble = DynamicsEnsemble(
        state_dim=state_dim,
        act_dim=act_dim,
        num_models=num_models,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=torch_device,
        seed=seed,
        state_mean=state_mean,
        state_std=state_std,
        horizon=horizon,
        reward_scale=scale,
        normalization_provenance=norm_provenance,
    )

    ensemble_metrics: list[dict[str, Any]] = []

    train_positions = np.flatnonzero(train_mask)
    train_episodes_per_row = episode_ids[train_positions]

    # Train each model independently with trajectory-level bootstrap.
    for m_idx in range(num_models):
        head_seed = (seed + m_idx * 1000) if seed is not None else None

        model = ensemble.models[m_idx]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        bootstrap_rng = np.random.default_rng(head_seed)
        sampled_episodes = bootstrap_rng.choice(
            train_episode_ids,
            size=len(train_episode_ids),
            replace=True,
        )
        episode_counts = {
            int(eid): int(count)
            for eid, count in zip(*np.unique(sampled_episodes, return_counts=True))
        }
        # Preserve every transition of each sampled episode, including each
        # repetition of an episode drawn more than once.
        boot_rows: list[np.ndarray] = []
        for eid, count in episode_counts.items():
            rows = train_positions[train_episodes_per_row == eid]
            boot_rows.extend([rows] * count)
        train_rows = np.concatenate(boot_rows).astype(np.int64)

        valid_delta_count = int(np.sum(delta_valid[train_rows]))
        valid_reward_count = int(np.sum(reward_valid[train_rows]))
        valid_terminal_count = int(np.sum(terminal_valid[train_rows]))

        if valid_delta_count == 0:
            raise RuntimeError(
                f"Bootstrap head {m_idx} contains 0 valid delta-state transitions. "
                "Dynamics model cannot be trained without delta-state supervision."
            )

        bootstrap_info = {
            "init_seed": ensemble.init_seeds[m_idx],
            "bootstrap_seed": head_seed,
            "shuffle_seed": head_seed,
            "sampled_episode_ids": [int(v) for v in sampled_episodes.tolist()],
            "episode_counts": episode_counts,
            "unique_episodes": len(episode_counts),
            "bootstrap_transitions": int(len(train_rows)),
            "valid_delta_transitions": valid_delta_count,
            "valid_reward_transitions": valid_reward_count,
            "valid_terminal_transitions": valid_terminal_count,
        }

        # Datasets. Validity masks travel with the batch so each output uses
        # its own denominator instead of silently dropping whole transitions.
        train_ds = TensorDataset(
            torch.from_numpy(norm_obs[train_rows]),
            torch.from_numpy(actions[train_rows]),
            torch.from_numpy(t_rel[train_rows]),
            torch.from_numpy(delta_z[train_rows]),
            torch.from_numpy(scaled_rewards[train_rows]),
            torch.from_numpy(terminals[train_rows]),
            torch.from_numpy(delta_mask_f[train_rows]),
            torch.from_numpy(reward_mask_f[train_rows]),
            torch.from_numpy(terminal_mask_f[train_rows]),
        )
        val_ds = TensorDataset(
            torch.from_numpy(norm_obs[val_mask]),
            torch.from_numpy(actions[val_mask]),
            torch.from_numpy(t_rel[val_mask]),
            torch.from_numpy(delta_z[val_mask]),
            torch.from_numpy(scaled_rewards[val_mask]),
            torch.from_numpy(terminals[val_mask]),
            torch.from_numpy(delta_mask_f[val_mask]),
            torch.from_numpy(reward_mask_f[val_mask]),
            torch.from_numpy(terminal_mask_f[val_mask]),
        )

        train_gen = torch.Generator()
        if head_seed is not None:
            train_gen.manual_seed(head_seed)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=train_gen)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        best_val_loss = float("inf")
        best_weights = copy.deepcopy(model.state_dict())
        best_epoch = 0
        stagnant = 0
        train_loss_hist = []
        val_loss_hist = []
        train_delta_hist = []
        train_reward_hist = []
        train_terminal_hist = []
        total_skipped_minibatches = 0

        for epoch in range(epochs):
            model.train()
            train_acc = _LossAccumulator(state_dim)
            for batch in train_loader:
                b_z, b_a, b_t, b_dz, b_r, b_term, b_dm, b_rm, b_tm = batch
                b_z, b_a, b_t = b_z.to(torch_device), b_a.to(torch_device), b_t.to(torch_device)
                b_dz, b_r, b_term = b_dz.to(torch_device), b_r.to(torch_device), b_term.to(torch_device)
                b_dm, b_rm, b_tm = b_dm.to(torch_device), b_rm.to(torch_device), b_tm.to(torch_device)

                pred_dz, pred_r, pred_term = model(b_z, b_a, b_t)

                loss_dz, _ = _masked_mse(pred_dz, b_dz, b_dm)
                loss_r, _ = _masked_mse(pred_r, b_r, b_rm)
                loss_term, _ = _masked_bce(pred_term, b_term, b_tm)

                has_supervision = False
                batch_loss = None
                if loss_dz is not None:
                    batch_loss = loss_dz if batch_loss is None else batch_loss + loss_dz
                    has_supervision = True
                if loss_r is not None:
                    loss_r_weighted = w_reward * loss_r
                    batch_loss = loss_r_weighted if batch_loss is None else batch_loss + loss_r_weighted
                    has_supervision = True
                if loss_term is not None:
                    loss_term_weighted = w_terminal * loss_term
                    batch_loss = loss_term_weighted if batch_loss is None else batch_loss + loss_term_weighted
                    has_supervision = True

                if has_supervision and batch_loss is not None:
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    train_acc.add(
                        pred_dz.detach(), b_dz, b_dm,
                        pred_r.detach(), b_r, b_rm,
                        pred_term.detach(), b_term, b_tm,
                    )
                else:
                    total_skipped_minibatches += 1

            epoch_train_loss = train_acc.weighted_loss(w_reward, w_terminal)
            train_loss_hist.append(epoch_train_loss)
            train_delta_hist.append(train_acc.delta_loss)
            train_reward_hist.append(train_acc.reward_loss)
            train_terminal_hist.append(train_acc.terminal_loss)

            # Validation
            model.eval()
            val_acc = _LossAccumulator(state_dim)
            with torch.no_grad():
                for batch in val_loader:
                    b_z, b_a, b_t, b_dz, b_r, b_term, b_dm, b_rm, b_tm = batch
                    b_z, b_a, b_t = b_z.to(torch_device), b_a.to(torch_device), b_t.to(torch_device)
                    b_dz, b_r, b_term = b_dz.to(torch_device), b_r.to(torch_device), b_term.to(torch_device)
                    b_dm, b_rm, b_tm = b_dm.to(torch_device), b_rm.to(torch_device), b_tm.to(torch_device)

                    pred_dz, pred_r, pred_term = model(b_z, b_a, b_t)
                    val_acc.add(pred_dz, b_dz, b_dm, pred_r, b_r, b_rm,
                                pred_term, b_term, b_tm)

            epoch_val_loss = val_acc.weighted_loss(w_reward, w_terminal)
            val_loss_hist.append(epoch_val_loss)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                best_weights = copy.deepcopy(model.state_dict())
                stagnant = 0
            else:
                stagnant += 1
                if stagnant >= patience:
                    break

        model.load_state_dict(best_weights)
        model.eval()

        ensemble_metrics.append({
            "model_index": m_idx,
            "train_loss": train_loss_hist,
            "val_loss": val_loss_hist,
            "train_delta_loss": train_delta_hist,
            "train_reward_loss": train_reward_hist,
            "train_terminal_loss": train_terminal_hist,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "skipped_minibatches": total_skipped_minibatches,
            "bootstrap": bootstrap_info,
        })

    data_summary.update({
        "train_episode_ids": [int(v) for v in train_episode_ids.tolist()],
        "validation_episode_ids": [int(v) for v in validation_episode_ids.tolist()],
        "train_episodes": int(len(train_episode_ids)),
        "validation_episodes": int(len(validation_episode_ids)),
    })
    training_config = {
        "num_models": int(num_models),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "w_reward": float(w_reward),
        "w_terminal": float(w_terminal),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "patience": int(patience),
        "seed": None if seed is None else int(seed),
        "device": str(device),
    }
    return ensemble, {
        "models": ensemble_metrics,
        "data_summary": data_summary,
        "base_seed": seed,
        "training_config": training_config,
    }


def save_dynamics_checkpoint(
    path: str | Path,
    ensemble: DynamicsEnsemble,
    *,
    metrics: dict[str, Any] | None = None,
    dataset_provenance: dict[str, Any] | None = None,
) -> Path:
    """Save weights together with the exact preprocessing and training recipe."""
    import json
    from pathlib import Path as _Path

    target = _Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if ensemble.state_mean is None or ensemble.state_std is None:
        raise ValueError("Cannot save dynamics checkpoint without normalization statistics")
    if ensemble.horizon is None or ensemble.reward_scale is None:
        raise ValueError("Cannot save dynamics checkpoint without horizon and reward_scale")

    norm_provenance = getattr(ensemble, "normalization_provenance", None) or {}
    source = norm_provenance.get("source", "external_unverified")
    verified = norm_provenance.get("verified", False)
    norm_dict = {
        "state_mean": np.asarray(ensemble.state_mean).tolist(),
        "state_std": np.asarray(ensemble.state_std).tolist(),
        "coordinate_system": "raw_observation",
        "source": source,
        "verified": verified,
    }
    for k, v in norm_provenance.items():
        if k not in norm_dict:
            norm_dict[k] = v

    payload = {
        "format_version": 1,
        "kind": "udt_dynamics_ensemble",
        "architecture": {
            "state_dim": ensemble.state_dim,
            "act_dim": ensemble.act_dim,
            "num_models": ensemble.num_models,
            "hidden_dim": ensemble.models[0].backbone[0].out_features,
            "num_layers": _count_mlp_layers(ensemble.models[0]),
        },
        "normalization": norm_dict,
        "horizon": int(ensemble.horizon),
        "reward_scale": float(ensemble.reward_scale),
        "loss_weights": {
            "delta_state": 1.0,
            "reward": (metrics or {}).get("training_config", {}).get("w_reward"),
            "terminal": (metrics or {}).get("training_config", {}).get("w_terminal"),
        },
        "training_config": (metrics or {}).get("training_config"),
        "bootstrap": [m.get("bootstrap") for m in (metrics or {}).get("models", [])],
        "init_seeds": list(ensemble.init_seeds),
        "dataset_provenance": dataset_provenance,
        "metrics": metrics,
        "models_state_dict": [m.state_dict() for m in ensemble.models],
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)
    return target


def _count_mlp_layers(model: DynamicsMLP) -> int:
    """Recover the configured number of linear layers from the backbone."""
    import torch.nn as _nn

    return sum(1 for module in model.backbone if isinstance(module, _nn.Linear)) + 1


def load_dynamics_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_state_dim: int | None = None,
    expected_act_dim: int | None = None,
    expected_horizon: int | None = None,
    expected_reward_scale: float | None = None,
    weights_only: bool = True,
    allow_unsafe_legacy: bool = False,
) -> tuple[DynamicsEnsemble, dict[str, Any]]:
    """Load a dynamics checkpoint and reject incompatible configurations."""
    payload = load_checkpoint_payload(
        path,
        weights_only=weights_only,
        allow_unsafe_legacy=allow_unsafe_legacy,
    )
    if payload.get("kind") != "udt_dynamics_ensemble":
        raise ValueError(f"{path} is not a U-DT dynamics ensemble checkpoint")
    if payload.get("format_version") != 1:
        raise ValueError(
            f"Unsupported dynamics checkpoint format_version={payload.get('format_version')}"
        )

    arch = payload.get("architecture", {})
    required = ("state_dim", "act_dim", "num_models", "hidden_dim", "num_layers")
    missing = [key for key in required if key not in arch]
    if missing:
        raise ValueError(f"Dynamics checkpoint architecture missing fields: {missing}")

    checks = {
        "state_dim": (arch["state_dim"], expected_state_dim),
        "act_dim": (arch["act_dim"], expected_act_dim),
        "horizon": (payload.get("horizon"), expected_horizon),
        "reward_scale": (payload.get("reward_scale"), expected_reward_scale),
    }
    for name, (actual, expected) in checks.items():
        if expected is None:
            continue
        if name == "reward_scale":
            if not np.isclose(float(actual), float(expected)):
                raise ValueError(
                    f"Checkpoint {name} ({actual}) is incompatible with expected "
                    f"configuration ({expected})"
                )
        elif int(actual) != int(expected):
            raise ValueError(
                f"Checkpoint {name} ({actual}) is incompatible with expected "
                f"configuration ({expected})"
            )

    norm = payload.get("normalization", {})
    if "state_mean" not in norm or "state_std" not in norm:
        raise ValueError("Dynamics checkpoint is missing normalization statistics")
    state_mean = np.asarray(norm["state_mean"], dtype=np.float32)
    state_std = np.asarray(norm["state_std"], dtype=np.float32)

    norm_provenance = {
        "source": norm.get("source", "external_unverified"),
        "verified": norm.get("verified", False),
    }
    for k, v in norm.items():
        if k not in ("state_mean", "state_std", "coordinate_system", "source", "verified"):
            norm_provenance[k] = v

    ensemble = DynamicsEnsemble(
        state_dim=int(arch["state_dim"]),
        act_dim=int(arch["act_dim"]),
        num_models=int(arch["num_models"]),
        hidden_dim=int(arch["hidden_dim"]),
        num_layers=int(arch["num_layers"]),
        device=device,
        state_mean=state_mean,
        state_std=state_std,
        horizon=int(payload["horizon"]),
        reward_scale=float(payload["reward_scale"]),
        normalization_provenance=norm_provenance,
    )
    state_dicts = payload.get("models_state_dict")
    if not isinstance(state_dicts, list) or len(state_dicts) != ensemble.num_models:
        raise ValueError("Dynamics checkpoint model list does not match num_models")
    for model, state_dict in zip(ensemble.models, state_dicts):
        model.load_state_dict(state_dict)
        model.eval()
    return ensemble, payload


class ModuleUController:
    """Module U: Disagreement-Penalized Action Reranking Controller.

    Evaluates N candidate actions proposed around the baseline Decision Transformer
    action a_{0, t}, predicting next states and rewards via an M-model dynamics ensemble,
    and scoring with disagreement penalty beta.
    """

    def __init__(
        self,
        dt_adapter: DTAdapter,
        ensemble: DynamicsEnsemble,
        return_model: ReturnCapper | None = None,
        *,
        capper: ReturnCapper | None = None,
        scorer_mode: Literal["return_model", "immediate_reward"] = "return_model",
        num_candidates: int = 10,
        radius: float = 0.05,
        beta: float = 0.1,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        horizon: int | None = None,
        seed: int | None = None,
    ):
        self.dt_adapter = dt_adapter
        self.ensemble = ensemble

        # `capper` is retained as a backwards-compatible alias, but scoring is
        # controlled only by the continuation return model and scorer_mode.
        if return_model is not None and capper is not None and return_model is not capper:
            raise ValueError("Pass either return_model or capper, not two different models")
        self.return_model = return_model if return_model is not None else capper
        self.capper = self.return_model
        if scorer_mode not in ("return_model", "immediate_reward"):
            raise ValueError(
                f"scorer_mode must be 'return_model' or 'immediate_reward', got {scorer_mode}"
            )
        self.scorer_mode = scorer_mode
        if scorer_mode == "return_model" and self.return_model is None:
            raise ValueError(
                "Module U requires a return model g_psi for continuation values even when "
                "Module R (RTG clipping) is disabled. Pass return_model=capper; for the "
                "explicit immediate-reward ablation use scorer_mode='immediate_reward'."
            )
        if scorer_mode == "immediate_reward":
            self.return_model = None
            self.capper = None

        self.num_candidates = int(num_candidates)
        if self.num_candidates < 1:
            raise ValueError(f"num_candidates must be >= 1, got {num_candidates}")
        self.radius = float(radius)
        if not np.isfinite(self.radius) or self.radius < 0:
            raise ValueError(f"radius must be finite and nonnegative, got {radius}")
        self.beta = float(beta)
        if not np.isfinite(self.beta) or self.beta < 0:
            raise ValueError(f"beta must be finite and nonnegative, got {beta}")
        resolved_horizon = horizon
        if resolved_horizon is None and self.return_model is not None:
            resolved_horizon = self.return_model.horizon
        if resolved_horizon is None:
            resolved_horizon = ensemble.horizon if ensemble.horizon is not None else 1000
        self.horizon = int(resolved_horizon)
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {resolved_horizon}")

        if ensemble.horizon is not None and ensemble.horizon != self.horizon:
            raise ValueError(
                f"Dynamics ensemble horizon ({ensemble.horizon}) != controller horizon ({self.horizon})"
            )

        if ensemble.state_dim != dt_adapter.state_dim:
            raise ValueError(
                f"Dynamics state_dim ({ensemble.state_dim}) != DT state_dim "
                f"({dt_adapter.state_dim})"
            )
        if ensemble.act_dim != dt_adapter.act_dim:
            raise ValueError(
                f"Dynamics act_dim ({ensemble.act_dim}) != DT act_dim ({dt_adapter.act_dim})"
            )
        if self.return_model is not None:
            if (ensemble.reward_scale is not None
                    and not np.isclose(ensemble.reward_scale, self.return_model.scale)):
                raise ValueError(
                    "Dynamics reward_scale "
                    f"({ensemble.reward_scale}) != return model scale "
                    f"({self.return_model.scale})"
                )
            if (self.return_model.horizon is not None
                    and self.return_model.horizon != self.horizon):
                raise ValueError(
                    f"Return model horizon ({self.return_model.horizon}) != controller "
                    f"horizon ({self.horizon})"
                )

        act_dim = dt_adapter.act_dim
        self.action_low = np.asarray(action_low, dtype=np.float32) if action_low is not None else np.full(act_dim, -1.0)
        self.action_high = np.asarray(action_high, dtype=np.float32) if action_high is not None else np.full(act_dim, 1.0)
        if self.action_low.shape != (act_dim,) or self.action_high.shape != (act_dim,):
            raise ValueError(
                f"action_low/high must have shape ({act_dim},), got "
                f"{self.action_low.shape}/{self.action_high.shape}"
            )
        if not np.isfinite(self.action_low).all() or not np.isfinite(self.action_high).all():
            raise ValueError("action bounds must be finite")
        if np.any(self.action_low >= self.action_high):
            raise ValueError("action_low must be strictly below action_high")
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """Reset the internal DT adapter context for a new episode."""
        self.dt_adapter.reset()

    @property
    def requires_proposal_rng(self) -> bool:
        """Whether this controller requires candidate-proposal RNG injection."""
        return self.num_candidates > 1

    def policy_fn(
        self,
        tokens_used: list[float],
        obs: np.ndarray,
        step: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Policy callback compatible with evaluate_with_budget."""
        rtg_token = float(tokens_used[-1]) if tokens_used else 0.0
        return self.get_action(obs=obs, rtg_token=rtg_token, step=step)

    def set_rng(self, rng: np.random.Generator) -> None:
        """Inject the evaluation harness candidate-proposal stream."""
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy Generator")
        self.rng = rng

    def get_action(
        self,
        obs: np.ndarray,
        rtg_token: float,
        step: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Select action via DT proposal and optional Module U reranking.

        Args:
            obs: Raw environment observation.
            rtg_token: Clipped or unclipped conditioning token.
            step: Current episode timestep t.

        Returns:
            (selected_action, info_dict).
        """
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.shape != (self.ensemble.state_dim,):
            raise ValueError(
                f"obs must have shape ({self.ensemble.state_dim},), got {obs_arr.shape}"
            )
        if not np.isfinite(obs_arr).all():
            raise ValueError("obs contains non-finite values")
        if not np.isfinite(rtg_token):
            raise ValueError(f"rtg_token must be finite, got {rtg_token}")
        if not isinstance(step, (int, np.integer)) or step < 0:
            raise ValueError(f"step must be a nonnegative integer, got {step}")

        # Step 1: Query frozen Decision Transformer for deterministic baseline action a_{0, t}
        a_0 = self.dt_adapter.step(obs=obs_arr, rtg_token=rtg_token, step=step)

        # If N == 1, bypass reranking entirely and return a_{0, t}
        if self.num_candidates <= 1:
            self.dt_adapter.record_commanded_action(a_0)
            return a_0, {
                "selected_index": 0,
                "candidate_count": 1,
                "original_action": a_0,
                "candidate_actions": np.array([a_0]),
                "at_horizon": step + 1 >= self.horizon,
                "scorer_mode": self.scorer_mode,
            }

        # Step 2: Propose N candidate actions around a_{0, t}
        candidates = propose_actions(
            action=a_0,
            low=self.action_low,
            high=self.action_high,
            count=self.num_candidates,
            radius=self.radius,
            rng=self.rng,
        )

        # Step 3: Ensemble dynamics predictions in dynamics-normalized coordinates
        norm_obs = self.ensemble.normalize_obs(obs_arr)
        t_rel = min(1.0, max(0.0, float(step) / float(self.horizon)))
        next_states, rewards, terminals = self.ensemble.predict(norm_obs, candidates, t_rel)
        # next_states: (M, N, state_dim), rewards: (M, N), terminals: (M, N)

        # Step 4: Evaluate next return values v_m(s'_m) per head and candidate.
        at_horizon = (step + 1 >= self.horizon)
        m, n = rewards.shape
        next_returns = np.zeros((m, n), dtype=np.float32)

        if not at_horizon and self.scorer_mode == "return_model":
            # Dynamics-normalized states are converted to raw observation units
            # with the dynamics statistics, then the return model applies its own
            # separate normalization exactly once. Batching never replaces the
            # mean of per-head values with a value at the mean state.
            raw_next_states = self.ensemble.unnormalize_state(next_states)
            flat_states = raw_next_states.reshape(m * n, self.ensemble.state_dim)
            flat_returns = self.return_model.get_cap_batch(flat_states, step + 1)
            next_returns = np.asarray(flat_returns, dtype=np.float32).reshape(m, n)

        # Step 5: Score candidates with disagreement penalty
        scores = ensemble_scores(
            rewards=rewards,
            next_returns=next_returns,
            terminal_probabilities=terminals,
            next_states=next_states,
            beta=self.beta,
            at_horizon=at_horizon,
        )

        # Step 6: Select candidate with highest score (tie-breaking prefers lowest index, i.e. a_0)
        selected_idx = int(np.argmax(scores))
        selected_action = candidates[selected_idx]

        # Record commanded action into DT context history
        self.dt_adapter.record_commanded_action(selected_action)

        # Calculate disagreement for telemetry
        centered = next_states - next_states.mean(axis=0)
        disagreement = np.sqrt(np.square(centered).sum(axis=-1).mean(axis=0))

        info = {
            "selected_index": selected_idx,
            "candidate_count": self.num_candidates,
            "original_action": a_0,
            "candidate_actions": candidates.copy(),
            "candidate_scores": scores,
            "disagreements": disagreement,
            "at_horizon": at_horizon,
            "scorer_mode": self.scorer_mode,
            "scaled_rewards": rewards,
            "terminal_probabilities": terminals,
            "next_returns": next_returns,
        }
        return selected_action, info
