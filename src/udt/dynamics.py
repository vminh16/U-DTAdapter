"""Module U: Ensemble Dynamics and Disagreement-Penalized Action Reranking.

Implements RRD Sections 2.2, 2.3, and 2.4:
- M independent MLP dynamics models predicting delta state, reward, and terminal probability.
- Ensemble disagreement calculation.
- Local candidate generation with original action inclusion.
- Action reranking with disagreement penalty and horizon-aware continuation.
"""

import copy
from typing import Any, Literal
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
    ):
        if num_models < 1:
            raise ValueError(f"num_models must be >= 1, got {num_models}")

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.num_models = num_models
        self.device = torch.device(device)

        self.models: list[DynamicsMLP] = [
            DynamicsMLP(
                state_dim=state_dim,
                act_dim=act_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            ).to(self.device)
            for _ in range(num_models)
        ]
        for m in self.models:
            m.eval()

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
    for req in ("observations", "next_observations", "actions", "rewards", "terminals", "timesteps", "is_validation"):
        if req not in dataset:
            raise ValueError(f"Dataset missing required key: {req}")

    n_samples = len(dataset["observations"])
    if n_samples == 0:
        raise ValueError("Dataset is empty")

    for k in ("next_observations", "actions", "rewards", "terminals", "timesteps", "is_validation"):
        if len(dataset[k]) != n_samples:
            raise ValueError(f"{k} length mismatch: {len(dataset[k])} != {n_samples}")

    is_val = dataset["is_validation"]
    if not (isinstance(is_val, np.ndarray) and np.issubdtype(is_val.dtype, np.bool_)):
        raise ValueError("is_validation must be boolean numpy array")

    train_mask = ~is_val
    val_mask = is_val
    if np.sum(train_mask) == 0 or np.sum(val_mask) == 0:
        raise ValueError("Dataset must contain both training and validation samples")

    state_dim = dataset["observations"].shape[1]
    act_dim = dataset["actions"].shape[1]

    if state_mean is None or state_std is None:
        train_obs_raw = dataset["observations"][train_mask]
        state_mean = np.mean(train_obs_raw, axis=0, dtype=np.float32)
        state_std = np.maximum(np.std(train_obs_raw, axis=0, dtype=np.float32), 1e-6)

    # Normalize observations and compute delta targets
    norm_obs = ((dataset["observations"] - state_mean) / state_std).astype(np.float32)
    norm_next_obs = ((dataset["next_observations"] - state_mean) / state_std).astype(np.float32)
    delta_z = (norm_next_obs - norm_obs).astype(np.float32)

    actions = dataset["actions"].astype(np.float32)
    scaled_rewards = (dataset["rewards"] / scale).astype(np.float32)[:, None]
    terminals = dataset["terminals"].astype(np.float32)[:, None]
    t_rel = np.clip(dataset["timesteps"].astype(np.float32) / float(horizon), 0.0, 1.0)[:, None]

    torch_device = torch.device(device)
    ensemble = DynamicsEnsemble(
        state_dim=state_dim,
        act_dim=act_dim,
        num_models=num_models,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=torch_device,
    )

    ensemble_metrics: list[dict[str, Any]] = []

    # Train each model independently
    for m_idx in range(num_models):
        head_seed = (seed + m_idx * 1000) if seed is not None else None
        if head_seed is not None:
            torch.manual_seed(head_seed)
            np.random.seed(head_seed)

        model = ensemble.models[m_idx]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Datasets
        train_ds = TensorDataset(
            torch.from_numpy(norm_obs[train_mask]),
            torch.from_numpy(actions[train_mask]),
            torch.from_numpy(t_rel[train_mask]),
            torch.from_numpy(delta_z[train_mask]),
            torch.from_numpy(scaled_rewards[train_mask]),
            torch.from_numpy(terminals[train_mask]),
        )
        val_ds = TensorDataset(
            torch.from_numpy(norm_obs[val_mask]),
            torch.from_numpy(actions[val_mask]),
            torch.from_numpy(t_rel[val_mask]),
            torch.from_numpy(delta_z[val_mask]),
            torch.from_numpy(scaled_rewards[val_mask]),
            torch.from_numpy(terminals[val_mask]),
        )

        train_gen = torch.Generator()
        if head_seed is not None:
            train_gen.manual_seed(head_seed)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=train_gen)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        mse_fn = nn.MSELoss(reduction="sum")
        bce_fn = nn.BCELoss(reduction="sum")

        best_val_loss = float("inf")
        best_weights = copy.deepcopy(model.state_dict())
        best_epoch = 0
        stagnant = 0
        train_loss_hist = []
        val_loss_hist = []

        for epoch in range(epochs):
            model.train()
            total_train_loss = 0.0
            total_train_samples = 0

            for b_z, b_a, b_t, b_dz, b_r, b_term in train_loader:
                b_z, b_a, b_t = b_z.to(torch_device), b_a.to(torch_device), b_t.to(torch_device)
                b_dz, b_r, b_term = b_dz.to(torch_device), b_r.to(torch_device), b_term.to(torch_device)

                optimizer.zero_grad()
                pred_dz, pred_r, pred_term = model(b_z, b_a, b_t)

                loss_dz = mse_fn(pred_dz, b_dz)
                loss_r = mse_fn(pred_r, b_r)
                loss_term = bce_fn(pred_term, b_term)
                batch_loss = (loss_dz + w_reward * loss_r + w_terminal * loss_term) / b_z.size(0)

                batch_loss.backward()
                optimizer.step()

                total_train_loss += (loss_dz.item() + w_reward * loss_r.item() + w_terminal * loss_term.item())
                total_train_samples += b_z.size(0)

            epoch_train_loss = total_train_loss / max(1, total_train_samples)
            train_loss_hist.append(epoch_train_loss)

            # Validation
            model.eval()
            total_val_loss = 0.0
            total_val_samples = 0

            with torch.no_grad():
                for b_z, b_a, b_t, b_dz, b_r, b_term in val_loader:
                    b_z, b_a, b_t = b_z.to(torch_device), b_a.to(torch_device), b_t.to(torch_device)
                    b_dz, b_r, b_term = b_dz.to(torch_device), b_r.to(torch_device), b_term.to(torch_device)

                    pred_dz, pred_r, pred_term = model(b_z, b_a, b_t)
                    loss_dz = mse_fn(pred_dz, b_dz)
                    loss_r = mse_fn(pred_r, b_r)
                    loss_term = bce_fn(pred_term, b_term)

                    total_val_loss += (loss_dz.item() + w_reward * loss_r.item() + w_terminal * loss_term.item())
                    total_val_samples += b_z.size(0)

            epoch_val_loss = total_val_loss / max(1, total_val_samples)
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
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        })

    return ensemble, {"models": ensemble_metrics}


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
        capper: ReturnCapper | None = None,
        num_candidates: int = 10,
        radius: float = 0.05,
        beta: float = 0.1,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        horizon: int = 1000,
        seed: int | None = None,
    ):
        self.dt_adapter = dt_adapter
        self.ensemble = ensemble
        self.capper = capper
        self.num_candidates = int(num_candidates)
        self.radius = float(radius)
        self.beta = float(beta)
        self.horizon = int(horizon)

        act_dim = dt_adapter.act_dim
        self.action_low = np.asarray(action_low, dtype=np.float32) if action_low is not None else np.full(act_dim, -1.0)
        self.action_high = np.asarray(action_high, dtype=np.float32) if action_high is not None else np.full(act_dim, 1.0)
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """Reset the internal DT adapter context for a new episode."""
        self.dt_adapter.reset()

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
        # Step 1: Query frozen Decision Transformer for deterministic baseline action a_{0, t}
        a_0 = self.dt_adapter.step(obs=obs, rtg_token=rtg_token, step=step)

        # If N == 1, bypass reranking entirely and return a_{0, t}
        if self.num_candidates <= 1:
            self.dt_adapter.record_commanded_action(a_0)
            return a_0, {
                "selected_index": 0,
                "candidate_count": 1,
                "original_action": a_0,
                "at_horizon": step + 1 >= self.horizon,
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

        # Step 3: Ensemble dynamics predictions
        norm_obs = self.dt_adapter.normalize_obs(obs)
        t_rel = min(1.0, max(0.0, float(step) / float(self.horizon)))
        next_states, rewards, terminals = self.ensemble.predict(norm_obs, candidates, t_rel)
        # next_states: (M, N, state_dim), rewards: (M, N), terminals: (M, N)

        # Step 4: Evaluate next return values v_m(s')
        at_horizon = (step + 1 >= self.horizon)
        m, n = rewards.shape
        next_returns = np.zeros((m, n), dtype=np.float32)

        if not at_horizon and self.capper is not None:
            # Query return model for each head's predicted next state
            # Note: next_states are already normalized, so we evaluate capper with raw inputs
            for m_i in range(m):
                for n_i in range(n):
                    # Unnormalize next_state if capper performs its own normalization
                    ns = next_states[m_i, n_i]
                    if self.dt_adapter.state_mean is not None and self.dt_adapter.state_std is not None:
                        ns_raw = ns * self.dt_adapter.state_std + self.dt_adapter.state_mean
                    else:
                        ns_raw = ns
                    next_returns[m_i, n_i] = self.capper.get_cap(ns_raw, step + 1)

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
            "candidate_scores": scores,
            "disagreements": disagreement,
            "at_horizon": at_horizon,
        }
        return selected_action, info
