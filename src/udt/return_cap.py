"""Module R: Return-to-Go Calibration and Capping for Frozen Decision Transformers.

Provides:
- ExpectileLoss: Asymmetric squared loss for upper-expectile regression.
- compute_global_expectile: Scalar expectile root-finding for Baseline B2.
- GlobalReturnCap: Constant scalar return cap (Baseline B2).
- ReturnCapMLP: Neural state-time return predictor (Baseline B3, B9, and ablation).
- ReturnCapper: Inference-time intervention interface hooking into ReturnBudget.
- train_return_cap: Complete training pipeline for expectile return models.
"""

from typing import Any, Literal
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from udt.intervention import ReturnBudget


class ExpectileLoss(nn.Module):
    """Asymmetric squared loss for expectile regression.

    Given target return y and prediction \\hat{y}, the error is e = y - \\hat{y}.
    The loss is defined as:
        L_\\tau(\\hat{y}, y) = |\\tau - I(y - \\hat{y} < 0)| * (y - \\hat{y})^2

    For \\tau > 0.5 (e.g. \\tau = 0.9):
    - Under-estimation (y > \\hat{y}, e > 0): penalized with weight \\tau (0.9).
    - Over-estimation (y < \\hat{y}, e < 0): penalized with weight 1 - \\tau (0.1).
    """

    def __init__(self, tau: float = 0.9, reduction: Literal["mean", "sum", "none"] = "mean"):
        super().__init__()
        if not (0.0 < tau < 1.0):
            raise ValueError(f"tau must be in the open interval (0, 1), got {tau}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction: {reduction}. Expected 'mean', 'sum', or 'none'.")
        self.tau = float(tau)
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the expectile loss.

        Args:
            input: Predicted return tensor (\\hat{y}).
            target: Ground truth return tensor (y).

        Returns:
            Computed loss according to the specified reduction.
        """
        if input.shape != target.shape:
            if input.ndim == target.ndim + 1 and input.shape[-1] == 1:
                target = target.unsqueeze(-1)
            elif target.ndim == input.ndim + 1 and target.shape[-1] == 1:
                target = target.squeeze(-1)
            else:
                raise ValueError(f"Shape mismatch in ExpectileLoss: input {input.shape} vs target {target.shape}")

        diff = target - input

        # Indicator is 1 when target < input (over-estimation), 0 otherwise
        weight = torch.abs(self.tau - (diff < 0.0).float())
        loss = weight * (diff ** 2)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_global_expectile(returns: np.ndarray, tau: float = 0.9, max_iter: int = 100, tol: float = 1e-6) -> float:
    """Compute the scalar expectile of a dataset using Iteratively Reweighted Least Squares (IRLS).

    Solves the first-order optimality condition:
        E[ |\\tau - I(Y < c)| * (Y - c) ] = 0

    Args:
        returns: 1D array of return values.
        tau: Expectile level in (0, 1).
        max_iter: Maximum number of IRLS iterations.
        tol: Convergence tolerance for the expectile estimate.

    Returns:
        Converged scalar expectile value.
    """
    if not (0.0 < tau < 1.0):
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    arr = np.asarray(returns, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("Cannot compute expectile on an empty array")

    c = float(np.mean(arr))
    for _ in range(max_iter):
        weights = np.where(arr < c, 1.0 - tau, tau)
        sum_weights = np.sum(weights)
        if sum_weights == 0.0:
            break
        c_next = float(np.sum(weights * arr) / sum_weights)
        if abs(c_next - c) < tol:
            return c_next
        c = c_next
    return c


class GlobalReturnCap:
    """Baseline B2: Uniform global scalar return cap.

    Applies the same precomputed scalar expectile across all states and timesteps.
    """

    def __init__(self, cap_value: float):
        self.cap_value = float(cap_value)

    def get_cap(self, state: np.ndarray | torch.Tensor, step: int = 0, horizon: int | None = None) -> float:
        """Return the constant global cap."""
        return self.cap_value


class ReturnCapMLP(nn.Module):
    """Multi-Layer Perceptron for upper-expectile return prediction g_psi(z_t, t/H).

    Supports three configuration modes to isolate baseline comparisons:
    - 'state_time' (Baseline B3): Inputs concatenated (z_t, t/H), dimension = state_dim + 1.
    - 'time_only' (Baseline B9): Input only (t/H), dimension = 1.
    - 'state_only' (Ablation): Input only (z_t), dimension = state_dim.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        mode: Literal["state_time", "time_only", "state_only"] = "state_time",
        activation: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        if mode not in ("state_time", "time_only", "state_only"):
            raise ValueError(f"Unknown mode: {mode}. Must be 'state_time', 'time_only', or 'state_only'.")
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.mode = mode

        # Determine input dimension based on mode
        if mode == "state_time":
            in_dim = state_dim + 1
        elif mode == "time_only":
            in_dim = 1
        else:  # state_only
            in_dim = state_dim

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), activation()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation()])
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        states: torch.Tensor | np.ndarray | None = None,
        t_rel: torch.Tensor | np.ndarray | float | None = None,
    ) -> torch.Tensor:
        """Forward pass predicting scaled return expectile.

        Args:
            states: Observation tensor of shape (batch, state_dim) or (state_dim,).
            t_rel: Relative timestep t/H in [0, 1], shape (batch, 1), (batch,), or scalar.

        Returns:
            Scalar return predictions of shape (batch, 1).
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        if self.mode == "state_time":
            if states is None or t_rel is None:
                raise ValueError("Both states and t_rel are required in 'state_time' mode")
            s = torch.as_tensor(states, device=device, dtype=dtype)
            t = torch.as_tensor(t_rel, device=device, dtype=dtype)
            if s.ndim == 1:
                s = s.unsqueeze(0)
            if t.ndim == 0:
                t = t.view(1, 1)
            elif t.ndim == 1:
                t = t.unsqueeze(-1)
            if s.shape[0] > 1 and t.shape[0] == 1:
                t = t.expand(s.shape[0], -1)
            x = torch.cat([s, t], dim=-1)


        elif self.mode == "time_only":
            if t_rel is None:
                raise ValueError("t_rel is required in 'time_only' mode")
            t = torch.as_tensor(t_rel, device=device, dtype=dtype)
            if t.ndim == 0:
                t = t.view(1, 1)
            elif t.ndim == 1:
                t = t.unsqueeze(-1)
            x = t

        else:  # state_only
            if states is None:
                raise ValueError("states is required in 'state_only' mode")
            s = torch.as_tensor(states, device=device, dtype=dtype)
            if s.ndim == 1:
                s = s.unsqueeze(0)
            x = s

        return self.network(x)


class ReturnCapper:
    """Inference-time return-to-go capping interface (Module R).

    Evaluates the return upper bound C_t = g_psi(z_t, t/H) + delta and applies
    clipping to the raw return budget B_t = min(B_t^raw, C_t).
    Critically, preserves the raw budget recurrence B_{t+1}^raw = B_t^raw - r_t/c.
    """

    def __init__(
        self,
        model: nn.Module | GlobalReturnCap,
        horizon: int,
        margin: float = 0.0,
        scale: float = 1000.0,
        state_mean: np.ndarray | None = None,
        state_std: np.ndarray | None = None,
        device: str = "cpu",
    ):
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        self.model = model
        self.horizon = int(horizon)
        self.margin = float(margin)
        self.scale = float(scale)
        self.device = torch.device(device)

        self.state_mean = np.asarray(state_mean, dtype=np.float32) if state_mean is not None else None
        self.state_std = np.asarray(state_std, dtype=np.float32) if state_std is not None else None

        if isinstance(self.model, nn.Module):
            self.model.to(self.device)
            self.model.eval()

    def get_cap(self, state: np.ndarray | torch.Tensor, step: int) -> float:
        """Compute the scalar return cap for a given state and timestep.

        Normalizes raw state observations if mean and std statistics were configured.

        Args:
            state: Observation vector.
            step: Current episode timestep t >= 0.

        Returns:
            Scalar return cap value.
        """
        if isinstance(self.model, GlobalReturnCap):
            return self.model.get_cap(state, step, self.horizon)

        # Safely convert observation to numpy float32 regardless of torch device/grad status
        if isinstance(state, torch.Tensor):
            s_arr = state.detach().cpu().numpy().astype(np.float32)
        else:
            s_arr = np.asarray(state, dtype=np.float32)

        # Normalize observation if statistics are registered
        if self.state_mean is not None and self.state_std is not None:
            s_arr = (s_arr - self.state_mean) / self.state_std

        # Compute relative elapsed timestep t / H clamped to [0.0, 1.0]
        t_rel = min(1.0, max(0.0, float(step) / float(self.horizon)))

        with torch.no_grad():
            s_tensor = torch.as_tensor(s_arr, device=self.device, dtype=torch.float32)
            t_tensor = torch.tensor([[t_rel]], device=self.device, dtype=torch.float32)
            out = self.model(s_tensor, t_tensor)
            return float(out.item())


    def step(self, budget: ReturnBudget, state: np.ndarray | torch.Tensor, step: int) -> float:
        """Compute cap and return the clipped token for DT input.

        Leaves budget.raw unclipped for accurate reward subtraction!

        Args:
            budget: Active ReturnBudget tracking raw return.
            state: Current environment observation.
            step: Current episode timestep.

        Returns:
            The clipped token B_t to be fed into Decision Transformer.
        """
        cap = self.get_cap(state, step)
        return budget.token(cap=cap, margin=self.margin)


def train_return_cap(
    dataset: dict[str, np.ndarray],
    horizon: int,
    tau: float = 0.9,
    mode: Literal["state_time", "time_only", "state_only", "global"] = "state_time",
    state_mean: np.ndarray | None = None,
    state_std: np.ndarray | None = None,
    margin: float = 0.0,
    scale: float = 1000.0,
    hidden_dim: int = 128,
    num_layers: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 15,
    batch_size: int = 256,
    device: str = "cpu",
    patience: int = 5,
) -> tuple[ReturnCapper, dict[str, Any]]:
    """Train the expectile return model g_psi on an audited offline dataset.

    Args:
        dataset: Prepared dataset containing 'observations', 'returns_to_go',
                 'timesteps', and 'is_validation' (e.g. from udt.data.prepare_dataset).
        horizon: Task episode horizon H.
        tau: Expectile asymmetry parameter in (0, 1).
        mode: Return model type ('state_time', 'time_only', 'state_only', 'global').
        state_mean: Precomputed observation mean for normalization (optional).
        state_std: Precomputed observation std for normalization (optional).
        margin: Default delta margin added to predicted cap at inference.
        scale: Return scaling constant c (e.g. 1000 for D4RL).
        hidden_dim: Hidden dimension of MLP layers.
        num_layers: Number of MLP layers.
        lr: Adam learning rate.
        weight_decay: L2 regularization coefficient.
        epochs: Maximum training epochs.
        batch_size: Batch size for training.
        device: Torch compute device string ('cpu' or 'cuda').
        patience: Epochs without validation improvement before early stopping.

    Returns:
        A tuple of (ReturnCapper, metrics_dict).
    """
    for required_key in ("observations", "returns_to_go", "timesteps", "is_validation"):
        if required_key not in dataset:
            raise ValueError(f"Dataset missing required key: {required_key}")

    train_mask = ~dataset["is_validation"]
    val_mask = dataset["is_validation"]

    if np.sum(train_mask) == 0 or np.sum(val_mask) == 0:
        raise ValueError("Dataset must contain both training and validation samples")

    # Handle Baseline B2: Uniform global scalar expectile
    if mode == "global":
        train_rtgs = dataset["returns_to_go"][train_mask]
        cap_val = compute_global_expectile(train_rtgs, tau=tau)
        model = GlobalReturnCap(cap_value=cap_val)
        val_rtgs = dataset["returns_to_go"][val_mask]
        val_cov = float(np.mean(val_rtgs <= cap_val + margin))
        capper = ReturnCapper(
            model=model,
            horizon=horizon,
            margin=margin,
            scale=scale,
            device=device,
        )
        metrics = {
            "cap_value": cap_val,
            "train_loss": [0.0],
            "val_loss": [0.0],
            "val_coverage": [val_cov],
        }
        return capper, metrics

    # Neural baseline modes (B3: state_time, B9: time_only, ablation: state_only)
    state_dim = dataset["observations"].shape[1]

    # Compute normalization statistics from training trajectories if not provided
    if state_mean is None or state_std is None:
        train_obs_raw = dataset["observations"][train_mask]
        state_mean = np.mean(train_obs_raw, axis=0, dtype=np.float32)
        state_std = np.maximum(np.std(train_obs_raw, axis=0, dtype=np.float32), 1e-6)

    # Normalize observations
    train_obs = ((dataset["observations"][train_mask] - state_mean) / state_std).astype(np.float32)
    val_obs = ((dataset["observations"][val_mask] - state_mean) / state_std).astype(np.float32)

    # Timestep normalized fraction t / H in [0, 1]
    train_t_rel = np.clip(dataset["timesteps"][train_mask].astype(np.float32) / float(horizon), 0.0, 1.0)[:, None]
    val_t_rel = np.clip(dataset["timesteps"][val_mask].astype(np.float32) / float(horizon), 0.0, 1.0)[:, None]

    # Target returns
    train_y = dataset["returns_to_go"][train_mask].astype(np.float32)[:, None]
    val_y = dataset["returns_to_go"][val_mask].astype(np.float32)[:, None]

    # Build PyTorch DataLoaders
    train_ds = TensorDataset(
        torch.from_numpy(train_obs),
        torch.from_numpy(train_t_rel),
        torch.from_numpy(train_y),
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_obs),
        torch.from_numpy(val_t_rel),
        torch.from_numpy(val_y),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    torch_device = torch.device(device)
    model = ReturnCapMLP(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        mode=mode,
    ).to(torch_device)

    criterion = ExpectileLoss(tau=tau, reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss_hist: list[float] = []
    val_loss_hist: list[float] = []
    val_cov_hist: list[float] = []

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    stagnant_epochs = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        train_batches = 0
        for b_obs, b_t, b_y in train_loader:
            b_obs = b_obs.to(torch_device)
            b_t = b_t.to(torch_device)
            b_y = b_y.to(torch_device)

            optimizer.zero_grad()
            preds = model(b_obs, b_t)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

        avg_train_loss = total_train_loss / max(1, train_batches)
        train_loss_hist.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        covered_count = 0
        total_samples = 0

        with torch.no_grad():
            for b_obs, b_t, b_y in val_loader:
                b_obs = b_obs.to(torch_device)
                b_t = b_t.to(torch_device)
                b_y = b_y.to(torch_device)

                preds = model(b_obs, b_t)
                loss = criterion(preds, b_y)
                total_val_loss += loss.item()
                val_batches += 1

                # Empirical quantile coverage: proportion where ground truth return <= cap + margin
                covered = (b_y <= preds + margin).sum().item()
                covered_count += covered
                total_samples += b_y.numel()

        avg_val_loss = total_val_loss / max(1, val_batches)
        val_coverage = float(covered_count) / max(1, total_samples)

        val_loss_hist.append(avg_val_loss)
        val_cov_hist.append(val_coverage)

        # Early stopping tracking
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1
            if stagnant_epochs >= patience:
                break

    # Restore best checkpoint
    model.load_state_dict(best_weights)
    model.eval()

    capper = ReturnCapper(
        model=model,
        horizon=horizon,
        margin=margin,
        scale=scale,
        state_mean=state_mean,
        state_std=state_std,
        device=device,
    )
    metrics = {
        "train_loss": train_loss_hist,
        "val_loss": val_loss_hist,
        "val_coverage": val_cov_hist,
        "best_val_loss": best_val_loss,
    }
    return capper, metrics
