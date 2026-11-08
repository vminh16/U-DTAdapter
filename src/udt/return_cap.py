"""Module R: Return-to-Go Calibration and Capping for Frozen Decision Transformers.

Provides:
- ExpectileLoss: Asymmetric squared loss for upper-expectile regression.
- compute_global_expectile: Scalar expectile root-finding for Baseline B2.
- GlobalReturnCap: Constant scalar return cap (Baseline B2).
- ReturnCapMLP: Neural state-time return predictor (Baseline B3, B9, and ablation).
- ReturnCapper: Inference-time intervention interface hooking into ReturnBudget.
- train_return_cap: Complete training pipeline for expectile return models.
"""

from pathlib import Path
from typing import Any, Literal
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from udt.checkpoint import load_checkpoint_payload
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
        (1/n) * sum_i |tau - I(Y_i < c)| * (Y_i - c) = 0

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


def compute_expectile_foc_residual(returns: np.ndarray, expectile: float, tau: float) -> float:
    """Compute empirical first-order optimality condition (FOC) residual:

        (1/n) * sum_i |tau - I(Y_i < e)| * (Y_i - e)

    For the true minimizer of the expectile loss, this residual must be approximately 0.
    Expectiles are asymmetric L2 minimizers and do NOT guarantee quantile coverage.
    """
    arr = np.asarray(returns, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("Cannot evaluate FOC residual on an empty array")
    weights = np.where(arr < expectile, 1.0 - tau, tau)
    return float(np.mean(weights * (arr - expectile)))


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


    def get_cap_batch(
        self,
        states: np.ndarray | torch.Tensor,
        step: int | np.ndarray,
    ) -> np.ndarray:
        """Evaluate the return model on a batch of raw observations.

        Args:
            states: Raw observation array of shape (K, state_dim).
            step: Timestep shared by every state in the batch, or an integer
                array of shape (K,) with one timestep per state.

        Returns:
            Scaled return predictions of shape (K,).
        """
        if isinstance(self.model, GlobalReturnCap):
            batch = np.asarray(states)
            if batch.ndim != 2 or batch.shape[0] < 1 or batch.shape[1] < 1:
                raise ValueError(f"states must have shape (K, state_dim), got {batch.shape}")
            return np.full(batch.shape[0], self.model.cap_value, dtype=np.float32)

        if isinstance(states, torch.Tensor):
            s_arr = states.detach().cpu().numpy().astype(np.float32)
        else:
            s_arr = np.asarray(states, dtype=np.float32)
        if s_arr.ndim != 2 or s_arr.shape[0] < 1 or s_arr.shape[1] < 1:
            raise ValueError(f"states must have shape (K, state_dim), got {s_arr.shape}")
        if not np.isfinite(s_arr).all():
            raise ValueError("states contains non-finite values")
        if self.state_mean is not None and self.state_std is not None:
            if s_arr.shape[1] != self.state_mean.shape[0]:
                raise ValueError(
                    f"state dimension {s_arr.shape[1]} does not match return-model "
                    f"normalization dimension {self.state_mean.shape[0]}"
                )
            s_arr = (s_arr - self.state_mean) / self.state_std

        steps = np.asarray(step)
        if steps.ndim == 0:
            t_rel = np.full(
                s_arr.shape[0], min(1.0, max(0.0, float(steps) / float(self.horizon))),
                dtype=np.float32,
            )
        else:
            if steps.shape != (s_arr.shape[0],):
                raise ValueError(
                    f"step array must have shape ({s_arr.shape[0]},), got {steps.shape}"
                )
            if not np.isfinite(steps).all():
                raise ValueError("step array contains non-finite values")
            t_rel = np.clip(steps.astype(np.float32) / float(self.horizon), 0.0, 1.0)
        with torch.no_grad():
            s_tensor = torch.as_tensor(s_arr, device=self.device, dtype=torch.float32)
            t_tensor = torch.as_tensor(
                t_rel[:, None], device=self.device, dtype=torch.float32
            )
            out = self.model(s_tensor, t_tensor)
            return out.detach().cpu().numpy().astype(np.float32).reshape(-1)

    def step(self, budget: ReturnBudget, state: np.ndarray | torch.Tensor, step: int,
             alpha: float = 1.0) -> float:
        """Compute cap and return the clipped token for DT input.

        Leaves budget.raw unclipped for accurate reward subtraction!

        Args:
            budget: Active ReturnBudget tracking raw return.
            state: Current environment observation.
            step: Current episode timestep.
            alpha: Intervention strength in [0, 1]. 1.0 = hard cap, 0.0 = no cap.

        Returns:
            The clipped token B_t to be fed into Decision Transformer.

        Raises:
            ValueError: If budget.reward_scale != self.scale.
        """
        if not np.isclose(budget.reward_scale, self.scale):
            raise ValueError(
                f"Scale mismatch: ReturnBudget scale ({budget.reward_scale}) != ReturnCapper scale ({self.scale})"
            )
        cap = self.get_cap(state, step)
        return budget.token(cap=cap, margin=self.margin, alpha=alpha)


def evaluate_return_capper(
    dataset: dict[str, np.ndarray],
    capper: ReturnCapper,
    *,
    horizon: int,
    margin: float = 0.0,
    tau: float = 0.9,
    split: str = "validation",
    batch_size: int = 8192,
) -> dict[str, float]:
    """Independently evaluate a restored capper on one dataset split.

    This uses the same preprocessing stored with the capper, so it can detect a
    mismatch between best-checkpoint telemetry and the returned model.
    """
    for key in ("observations", "returns_to_go", "timesteps", "is_validation"):
        if key not in dataset:
            raise ValueError(f"Dataset missing required key: {key}")
    if capper.horizon != int(horizon):
        raise ValueError(
            f"Capper horizon ({capper.horizon}) does not match evaluation horizon ({horizon})"
        )
    is_val = np.asarray(dataset["is_validation"])
    if not np.issubdtype(is_val.dtype, np.bool_):
        raise ValueError("is_validation must be boolean")
    if split == "validation":
        mask = is_val
    elif split == "train":
        mask = ~is_val
    else:
        raise ValueError(f"split must be 'train' or 'validation', got {split}")
    if not np.any(mask):
        raise ValueError(f"Dataset split '{split}' is empty")

    obs = np.asarray(dataset["observations"])[mask]
    returns = np.asarray(dataset["returns_to_go"])[mask].astype(np.float64)
    timesteps = np.asarray(dataset["timesteps"])[mask]
    n = len(obs)
    loss_sum = 0.0
    covered = 0
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        preds = capper.get_cap_batch(obs[start:stop], timesteps[start:stop])
        batch_y = returns[start:stop]
        diff = batch_y - preds
        weights = np.where(diff < 0.0, 1.0 - tau, tau)
        loss_sum += float(np.sum(weights * (diff ** 2)))
        covered += int(np.sum(batch_y <= preds + margin))
    return {
        "samples": int(n),
        "loss": loss_sum / n,
        "coverage": covered / n,
    }


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
    seed: int | None = None,
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
        seed: Optional random seed for reproducible training.

    Returns:
        A tuple of (ReturnCapper, metrics_dict).
    """
    for required_key in ("observations", "returns_to_go", "timesteps", "is_validation"):
        if required_key not in dataset:
            raise ValueError(f"Dataset missing required key: {required_key}")

    n_samples = len(dataset["observations"])
    if n_samples == 0:
        raise ValueError("Dataset is empty (0 transitions)")

    for key in ("returns_to_go", "timesteps", "is_validation"):
        if len(dataset[key]) != n_samples:
            raise ValueError(
                f"{key} transition count mismatch: {len(dataset[key])} != {n_samples}"
            )

    is_val = dataset["is_validation"]
    if not (isinstance(is_val, np.ndarray) and np.issubdtype(is_val.dtype, np.bool_)):
        raise ValueError(
            f"is_validation must be a boolean numpy array, got dtype {getattr(is_val, 'dtype', type(is_val))}"
        )

    if not np.all(np.isfinite(dataset["observations"])):
        raise ValueError("observations contains non-finite values (NaN or Inf)")
    if not np.all(np.isfinite(dataset["returns_to_go"])):
        raise ValueError("returns_to_go contains non-finite values (NaN or Inf)")
    if not np.all(np.isfinite(dataset["timesteps"])):
        raise ValueError("timesteps contains non-finite values (NaN or Inf)")

    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError(f"scale must be positive and finite, got {scale}")

    t_arr = dataset["timesteps"]
    if np.any(t_arr < 0) or np.any(t_arr >= horizon):
        raise ValueError(
            f"timesteps out of domain [0, {horizon - 1}]: min={np.min(t_arr)}, max={np.max(t_arr)}"
        )

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

    train_mask = ~dataset["is_validation"]
    val_mask = dataset["is_validation"]

    if np.sum(train_mask) == 0:
        raise ValueError("Dataset must contain training samples")
    if np.sum(val_mask) == 0:
        raise ValueError("Dataset must contain validation samples")

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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
        # Compute true expectile loss for train and val
        train_diff = train_rtgs - cap_val
        train_weights = np.where(train_diff < 0.0, 1.0 - tau, tau)
        train_loss = float(np.mean(train_weights * (train_diff ** 2)))

        val_diff = val_rtgs - cap_val
        val_weights = np.where(val_diff < 0.0, 1.0 - tau, tau)
        val_loss = float(np.mean(val_weights * (val_diff ** 2)))

        metrics = {
            "cap_value": cap_val,
            "train_loss": [train_loss],
            "val_loss": [val_loss],
            "val_coverage": [val_cov],
            "best_val_loss": val_loss,
            "best_val_coverage": val_cov,
            "best_epoch": 0,
        }
        return capper, metrics

    # Neural baseline modes (B3: state_time, B9: time_only, ablation: state_only)
    state_dim = dataset["observations"].shape[1]

    # Compute normalization statistics from training trajectories if not provided
    if state_mean is None or state_std is None:
        train_obs_raw = dataset["observations"][train_mask]
        state_mean = np.mean(train_obs_raw, axis=0, dtype=np.float32)
        state_std = np.maximum(np.std(train_obs_raw, axis=0, dtype=np.float32), 1e-6)
        norm_provenance = {"source": "return_training_split", "verified": True}
    else:
        norm_provenance = {"source": "external_unverified", "verified": False}

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

    train_generator = torch.Generator()
    if seed is not None:
        train_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    torch_device = torch.device(device)
    model = ReturnCapMLP(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        mode=mode,
    ).to(torch_device)

    criterion_train = ExpectileLoss(tau=tau, reduction="mean")
    criterion_sum = ExpectileLoss(tau=tau, reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss_hist: list[float] = []
    val_loss_hist: list[float] = []
    val_cov_hist: list[float] = []

    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    stagnant_epochs = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss_sum = 0.0
        total_train_samples = 0
        for b_obs, b_t, b_y in train_loader:
            b_obs = b_obs.to(torch_device)
            b_t = b_t.to(torch_device)
            b_y = b_y.to(torch_device)

            optimizer.zero_grad()
            preds = model(b_obs, b_t)
            loss = criterion_train(preds, b_y)
            loss.backward()
            optimizer.step()

            batch_sz = b_y.size(0)
            total_train_loss_sum += loss.item() * batch_sz
            total_train_samples += batch_sz

        avg_train_loss = total_train_loss_sum / max(1, total_train_samples)
        train_loss_hist.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss_sum = 0.0
        total_val_samples = 0
        covered_count = 0

        with torch.no_grad():
            for b_obs, b_t, b_y in val_loader:
                b_obs = b_obs.to(torch_device)
                b_t = b_t.to(torch_device)
                b_y = b_y.to(torch_device)

                preds = model(b_obs, b_t)
                loss_sum = criterion_sum(preds, b_y)
                total_val_loss_sum += loss_sum.item()
                batch_sz = b_y.size(0)
                total_val_samples += batch_sz

                # Diagnostic empirical coverage: fraction where ground truth return <= cap + margin
                # Note: Coverage is purely diagnostic; expectiles do NOT have quantile coverage guarantees.
                covered = (b_y <= preds + margin).sum().item()
                covered_count += covered

        avg_val_loss = total_val_loss_sum / max(1, total_val_samples)
        val_coverage = float(covered_count) / max(1, total_val_samples)

        val_loss_hist.append(avg_val_loss)
        val_cov_hist.append(val_coverage)

        # Early stopping tracking
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
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

    # Report metrics for the checkpoint actually restored, not the last epoch.
    independent = evaluate_return_capper(
        dataset, capper, horizon=horizon, margin=margin, tau=tau, split="validation"
    )
    best_val_loss = independent["loss"]
    best_val_coverage = independent["coverage"]
    metrics = {
        "train_loss": train_loss_hist,
        "val_loss": val_loss_hist,
        "val_coverage": val_cov_hist,
        "best_val_loss": best_val_loss,
        "best_val_coverage": best_val_coverage,
        "best_epoch": best_epoch,
        "restored_checkpoint_evaluation": independent,
        "normalization_provenance": norm_provenance,
    }
    return capper, metrics


def _return_architecture(payload: dict[str, Any]) -> dict[str, Any]:
    """Read architecture fields, inferring them for legacy neural checkpoints."""
    mode = payload.get("mode")
    state_dict = payload.get("model_state_dict")
    if mode == "global" or state_dict is None:
        if payload.get("cap_value") is None:
            raise ValueError("Return checkpoint has neither model_state_dict nor cap_value")
        return {"mode": "global", "state_dim": None, "hidden_dim": None, "num_layers": None}

    if mode not in ("state_time", "time_only", "state_only"):
        raise ValueError(f"Unsupported return-model mode: {mode!r}")

    if all(key in payload for key in ("state_dim", "hidden_dim", "num_layers")):
        architecture = {
            "mode": mode,
            "state_dim": int(payload["state_dim"]),
            "hidden_dim": int(payload["hidden_dim"]),
            "num_layers": int(payload["num_layers"]),
        }
    else:
        # Legacy checkpoint: recover the MLP shape from its state dict.
        linear_weights = {
            int(key.split(".")[1]): value
            for key, value in state_dict.items()
            if key.startswith("network.") and key.endswith(".weight") and value.ndim == 2
        }
        if not linear_weights:
            raise ValueError("Legacy return checkpoint has no reconstructable MLP layers")
        ordered = [linear_weights[index] for index in sorted(linear_weights)]
        state_mean = payload.get("state_mean")
        if state_mean is None:
            raise ValueError(
                "Legacy return checkpoint is missing state_mean; cannot infer state_dim"
            )
        architecture = {
            "mode": mode,
            "state_dim": int(np.asarray(state_mean).shape[0]),
            "hidden_dim": int(ordered[0].shape[0]),
            "num_layers": len(ordered),
        }

    expected_input = {
        "state_time": architecture["state_dim"] + 1,
        "time_only": 1,
        "state_only": architecture["state_dim"],
    }[mode]
    actual_input = int(ordered[0].shape[1]) if "ordered" in locals() else None
    if actual_input is not None and actual_input != expected_input:
        raise ValueError(
            f"Return checkpoint input dimension {actual_input} is incompatible with "
            f"mode {mode!r} and state_dim {architecture['state_dim']}"
        )
    if architecture["num_layers"] < 2:
        raise ValueError("Return checkpoint must contain at least two linear layers")
    return architecture


def save_return_capper(
    path: str | Path,
    capper: ReturnCapper,
    *,
    metrics: dict[str, Any] | None = None,
    seed: int | None = None,
    tau: float | None = None,
    training_config: dict[str, Any] | None = None,
    dataset_provenance: dict[str, Any] | None = None,
) -> Path:
    """Save Module R weights plus the configuration needed to rebuild them."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metrics = metrics or {}

    if isinstance(capper.model, GlobalReturnCap):
        architecture = {"mode": "global", "state_dim": None, "hidden_dim": None, "num_layers": None}
        model_state_dict = None
        cap_value = float(capper.model.cap_value)
        norm_provenance = {"source": "not_applicable", "verified": True}
    elif isinstance(capper.model, ReturnCapMLP):
        architecture = {
            "mode": capper.model.mode,
            "state_dim": int(capper.model.state_dim),
            "hidden_dim": int(capper.model.hidden_dim),
            "num_layers": sum(1 for layer in capper.model.network if isinstance(layer, nn.Linear)),
        }
        model_state_dict = capper.model.state_dict()
        cap_value = None
        norm_provenance = metrics.get("normalization_provenance") or {
            "source": "unknown",
            "verified": False,
        }
    else:
        raise ValueError(f"Unsupported return-capper model type: {type(capper.model)!r}")

    payload = {
        "format_version": 1,
        "kind": "udt_return_capper",
        "architecture": architecture,
        "mode": architecture["mode"],
        "state_dim": architecture["state_dim"],
        "hidden_dim": architecture["hidden_dim"],
        "num_layers": architecture["num_layers"],
        "horizon": int(capper.horizon),
        "tau": tau,
        "margin": float(capper.margin),
        "scale": float(capper.scale),
        "seed": seed,
        "state_mean": None if capper.state_mean is None else np.asarray(capper.state_mean).tolist(),
        "state_std": None if capper.state_std is None else np.asarray(capper.state_std).tolist(),
        "normalization_provenance": norm_provenance,
        "training_config": training_config,
        "dataset_provenance": dataset_provenance,
        "metrics": metrics,
        "model_state_dict": model_state_dict,
        "cap_value": cap_value,
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def load_return_capper(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_horizon: int | None = None,
    expected_scale: float | None = None,
    expected_state_dim: int | None = None,
    weights_only: bool = True,
    allow_unsafe_legacy: bool = False,
) -> tuple[ReturnCapper, dict[str, Any]]:
    """Load Module R and reject incompatible configurations.

    Supports new ``.pt`` checkpoints, legacy neural ``.pt`` checkpoints, and the
    legacy global ``.json`` baseline. Legacy architectures are inferred from the
    saved state dict; anything that cannot be reconstructed fails loudly.
    """
    payload = load_checkpoint_payload(
        path,
        weights_only=weights_only,
        allow_unsafe_legacy=allow_unsafe_legacy,
    )
    if payload.get("kind") not in (None, "udt_return_capper"):
        raise ValueError(f"{path} is not a U-DT return-capper checkpoint")
    if payload.get("format_version") not in (None, 1):
        raise ValueError(
            f"Unsupported return checkpoint format_version={payload.get('format_version')}"
        )

    architecture = _return_architecture(payload)
    horizon = payload.get("horizon")
    scale = payload.get("scale")
    if horizon is None or scale is None:
        raise ValueError("Return checkpoint is missing horizon or scale")

    checks = {
        "horizon": (int(horizon), expected_horizon),
        "scale": (float(scale), expected_scale),
        "state_dim": (architecture["state_dim"], expected_state_dim),
    }
    for name, (actual, expected) in checks.items():
        if expected is None or actual is None:
            continue
        if name == "scale":
            if not np.isclose(actual, float(expected)):
                raise ValueError(
                    f"Checkpoint {name} ({actual}) is incompatible with expected configuration ({expected})"
                )
        elif int(actual) != int(expected):
            raise ValueError(
                f"Checkpoint {name} ({actual}) is incompatible with expected configuration ({expected})"
            )

    state_mean = payload.get("state_mean")
    state_std = payload.get("state_std")
    if architecture["mode"] == "global":
        model: nn.Module | GlobalReturnCap = GlobalReturnCap(float(payload["cap_value"]))
        state_mean = None
        state_std = None
    else:
        if state_mean is None or state_std is None:
            raise ValueError("Return checkpoint is missing normalization statistics")
        model = ReturnCapMLP(
            state_dim=architecture["state_dim"],
            hidden_dim=architecture["hidden_dim"],
            num_layers=architecture["num_layers"],
            mode=architecture["mode"],
        )
        model.load_state_dict(payload["model_state_dict"])
        model.eval()

    capper = ReturnCapper(
        model=model,
        horizon=int(horizon),
        margin=float(payload.get("margin", 0.0)),
        scale=float(scale),
        state_mean=state_mean,
        state_std=state_std,
        device=str(device),
    )
    return capper, payload
