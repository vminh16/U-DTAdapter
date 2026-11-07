"""Module R: Return-to-Go Calibration and Capping for Frozen Decision Transformers.

Provides:
- ExpectileLoss: Asymmetric squared loss for upper-expectile regression.
- compute_global_expectile: Scalar expectile root-finding for Baseline B2.
- GlobalReturnCap: Constant scalar return cap (Baseline B2).
- ReturnCapMLP: Neural state-time return predictor (Baseline B3, B9, and ablation).
- ReturnCapper: Inference-time intervention interface hooking into ReturnBudget.
"""

from typing import Literal
import numpy as np
import torch
import torch.nn as nn

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
        device: str = "cpu",
    ):
        self.model = model
        self.horizon = int(horizon)
        self.margin = float(margin)
        self.scale = float(scale)
        self.device = torch.device(device)

        if isinstance(self.model, nn.Module):
            self.model.to(self.device)
            self.model.eval()

    def get_cap(self, state: np.ndarray | torch.Tensor, step: int) -> float:
        """Compute the scalar return cap for a given state and timestep.

        Args:
            state: Observation vector.
            step: Current episode timestep t >= 0.

        Returns:
            Scalar return cap value.
        """
        if isinstance(self.model, GlobalReturnCap):
            return self.model.get_cap(state, step, self.horizon)

        # Compute relative elapsed timestep t / H clamped to [0.0, 1.0]
        t_rel = min(1.0, max(0.0, float(step) / float(self.horizon)))

        with torch.no_grad():
            s_tensor = torch.as_tensor(state, device=self.device, dtype=torch.float32)
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

