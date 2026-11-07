"""Module R: Return-to-Go Calibration and Capping for Frozen Decision Transformers.

Provides:
- ExpectileLoss: Asymmetric squared loss for upper-expectile regression.
- compute_global_expectile: Scalar expectile root-finding for Baseline B2.
- GlobalReturnCap: Constant scalar return cap (Baseline B2).
"""

from typing import Literal
import numpy as np
import torch
import torch.nn as nn


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

    def get_cap(self, state: np.ndarray, step: int = 0, horizon: int | None = None) -> float:
        """Return the constant global cap."""
        return self.cap_value
