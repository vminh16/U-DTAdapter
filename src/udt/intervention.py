"""Numerical intervention primitives; callers supply frozen model predictions.

Scores implement the RRD heuristic, not calibrated values or safety bounds.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ReturnBudget:
    """Track raw requested scaled return separately from the token sent to DT."""

    raw: float
    reward_scale: float = 1000.

    def __post_init__(self) -> None:
        if not np.isfinite(self.raw) or not np.isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError('Budget must be finite and reward_scale positive')

    def token(self, cap: float | None = None, margin: float = 0.,
              alpha: float = 1.0) -> float:
        if not np.isfinite(margin) or margin < 0 or (cap is not None and not np.isfinite(cap)):
            raise ValueError('Cap must be finite and margin nonnegative')
        if not np.isfinite(alpha) or alpha < 0 or alpha > 1:
            raise ValueError('alpha must be finite and in [0, 1]')
        if cap is None or alpha == 0.0:
            return float(self.raw)
        ceiling = cap + margin
        if alpha == 1.0:
            return float(min(self.raw, ceiling))
        if self.raw <= ceiling:
            return float(self.raw)
        tok = (1.0 - alpha) * self.raw + alpha * ceiling
        return float(min(self.raw, max(ceiling, tok)))

    def observe_reward(self, reward: float) -> None:
        if not np.isfinite(reward):
            raise ValueError('Reward must be finite')
        self.raw -= reward / self.reward_scale


def propose_actions(action: np.ndarray, *, low: np.ndarray, high: np.ndarray,
                    count: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Return original action first, followed by bounded Gaussian perturbations."""
    action, low, high = (np.asarray(v, dtype=float) for v in (action, low, high))
    if action.ndim != 1 or not action.size or low.shape != action.shape or high.shape != action.shape:
        raise ValueError('Action and bounds must be matching nonempty vectors')
    if not all(np.isfinite(v).all() for v in (action, low, high)) or np.any(low >= high):
        raise ValueError('Require finite, strictly ordered action bounds')
    if np.any(action < low) or np.any(action > high):
        raise ValueError('Original action must already satisfy bounds')
    if not isinstance(count, (int, np.integer)) or count < 1 or not np.isfinite(radius) or radius < 0:
        raise ValueError('Require integer count >= 1 and finite nonnegative radius')
    noise = rng.normal(size=(count - 1, action.size))
    others = np.clip(action + radius * (high - low) / 2 * noise, low, high)
    return np.concatenate((action[None], others), axis=0)


def ensemble_scores(*, rewards: np.ndarray, next_returns: np.ndarray,
                    terminal_probabilities: np.ndarray, next_states: np.ndarray,
                    beta: float, at_horizon: bool) -> np.ndarray:
    """Score N candidates from M heads; argmax uses original-first tie breaking.

    rewards, next_returns and probabilities have shape (M, N). States are
    normalized predictions with shape (M, N, D). next_returns must already be
    evaluated per head, never at the mean state. At the horizon callers can pass
    zeros for next_returns without evaluating a return model outside its domain.
    """
    rewards, next_returns, terminal_probabilities, next_states = (
        np.asarray(v, dtype=float) for v in
        (rewards, next_returns, terminal_probabilities, next_states))
    if rewards.ndim != 2 or min(rewards.shape) < 1:
        raise ValueError('Require rewards shape (M, N), M and N positive')
    if next_returns.shape != rewards.shape or terminal_probabilities.shape != rewards.shape:
        raise ValueError('Return and probability shapes must match rewards')
    if next_states.ndim != 3 or next_states.shape[:2] != rewards.shape or next_states.shape[2] < 1:
        raise ValueError('Require states shape (M, N, D)')
    if not all(np.isfinite(v).all() for v in (rewards, next_returns, terminal_probabilities, next_states)):
        raise ValueError('Predictions must be finite')
    if np.any(terminal_probabilities < 0) or np.any(terminal_probabilities > 1):
        raise ValueError('Terminal probabilities must lie in [0, 1]')
    if not np.isfinite(beta) or beta < 0:
        raise ValueError('beta must be finite and nonnegative')
    centered = next_states - next_states.mean(axis=0)
    disagreement = np.sqrt(np.square(centered).sum(axis=-1).mean(axis=0))
    continuation = np.zeros_like(next_returns) if at_horizon else (1 - terminal_probabilities) * next_returns
    return (rewards + continuation).mean(axis=0) - beta * disagreement
