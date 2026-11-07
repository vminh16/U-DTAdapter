"""Evaluation rollout harness for Decision Transformers under Module R Return Capping.

Provides:
- RolloutResult: Telemetry and trajectories from evaluation episodes.
- evaluate_with_budget: Closed-loop rollout executing inference-time return capping.
"""

from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np

from udt.intervention import ReturnBudget
from udt.return_cap import ReturnCapper


@dataclass
class RolloutResult:
    """Telemetry and trajectories captured during a closed-loop evaluation episode."""

    total_reward: float
    length: int
    clipping_frequency: float
    final_raw_budget: float
    raw_budgets: list[float] = field(default_factory=list)
    tokens_used: list[float] = field(default_factory=list)
    caps: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)



def evaluate_with_budget(
    env: Any,
    policy_fn: Callable[[list[float], np.ndarray, int], np.ndarray],
    capper: ReturnCapper | None = None,
    initial_target: float = 1000.0,
    horizon: int = 1000,
    reward_scale: float = 1000.0,
) -> RolloutResult:
    """Execute a single closed-loop evaluation rollout with Return Budgeting.

    Hooks Module R (ReturnCapper) into the decision loop at each environment step:
    1. Observe current state z_t.
    2. Compute return cap C_t = g_psi(z_t, t/H) + delta (if capper is provided).
    3. Generate clipped conditioning token B_t = min(B_t^raw, C_t).
    4. Query policy with conditioned token B_t.
    5. Step environment, observe reward r_t, and update raw budget:
       B_{t+1}^raw = B_t^raw - r_t / reward_scale.

    Args:
        env: Gym/Gymnasium compatible environment.
        policy_fn: Callable accepting (token_history, current_state, step) -> action.
        capper: Optional ReturnCapper instance. If None, runs unclipped Baseline B0.
        initial_target: Requested target return G_0 (unscaled).
        horizon: Maximum episode horizon H.
        reward_scale: Reward/Return scaling factor c (default 1000.0).

    Returns:
        RolloutResult containing complete episode telemetry.
    """
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    # Initialize ReturnBudget with scaled raw return target
    initial_raw = float(initial_target) / float(reward_scale)
    budget = ReturnBudget(raw=initial_raw, reward_scale=reward_scale)

    raw_budgets: list[float] = []
    tokens_used: list[float] = []
    caps: list[float] = []
    rewards: list[float] = []
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    clipped_flags: list[bool] = []

    total_reward = 0.0

    for step in range(horizon):
        observations.append(obs)
        raw_budget_at_t = float(budget.raw)
        raw_budgets.append(raw_budget_at_t)

        if capper is not None:
            cap_val = capper.get_cap(obs, step)
            token = budget.token(cap=cap_val, margin=capper.margin)
            was_clipped = raw_budget_at_t > (cap_val + capper.margin)
        else:
            cap_val = float("inf")
            token = budget.token(cap=None)
            was_clipped = False


        caps.append(cap_val)
        tokens_used.append(token)
        clipped_flags.append(was_clipped)

        # Policy forward pass
        action = policy_fn(tokens_used, obs, step)
        actions.append(action)

        step_out = env.step(action)
        # Support both classic Gym (obs, rew, done, info) and Gymnasium (obs, rew, terminated, truncated, info)
        if len(step_out) == 5:
            next_obs, rew, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            next_obs, rew, done, _ = step_out

        rew_float = float(rew)
        rewards.append(rew_float)
        total_reward += rew_float

        # Decrement raw budget by actual observed reward
        budget.observe_reward(rew_float)

        if done:
            break
        obs = next_obs

    clipping_freq = float(np.mean(clipped_flags)) if clipped_flags else 0.0

    return RolloutResult(
        total_reward=total_reward,
        length=len(rewards),
        clipping_frequency=clipping_freq,
        final_raw_budget=float(budget.raw),
        raw_budgets=raw_budgets,
        tokens_used=tokens_used,
        caps=caps,
        rewards=rewards,
        observations=observations,
        actions=actions,
    )

