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
    executed_actions: list[np.ndarray] = field(default_factory=list)
    clean_observations: list[np.ndarray] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False



def evaluate_with_budget(
    env: Any,
    policy_fn: Callable[[list[float], np.ndarray, int], np.ndarray],
    capper: ReturnCapper | None = None,
    initial_target: float = 1000.0,
    horizon: int = 1000,
    reward_scale: float = 1000.0,
    obs_noise_scale: float = 0.0,
    act_noise_scale: float = 0.0,
    state_std: np.ndarray | None = None,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
    rng: np.random.Generator | int | None = None,
) -> RolloutResult:
    """Execute a single closed-loop evaluation rollout with Return Budgeting.

    Hooks Module R (ReturnCapper) into the decision loop at each environment step:
    1. Observe current state s_t, apply observation noise o_t = s_t + alpha * sigma * eps.
    2. Compute return cap C_t = g_psi(o_t, t/H) + delta (if capper is provided).
    3. Generate clipped conditioning token B_t = min(B_t^raw, C_t).
    4. Query policy with conditioned token B_t and observation o_t.
    5. Optionally apply actuator noise a_exec = clip(a* + kappa * D_a * eps).
    6. Step environment with a_exec, observe reward r_t, and update raw budget:
       B_{t+1}^raw = B_t^raw - r_t / reward_scale.

    Args:
        env: Gym/Gymnasium compatible environment.
        policy_fn: Callable accepting (token_history, current_state, step) -> action.
        capper: Optional ReturnCapper instance. If None, runs unclipped Baseline B0.
        initial_target: Requested target return G_0 (unscaled).
        horizon: Maximum episode horizon H.
        reward_scale: Reward/Return scaling factor c (default 1000.0).
        obs_noise_scale: Observation corruption scale alpha >= 0.
        act_noise_scale: Actuator corruption scale kappa >= 0.
        state_std: Observation standard deviation sigma_s for noise scaling.
        action_low: Minimum action bounds for actuator noise clipping.
        action_high: Maximum action bounds for actuator noise clipping.
        rng: Random generator or integer seed for perturbation streams.

    Returns:
        RolloutResult containing complete episode telemetry.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if reward_scale <= 0 or not np.isfinite(reward_scale):
        raise ValueError(f"reward_scale must be positive and finite, got {reward_scale}")
    if obs_noise_scale < 0 or not np.isfinite(obs_noise_scale):
        raise ValueError(f"obs_noise_scale must be non-negative, got {obs_noise_scale}")
    if act_noise_scale < 0 or not np.isfinite(act_noise_scale):
        raise ValueError(f"act_noise_scale must be non-negative, got {act_noise_scale}")

    if isinstance(rng, (int, np.integer)):
        rng_obj = np.random.default_rng(int(rng))
    elif isinstance(rng, np.random.Generator):
        rng_obj = rng
    else:
        rng_obj = np.random.default_rng()

    if capper is not None:
        if not np.isclose(capper.scale, reward_scale):
            raise ValueError(
                f"Reward scale mismatch: capper scale ({capper.scale}) != rollout reward_scale ({reward_scale})"
            )
        if capper.horizon != horizon:
            raise ValueError(
                f"Horizon mismatch: capper horizon ({capper.horizon}) != rollout horizon ({horizon})"
            )

    reset_out = env.reset()
    clean_obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    clean_obs = np.asarray(clean_obs, dtype=np.float32)

    # Initialize ReturnBudget with scaled raw return target
    initial_raw = float(initial_target) / float(reward_scale)
    budget = ReturnBudget(raw=initial_raw, reward_scale=reward_scale)

    raw_budgets: list[float] = []
    tokens_used: list[float] = []
    caps: list[float] = []
    rewards: list[float] = []
    clean_observations: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    commanded_actions: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    clipped_flags: list[bool] = []

    total_reward = 0.0
    is_terminated = False
    is_truncated = False

    for step in range(horizon):
        clean_observations.append(clean_obs)

        # Apply observation noise if configured: o_t = s_t + alpha * sigma * eps
        if obs_noise_scale > 0.0:
            std = state_std if state_std is not None else np.ones_like(clean_obs)
            noise = rng_obj.normal(size=clean_obs.shape).astype(np.float32)
            policy_obs = clean_obs + float(obs_noise_scale) * std * noise
        else:
            policy_obs = clean_obs.copy()

        observations.append(policy_obs)

        raw_budget_at_t = float(budget.raw)
        raw_budgets.append(raw_budget_at_t)

        if capper is not None:
            cap_val = capper.get_cap(policy_obs, step)
            token = budget.token(cap=cap_val, margin=capper.margin)
            was_clipped = raw_budget_at_t > (cap_val + capper.margin)
        else:
            cap_val = float("inf")
            token = budget.token(cap=None)
            was_clipped = False

        caps.append(cap_val)
        tokens_used.append(token)
        clipped_flags.append(was_clipped)

        # Policy forward pass produces commanded action a*
        action = policy_fn(tokens_used, policy_obs, step)
        commanded_actions.append(action)

        # Apply actuator noise: a_exec = clip(a* + kappa * D_a * eps)
        if act_noise_scale > 0.0:
            if action_low is not None and action_high is not None:
                d_a = (action_high - action_low) / 2.0
            else:
                d_a = np.ones_like(action)
            act_noise = rng_obj.normal(size=action.shape).astype(np.float32)
            exec_action = action + float(act_noise_scale) * d_a * act_noise
            if action_low is not None and action_high is not None:
                exec_action = np.clip(exec_action, action_low, action_high)
        else:
            exec_action = action

        executed_actions.append(exec_action)

        step_out = env.step(exec_action)
        # Support both classic Gym (obs, rew, done, info) and Gymnasium (obs, rew, terminated, truncated, info)
        if len(step_out) == 5:
            next_obs, rew, term, trunc, _ = step_out
            is_terminated = bool(term)
            is_truncated = bool(trunc) or (step + 1 >= horizon)
            done = is_terminated or is_truncated
        else:
            next_obs, rew, done, _ = step_out
            if done:
                if step + 1 >= horizon:
                    is_truncated = True
                    is_terminated = False
                else:
                    is_terminated = True
                    is_truncated = False
            elif step + 1 >= horizon:
                is_truncated = True
                done = True

        rew_float = float(rew)
        rewards.append(rew_float)
        total_reward += rew_float

        # Decrement raw budget by actual observed reward
        budget.observe_reward(rew_float)

        if done:
            break
        clean_obs = np.asarray(next_obs, dtype=np.float32)

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
        actions=commanded_actions,
        executed_actions=executed_actions,
        clean_observations=clean_observations,
        terminated=is_terminated,
        truncated=is_truncated,
    )

