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
    policy_infos: list[dict[str, Any]] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    seed_mapping: dict[str, Any] = field(default_factory=dict)
    env_seeding: str = "unknown"
    warnings: list[str] = field(default_factory=list)



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
    seed: int | None = None,
    policy_reset_fn: Callable[[], None] | None = None,
    policy_set_rng_fn: Callable[[np.random.Generator], None] | None = None,
    requires_proposal_rng: bool | None = None,
    strict_env_seed: bool = False,
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
        rng: Legacy random generator or integer seed. Prefer ``seed``.
        seed: Base seed used to spawn independent environment, observation,
            actuator, and candidate-proposal RNG streams.
        policy_reset_fn: Optional callback invoked before the episode to clear
            policy/controller history. Detected automatically when the policy
            exposes ``reset``.
        strict_env_seed: If True, fail when the environment cannot be seeded.

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

    if seed is not None and rng is not None:
        raise ValueError("Pass either seed or rng, not both")
    if seed is None and isinstance(rng, (int, np.integer)):
        seed = int(rng)
    elif seed is None and isinstance(rng, np.random.Generator):
        # Derive a reproducible base seed from the caller's generator without
        # reusing that generator for any perturbation stream.
        seed = int(rng.integers(0, np.iinfo(np.int64).max))

    base_sequence = np.random.SeedSequence(None if seed is None else int(seed))
    env_seq, obs_seq, act_seq, proposal_seq, _aux = base_sequence.spawn(5)
    env_rng = np.random.default_rng(env_seq)
    obs_rng = np.random.default_rng(obs_seq)
    act_rng = np.random.default_rng(act_seq)
    proposal_rng = np.random.default_rng(proposal_seq)
    env_seed = int(env_seq.generate_state(1, dtype=np.uint32)[0])

    seed_mapping = {
        "base_entropy": base_sequence.entropy,
        "streams": {
            "environment": {"spawn_key": list(env_seq.spawn_key), "seed": env_seed},
            "observation_noise": {
                "spawn_key": list(obs_seq.spawn_key),
                "seed": int(obs_seq.generate_state(1, dtype=np.uint32)[0]),
            },
            "actuator_noise": {
                "spawn_key": list(act_seq.spawn_key),
                "seed": int(act_seq.generate_state(1, dtype=np.uint32)[0]),
            },
            "candidate_proposal": {
                "spawn_key": list(proposal_seq.spawn_key),
                "seed": int(proposal_seq.generate_state(1, dtype=np.uint32)[0]),
            },
        },
    }

    # Candidate proposals are owned by the policy/controller; give it its own
    # stream when the policy supports explicit injection.
    policy_owner = getattr(policy_fn, "__self__", None)
    setter = policy_set_rng_fn
    if setter is None:
        for candidate in (policy_fn, policy_owner):
            possible = getattr(candidate, "set_rng", None) if candidate is not None else None
            if callable(possible):
                setter = possible
                break

    proposal_injected = False
    if callable(setter):
        setter(proposal_rng)
        proposal_injected = True

    policy_requires_proposals = requires_proposal_rng
    if policy_requires_proposals is None:
        for candidate in (policy_fn, policy_owner):
            req = getattr(candidate, "requires_proposal_rng", None)
            if req is not None:
                policy_requires_proposals = bool(req)
                break
        if policy_requires_proposals is None:
            policy_requires_proposals = False

    if policy_requires_proposals and not proposal_injected:
        raise RuntimeError(
            "Candidate-proposal RNG injection failed: policy requires proposal RNG "
            "for deterministic candidate proposals, but no set_rng interface was available."
        )

    seed_mapping["streams"]["candidate_proposal"]["injected"] = proposal_injected

    if capper is not None:
        if not np.isclose(capper.scale, reward_scale):
            raise ValueError(
                f"Reward scale mismatch: capper scale ({capper.scale}) != rollout reward_scale ({reward_scale})"
            )
        if capper.horizon != horizon:
            raise ValueError(
                f"Horizon mismatch: capper horizon ({capper.horizon}) != rollout horizon ({horizon})"
            )

    warnings_list: list[str] = []

    # Episode isolation: clear controller/DT history before the first step.
    resetter = policy_reset_fn
    if resetter is None:
        for candidate in (policy_fn, policy_owner):
            possible = getattr(candidate, "reset", None) if candidate is not None else None
            if callable(possible):
                resetter = possible
                break
    if resetter is not None:
        resetter()

    try:
        reset_out = env.reset(seed=env_seed)
        env_seeding = "reset_seed"
    except TypeError:
        legacy_seed = getattr(env, "seed", None)
        if callable(legacy_seed):
            legacy_seed(env_seed)
            reset_out = env.reset()
            env_seeding = "legacy_seed"
        else:
            reset_out = env.reset()
            env_seeding = "unsupported"
            warnings_list.append(
                "Environment does not accept reset(seed=...) and has no seed() method; "
                "environment-generated randomness is not controlled."
            )
            if strict_env_seed:
                raise RuntimeError(warnings_list[-1])

    clean_obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    clean_obs = np.asarray(clean_obs, dtype=np.float32)
    if not np.isfinite(clean_obs).all():
        raise ValueError("Environment returned a non-finite observation at reset")

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
    policy_infos: list[dict[str, Any]] = []

    if state_std is not None:
        state_std = np.asarray(state_std, dtype=np.float32)
        if state_std.shape != clean_obs.shape:
            raise ValueError(
                f"state_std shape {state_std.shape} does not match observation shape "
                f"{clean_obs.shape}"
            )
        if not np.isfinite(state_std).all() or np.any(state_std < 0):
            raise ValueError("state_std must be finite and nonnegative")

    if action_low is not None:
        action_low = np.asarray(action_low, dtype=np.float32)
    if action_high is not None:
        action_high = np.asarray(action_high, dtype=np.float32)
    if (action_low is None) != (action_high is None):
        raise ValueError("action_low and action_high must be provided together")
    if action_low is not None:
        if action_low.shape != action_high.shape:
            raise ValueError("action_low and action_high must have the same shape")
        if not np.isfinite(action_low).all() or not np.isfinite(action_high).all():
            raise ValueError("action bounds must be finite")
        if np.any(action_low >= action_high):
            raise ValueError("action_low must be strictly below action_high")

    action_space = getattr(env, "action_space", None)
    env_action_shape = getattr(action_space, "shape", None)
    if env_action_shape is None and action_low is not None:
        env_action_shape = action_low.shape
    action_shape: tuple[int, ...] | None = None

    total_reward = 0.0
    is_terminated = False
    is_truncated = False

    for step in range(horizon):
        clean_observations.append(clean_obs.copy())

        # Apply observation noise if configured: o_t = s_t + alpha * sigma * eps
        if obs_noise_scale > 0.0:
            std = state_std if state_std is not None else np.ones_like(clean_obs)
            noise = obs_rng.normal(size=clean_obs.shape).astype(np.float32)
            policy_obs = clean_obs + float(obs_noise_scale) * std * noise
        else:
            policy_obs = clean_obs.copy()

        observations.append(policy_obs.copy())

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

        # Policy forward pass produces commanded action a*. Controllers may
        # additionally return a diagnostics dict for mechanism analysis.
        policy_out = policy_fn(tokens_used, policy_obs, step)
        if isinstance(policy_out, tuple) and len(policy_out) == 2 and isinstance(policy_out[1], dict):
            policy_out, step_info = policy_out
            policy_infos.append(dict(step_info))
        action = np.asarray(policy_out, dtype=np.float32)
        if action.ndim == 0 or action.size == 0:
            raise ValueError(f"policy returned an empty or scalar action: {action!r}")
        if action_shape is None:
            action_shape = action.shape
            if env_action_shape is not None and tuple(env_action_shape) != action.shape:
                raise ValueError(
                    f"policy action shape {action.shape} does not match environment "
                    f"action shape {tuple(env_action_shape)}"
                )
        elif action.shape != action_shape:
            raise ValueError(
                f"policy action shape changed within episode: {action.shape} != {action_shape}"
            )
        if not np.isfinite(action).all():
            raise ValueError("policy returned a non-finite action")
        if action_low is not None:
            if action.shape != action_low.shape:
                raise ValueError(
                    f"policy action shape {action.shape} does not match bounds "
                    f"{action_low.shape}"
                )
            if np.any(action < action_low) or np.any(action > action_high):
                raise ValueError(
                    "policy commanded action is outside environment bounds; "
                    "clip it inside the policy/adapter before evaluation"
                )
        commanded_actions.append(action.copy())

        # Apply actuator noise: a_exec = clip(a* + kappa * D_a * eps)
        if act_noise_scale > 0.0:
            if action_low is not None and action_high is not None:
                d_a = (action_high - action_low) / 2.0
            else:
                d_a = np.ones_like(action)
            act_noise = act_rng.normal(size=action.shape).astype(np.float32)
            exec_action = action + float(act_noise_scale) * d_a * act_noise
            if action_low is not None and action_high is not None:
                exec_action = np.clip(exec_action, action_low, action_high)
            if not np.isfinite(exec_action).all():
                raise ValueError("actuator noise produced a non-finite action")
        else:
            exec_action = action.copy()

        executed_actions.append(exec_action.copy())

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
        if not np.isfinite(rew_float):
            raise ValueError("Environment returned a non-finite reward")
        rewards.append(rew_float)
        total_reward += rew_float

        # Decrement raw budget by actual observed reward
        budget.observe_reward(rew_float)

        if done:
            break
        clean_obs = np.asarray(next_obs, dtype=np.float32)
        if clean_obs.shape != observations[0].shape:
            raise ValueError(
                f"Environment observation shape changed: {clean_obs.shape} != "
                f"{observations[0].shape}"
            )
        if not np.isfinite(clean_obs).all():
            raise ValueError("Environment returned a non-finite observation")

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
        policy_infos=policy_infos,
        terminated=is_terminated,
        truncated=is_truncated,
        seed_mapping=seed_mapping,
        env_seeding=env_seeding,
        warnings=warnings_list,
    )

