"""Closed-loop entrypoints for the B0/B3/B4/B5 pilot variants."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from udt.dt_adapter import DTAdapter
from udt.dynamics import DynamicsEnsemble, ModuleUController
from udt.evaluation import RolloutResult, evaluate_with_budget
from udt.return_cap import ReturnCapper


SUPPORTED_VARIANTS = ("B0", "B3", "B4", "B5")


@dataclass
class VariantRun:
    """Rollout plus mechanism diagnostics for one variant."""

    variant: str
    rollout: RolloutResult

    def diagnostics(self) -> dict[str, Any]:
        infos = self.rollout.policy_infos
        return {
            "variant": self.variant,
            "raw_rtg": list(self.rollout.raw_budgets),
            "token_rtg": list(self.rollout.tokens_used),
            "original_actions": [np.asarray(i.get("original_action")).tolist()
                                 for i in infos if "original_action" in i],
            "selected_actions": [np.asarray(a).tolist() for a in self.rollout.actions],
            "executed_actions": [np.asarray(a).tolist()
                                 for a in self.rollout.executed_actions],
            "disagreements": [np.asarray(i.get("disagreements")).tolist()
                              for i in infos if "disagreements" in i],
            "selected_indices": [i.get("selected_index") for i in infos],
            "seed_mapping": self.rollout.seed_mapping,
            "env_seeding": self.rollout.env_seeding,
        }


def run_variant(
    variant: str,
    env: Any,
    dt_adapter: DTAdapter,
    *,
    capper: ReturnCapper | None = None,
    ensemble: DynamicsEnsemble | None = None,
    return_model: ReturnCapper | None = None,
    num_candidates: int = 10,
    radius: float = 0.05,
    beta: float = 0.1,
    horizon: int = 1000,
    initial_target: float = 1000.0,
    reward_scale: float = 1000.0,
    seed: int | None = None,
    obs_noise_scale: float = 0.0,
    act_noise_scale: float = 0.0,
    state_std: np.ndarray | None = None,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
    strict_env_seed: bool = False,
) -> VariantRun:
    """Run one closed-loop variant.

    B0: vanilla DT (R off, U off).
    B3: DT + state-time RTG cap (R on, U off).
    B4: DT + U reranking without cap (R off, U on, g_psi still required).
    B5: DT + R + U.
    """
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {SUPPORTED_VARIANTS}")

    # Validate scale and horizon invariants across active components:
    # c_dynamics == c_return == c_capper == c_rollout
    # H_dynamics == H_return == H_capper == H_controller == H_rollout
    if ensemble is not None:
        if ensemble.reward_scale is not None and not np.isclose(ensemble.reward_scale, reward_scale):
            raise ValueError(
                f"Dynamics ensemble reward_scale ({ensemble.reward_scale}) != rollout reward_scale ({reward_scale})"
            )
        if ensemble.horizon is not None and ensemble.horizon != horizon:
            raise ValueError(
                f"Dynamics ensemble horizon ({ensemble.horizon}) != rollout horizon ({horizon})"
            )

    if return_model is not None:
        if not np.isclose(return_model.scale, reward_scale):
            raise ValueError(
                f"Return model scale ({return_model.scale}) != rollout reward_scale ({reward_scale})"
            )
        if return_model.horizon is not None and return_model.horizon != horizon:
            raise ValueError(
                f"Return model horizon ({return_model.horizon}) != rollout horizon ({horizon})"
            )

    if capper is not None:
        if not np.isclose(capper.scale, reward_scale):
            raise ValueError(
                f"Capper scale ({capper.scale}) != rollout reward_scale ({reward_scale})"
            )
        if capper.horizon is not None and capper.horizon != horizon:
            raise ValueError(
                f"Capper horizon ({capper.horizon}) != rollout horizon ({horizon})"
            )

    if capper is not None and return_model is not None:
        if not np.isclose(capper.scale, return_model.scale):
            raise ValueError(
                f"Capper scale ({capper.scale}) != return model scale ({return_model.scale})"
            )
        if (
            capper.horizon is not None
            and return_model.horizon is not None
            and capper.horizon != return_model.horizon
        ):
            raise ValueError(
                f"Capper horizon ({capper.horizon}) != return model horizon ({return_model.horizon})"
            )

    if variant in ("B4", "B5"):
        if ensemble is None or return_model is None:
            raise ValueError(f"{variant} requires ensemble and return_model (g_psi)")
        controller = ModuleUController(
            dt_adapter=dt_adapter,
            ensemble=ensemble,
            return_model=return_model,
            num_candidates=num_candidates,
            radius=radius,
            beta=beta,
            action_low=action_low,
            action_high=action_high,
            horizon=horizon,
        )
        policy = controller.policy_fn
        reset_fn = controller.reset
        set_rng_fn = controller.set_rng
        requires_proposal_rng = num_candidates > 1
    else:
        controller = None
        policy = dt_adapter.policy_fn
        reset_fn = dt_adapter.reset
        set_rng_fn = getattr(dt_adapter, "set_rng", None)
        requires_proposal_rng = False

    active_capper = capper if variant in ("B3", "B5") else None
    if variant in ("B3", "B5") and active_capper is None:
        raise ValueError(f"{variant} requires a Module R capper")

    rollout = evaluate_with_budget(
        env=env,
        policy_fn=policy,
        capper=active_capper,
        initial_target=initial_target,
        horizon=horizon,
        reward_scale=reward_scale,
        obs_noise_scale=obs_noise_scale,
        act_noise_scale=act_noise_scale,
        state_std=state_std,
        action_low=action_low,
        action_high=action_high,
        seed=seed,
        policy_reset_fn=reset_fn,
        policy_set_rng_fn=set_rng_fn,
        requires_proposal_rng=requires_proposal_rng,
        strict_env_seed=strict_env_seed,
    )
    return VariantRun(variant=variant, rollout=rollout)
