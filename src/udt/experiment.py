"""Closed-loop entrypoints for the B0/B3/B4/B5 pilot variants."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from udt.dt_adapter import DTAdapter
from udt.dynamics import DynamicsEnsemble, ModuleUController
from udt.evaluation import RolloutResult, evaluate_with_budget
from udt.profiling import environment_metadata
from udt.provenance import compute_file_sha256, dataset_provenance, git_commit_sha
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
            "rtg_alpha": float(self.rollout.rtg_alpha),
            "raw_rtg": list(self.rollout.raw_budgets),
            "token_rtg": list(self.rollout.tokens_used),
            "caps": list(self.rollout.caps),
            "effective_intervention_frequency": float(self.rollout.effective_intervention_frequency),
            "mean_rtg_adjustment": float(self.rollout.mean_rtg_adjustment),
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
    rtg_alpha: float = 1.0,
) -> VariantRun:
    """Run one closed-loop variant.

    B0: vanilla DT (R off, U off).
    B3: DT + state-time RTG cap (R on, U off).
    B4: DT + U reranking without cap (R off, U on, g_psi still required).
    B5: DT + R + U.
    """
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {SUPPORTED_VARIANTS}")
    if not np.isfinite(rtg_alpha) or rtg_alpha < 0.0 or rtg_alpha > 1.0:
        raise ValueError(f"rtg_alpha must be finite and in [0, 1], got {rtg_alpha}")

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
        rtg_alpha=rtg_alpha,
    )
    return VariantRun(variant=variant, rollout=rollout)


def evaluate_variant(
    variant: str,
    env: Any,
    dt_adapter: DTAdapter,
    *,
    episodes: int,
    seed: int = 0,
    capper: ReturnCapper | None = None,
    ensemble: DynamicsEnsemble | None = None,
    return_model: ReturnCapper | None = None,
    num_candidates: int = 10,
    radius: float = 0.05,
    beta: float = 0.1,
    horizon: int = 1000,
    initial_target: float = 1000.0,
    reward_scale: float = 1000.0,
    obs_noise_scale: float = 0.0,
    act_noise_scale: float = 0.0,
    state_std: np.ndarray | None = None,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
    strict_env_seed: bool = False,
    collect_diagnostics: bool = False,
    rtg_alpha: float = 1.0,
) -> dict[str, Any]:
    """Run the canonical multi-episode evaluation loop for one variant."""
    if not isinstance(episodes, (int, np.integer)) or episodes <= 0:
        raise ValueError(f"episodes must be a positive integer, got {episodes}")
    if not np.isfinite(rtg_alpha) or rtg_alpha < 0.0 or rtg_alpha > 1.0:
        raise ValueError(f"rtg_alpha must be finite and in [0, 1], got {rtg_alpha}")

    episode_records: list[dict[str, Any]] = []
    returns: list[float] = []
    lengths: list[int] = []

    for episode in range(1, int(episodes) + 1):
        episode_seed = int(seed) + episode
        run = run_variant(
            variant,
            env,
            dt_adapter,
            capper=capper,
            ensemble=ensemble,
            return_model=return_model,
            num_candidates=num_candidates,
            radius=radius,
            beta=beta,
            horizon=horizon,
            initial_target=initial_target,
            reward_scale=reward_scale,
            seed=episode_seed,
            obs_noise_scale=obs_noise_scale,
            act_noise_scale=act_noise_scale,
            state_std=state_std,
            action_low=action_low,
            action_high=action_high,
            strict_env_seed=strict_env_seed,
            rtg_alpha=rtg_alpha,
        )
        rollout = run.rollout
        infos = rollout.policy_infos
        selected_indices = [
            info["selected_index"] for info in infos if "selected_index" in info
        ]
        disagreements = [
            float(np.mean(info["disagreements"]))
            for info in infos
            if info.get("disagreements") is not None
        ]
        record: dict[str, Any] = {
            "episode": episode,
            "seed": episode_seed,
            "total_reward": float(rollout.total_reward),
            "length": int(rollout.length),
            "terminated": bool(rollout.terminated),
            "truncated": bool(rollout.truncated),
            "final_raw_budget": float(rollout.final_raw_budget),
            "clipping_frequency": float(rollout.clipping_frequency),
            "effective_intervention_frequency": float(rollout.effective_intervention_frequency),
            "mean_rtg_adjustment": float(rollout.mean_rtg_adjustment),
            "rtg_alpha": float(rollout.rtg_alpha),
            "intervention_frequency": (
                float(np.mean([index != 0 for index in selected_indices]))
                if selected_indices
                else 0.0
            ),
            "mean_disagreement": float(np.mean(disagreements)) if disagreements else None,
            "rtg_raw": list(rollout.raw_budgets) if collect_diagnostics else [],
            "rtg_token": list(rollout.tokens_used) if collect_diagnostics else [],
            "caps": list(rollout.caps) if collect_diagnostics else [],
            "selected_candidates": selected_indices if collect_diagnostics else [],
            "warnings": list(rollout.warnings),
        }
        episode_records.append(record)
        returns.append(record["total_reward"])
        lengths.append(record["length"])

    returns_array = np.asarray(returns, dtype=np.float64)
    lengths_array = np.asarray(lengths, dtype=np.float64)
    terminated = sum(record["terminated"] for record in episode_records)
    truncated = sum(record["truncated"] for record in episode_records)
    return {
        "variant": variant,
        "episodes": int(episodes),
        "seed": int(seed),
        "target_return": float(initial_target),
        "reward_scale": float(reward_scale),
        "horizon": int(horizon),
        "rtg_alpha": float(rtg_alpha),
        "mean_clipping_frequency": (
            float(np.mean([r["clipping_frequency"] for r in episode_records]))
            if episode_records
            else 0.0
        ),
        "mean_effective_intervention_frequency": (
            float(np.mean([r["effective_intervention_frequency"] for r in episode_records]))
            if episode_records
            else 0.0
        ),
        "mean_rtg_adjustment": (
            float(np.mean([r["mean_rtg_adjustment"] for r in episode_records]))
            if episode_records
            else 0.0
        ),
        "mean_return": float(returns_array.mean()),
        "std_return": float(returns_array.std()),
        "median_return": float(np.median(returns_array)),
        "min_return": float(returns_array.min()),
        "max_return": float(returns_array.max()),
        "mean_length": float(lengths_array.mean()),
        "raw_episode_returns": returns,
        "episode_lengths": lengths,
        "termination_reasons": {
            "terminated": int(terminated),
            "truncated": int(truncated),
            "other": int(len(episode_records) - terminated - truncated),
        },
        "episodes_detail": episode_records,
    }


def build_evaluation_report(
    *,
    variant: str,
    env_id: str,
    summary: dict[str, Any],
    config: dict[str, Any],
    checkpoints: dict[str, str | Path | None],
    dataset_path: str | Path | None = None,
    dataset_manifest: dict[str, Any] | None = None,
    parity: dict[str, Any] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Attach one reproducible provenance schema to any variant's results."""
    checkpoint_hashes = {
        name: compute_file_sha256(path)
        for name, path in checkpoints.items()
        if path is not None and Path(path).is_file()
    }
    training_env = (dataset_manifest or {}).get("dataset_id")
    if training_env is None and dataset_path is not None:
        training_env = Path(dataset_path).parent.name
    report = {
        "variant": variant,
        "env": env_id,
        "config": config,
        "provenance": {
            "git_commit": git_commit_sha(),
            "checkpoint_sha256": checkpoint_hashes,
            "dataset": dataset_provenance(dataset_path, dataset_manifest)
            if dataset_path is not None
            else None,
            "software": environment_metadata(device),
            "seeds": {"episode_base": summary.get("seed")},
        },
        "environment_metadata": {
            "evaluation_env": env_id,
            "training_dataset_env": training_env,
            "d4rl_closed_loop_parity": "BLOCKED",
            "d4rl_closed_loop_parity_reason": (
                "Cross-version environment difference: Gymnasium evaluation "
                "(MuJoCo 3) vs D4RL *-v2 data (Gym 0.21 / MuJoCo 2.1). "
                "Comparability has not been demonstrated; do not report official "
                "D4RL normalized scores from this cross-version setup."
            ),
        },
        "parity": parity,
    }
    report.update(summary)
    return report
