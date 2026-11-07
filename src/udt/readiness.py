"""Pilot readiness checks for U-DT-Adapter experiments.

A run is only reportable when its artifacts carry enough evidence to reproduce
and audit it: dataset provenance, split/seed integrity, frozen-DT parity,
trajectory-bootstrap metadata, normalization statistics, all required R/U
components, final-test isolation, and traceable metrics/checkpoints.

This module intentionally reports ``blocked`` rather than pretending a missing
real-DT parity check passed. Unit-level mock checks are never treated as
research validation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SUPPORTED_VARIANTS = ("B0", "B3", "B4", "B5")
_LOGICAL_SPLIT_KEYS = ("train_episode_ids", "validation_episode_ids", "test_episode_ids")


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA-256 of a file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _check(status: str, detail: str, **evidence: Any) -> dict[str, Any]:
    return {"status": status, "detail": detail, "evidence": evidence}


def _as_int_list(values: Iterable[Any] | None) -> list[int]:
    if values is None:
        return []
    return [int(value) for value in values]


def _load_payload(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"{path} does not contain a JSON object")
        return payload

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a dictionary payload")
    return payload


def _recorded_hashes(
    manifest: Mapping[str, Any] | None,
    dynamics_payload: Mapping[str, Any] | None,
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    provenance = (manifest or {}).get("provenance") or {}
    for key in ("source_sha256", "dataset_sha256"):
        if provenance.get(key):
            hashes[f"manifest.{key}"] = str(provenance[key])
    dyn_provenance = (dynamics_payload or {}).get("dataset_provenance") or {}
    for key in ("dataset_sha256", "source_sha256"):
        if dyn_provenance.get(key):
            hashes[f"dynamics.{key}"] = str(dyn_provenance[key])
    return hashes


def _normalization_stats(
    ensemble: Any | None,
    dynamics_payload: Mapping[str, Any] | None,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if ensemble is not None and getattr(ensemble, "state_mean", None) is not None:
        return (
            np.asarray(ensemble.state_mean, dtype=np.float64),
            np.asarray(ensemble.state_std, dtype=np.float64),
            "ensemble",
        )
    normalization = (dynamics_payload or {}).get("normalization") or {}
    mean = normalization.get("state_mean")
    std = normalization.get("state_std")
    if mean is not None and std is not None:
        return (
            np.asarray(mean, dtype=np.float64),
            np.asarray(std, dtype=np.float64),
            "checkpoint",
        )
    return None, None, "missing"


def _return_stats(
    return_model: Any | None,
    return_payload: Mapping[str, Any] | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if return_model is not None and getattr(return_model, "state_mean", None) is not None:
        return (
            np.asarray(return_model.state_mean, dtype=np.float64),
            np.asarray(return_model.state_std, dtype=np.float64),
        )
    payload = return_payload or {}
    if payload.get("state_mean") is not None and payload.get("state_std") is not None:
        return (
            np.asarray(payload["state_mean"], dtype=np.float64),
            np.asarray(payload["state_std"], dtype=np.float64),
        )
    return None, None


def _has_return_component(
    capper: Any | None,
    return_model: Any | None,
    return_payload: Mapping[str, Any] | None,
) -> bool:
    if capper is not None or return_model is not None:
        return True
    payload = return_payload or {}
    return bool(payload.get("model_state_dict") or payload.get("cap_value") is not None)


def _has_dynamics_component(
    ensemble: Any | None,
    dynamics_payload: Mapping[str, Any] | None,
) -> bool:
    if ensemble is not None:
        return True
    payload = dynamics_payload or {}
    return bool(payload.get("models_state_dict"))


def _torch_load_if_needed(
    payload: Mapping[str, Any] | None,
    checkpoint: str | Path | None,
) -> dict[str, Any] | None:
    if payload is not None:
        return dict(payload)
    if checkpoint is None:
        return None
    try:
        return _load_payload(checkpoint)
    except Exception as exc:  # report as evidence instead of crashing readiness
        return {"_load_error": f"{type(exc).__name__}: {exc}", "_path": str(checkpoint)}


def _check_provenance(
    manifest: Mapping[str, Any] | None,
    dynamics_payload: Mapping[str, Any] | None,
    dataset_path: str | Path | None,
) -> dict[str, Any]:
    hashes = _recorded_hashes(manifest, dynamics_payload)
    if not hashes:
        return _check(
            "fail",
            "No dataset provenance hash found in the manifest or dynamics checkpoint.",
        )
    if dataset_path is None:
        return _check(
            "warn",
            "Provenance hashes are recorded but no dataset file was supplied to verify them.",
            recorded_hashes=hashes,
        )
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        return _check("fail", f"Dataset file does not exist: {dataset_path}", recorded_hashes=hashes)
    actual = compute_file_sha256(dataset_path)
    if actual not in set(hashes.values()):
        return _check(
            "fail",
            "Dataset SHA-256 does not match any recorded provenance hash.",
            actual_sha256=actual,
            recorded_hashes=hashes,
        )
    return _check(
        "pass",
        "Dataset SHA-256 matches recorded provenance.",
        actual_sha256=actual,
        recorded_hashes=hashes,
    )


def _split_ids(
    manifest: Mapping[str, Any] | None,
    dynamics_payload: Mapping[str, Any] | None,
) -> tuple[list[int], list[int], int | None]:
    manifest = manifest or {}
    metrics = (dynamics_payload or {}).get("metrics") or {}
    summary = metrics.get("data_summary") or {}
    train = _as_int_list(manifest.get("train_episode_ids") or summary.get("train_episode_ids"))
    validation = _as_int_list(
        manifest.get("validation_episode_ids") or summary.get("validation_episode_ids")
    )
    seed = manifest.get("seed")
    if seed is None:
        seed = summary.get("seed")
    if seed is None:
        seed = ((dynamics_payload or {}).get("training_config") or {}).get("seed")
    return train, validation, None if seed is None else int(seed)


def _check_splits(
    manifest: Mapping[str, Any] | None,
    dynamics_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    train, validation, seed = _split_ids(manifest, dynamics_payload)
    if not train or not validation:
        return _check(
            "fail",
            "Both train and validation episode IDs must be recorded and non-empty.",
            train_episodes=len(train),
            validation_episodes=len(validation),
        )
    overlap = sorted(set(train) & set(validation))
    if overlap:
        return _check(
            "fail",
            "Train and validation episodes overlap; this leaks validation data.",
            overlap=overlap[:20],
        )
    if seed is None:
        return _check("fail", "Split/training seed is not recorded.")
    return _check(
        "pass",
        "Train/validation episodes are disjoint and the split seed is recorded.",
        train_episodes=len(train),
        validation_episodes=len(validation),
        seed=seed,
    )


def _check_test_isolation(
    manifest: Mapping[str, Any] | None,
    dynamics_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    manifest = manifest or {}
    train, validation, _ = _split_ids(manifest, dynamics_payload)
    used = set(train) | set(validation)
    discovered: dict[str, list[int]] = {}
    for source, mapping in (("manifest", manifest), ("dynamics", dynamics_payload or {})):
        for key in _LOGICAL_SPLIT_KEYS:
            value = mapping.get(key)
            if value is not None:
                discovered[f"{source}.{key}"] = _as_int_list(value)
    test_ids = sorted(
        {
            episode
            for name, ids in discovered.items()
            if "test" in name.lower()
            for episode in ids
        }
    )
    if not test_ids:
        return _check("pass", "No final-test split is present in the training artifacts.")
    leaked = sorted(set(test_ids) & used)
    if leaked:
        return _check(
            "fail",
            "Final-test episodes appear in train or validation splits.",
            leaked=leaked[:20],
        )
    return _check(
        "pass",
        "Final-test episodes are recorded but isolated from train/validation.",
        test_episodes=len(test_ids),
    )


def _check_parity(dt_parity: Mapping[str, Any] | None) -> dict[str, Any]:
    if dt_parity is None:
        return _check(
            "blocked",
            "No frozen-DT parity report supplied. Real-DT equivalence is unverified.",
        )
    status = str(dt_parity.get("status", "")).lower()
    if status in {"blocked", "unavailable"}:
        return _check(
            "blocked",
            f"Frozen-DT parity is {status}: {dt_parity.get('detail', 'no detail')}",
        )
    if not dt_parity.get("passed", False):
        return _check(
            "fail",
            "Frozen-DT parity report did not pass.",
            max_abs_difference=dt_parity.get("max_abs_difference"),
            tolerance=dt_parity.get("tolerance"),
        )
    source = str(dt_parity.get("model_source", "unknown")).lower()
    if source != "real":
        return _check(
            "blocked",
            "Parity passed only on a mock/reference model, not the real frozen DT checkpoint.",
            model_source=source,
        )
    return _check(
        "pass",
        "Frozen-DT parity passed against the real checkpoint.",
        max_abs_difference=dt_parity.get("max_abs_difference"),
        tolerance=dt_parity.get("tolerance"),
    )


def _check_bootstrap(
    dynamics_payload: Mapping[str, Any] | None,
    manifest: Mapping[str, Any] | None,
    ensemble: Any | None,
) -> dict[str, Any]:
    metrics = (dynamics_payload or {}).get("metrics") or {}
    models = metrics.get("models") or []
    if not models:
        return _check("fail", "Dynamics training metrics contain no per-head bootstrap records.")
    train, _, _ = _split_ids(manifest, dynamics_payload)
    train_set = set(train)
    problems: list[str] = []
    seeds: list[Any] = []
    for index, model_metrics in enumerate(models):
        info = model_metrics.get("bootstrap") or {}
        seed = info.get("bootstrap_seed")
        seeds.append(seed)
        if seed is None:
            problems.append(f"head {index}: missing bootstrap_seed")
        sampled = _as_int_list(info.get("sampled_episode_ids"))
        if not sampled:
            problems.append(f"head {index}: missing sampled_episode_ids")
        invalid = sorted(set(sampled) - train_set) if train_set else []
        if invalid:
            problems.append(f"head {index}: sampled non-training episodes {invalid[:5]}")
        if int(info.get("bootstrap_transitions", 0)) <= 0:
            problems.append(f"head {index}: no bootstrap transitions")
    if ensemble is not None and getattr(ensemble, "num_models", len(models)) != len(models):
        problems.append(
            f"ensemble has {ensemble.num_models} heads but metrics record {len(models)}"
        )
    if problems:
        return _check("fail", "Trajectory-bootstrap metadata is incomplete.", problems=problems)
    return _check(
        "pass",
        "Every dynamics head records a distinct bootstrap seed and training-only episodes.",
        bootstrap_seeds=seeds,
        heads=len(models),
    )


def _check_normalization(
    ensemble: Any | None,
    dynamics_payload: Mapping[str, Any] | None,
    return_model: Any | None,
    return_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    mean, std, source = _normalization_stats(ensemble, dynamics_payload)
    if mean is None or std is None:
        return _check("fail", "Dynamics normalization statistics are missing.")
    if mean.shape != std.shape or mean.ndim != 1:
        return _check(
            "fail",
            "Dynamics state_mean/state_std must be matching 1-D arrays.",
            mean_shape=list(mean.shape),
            std_shape=list(std.shape),
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        return _check("fail", "Dynamics normalization statistics contain non-finite values.")
    if np.any(std <= 0):
        return _check("fail", "Dynamics state_std must be strictly positive.")
    coordinate_system = ((dynamics_payload or {}).get("normalization") or {}).get(
        "coordinate_system"
    )
    if dynamics_payload is not None and coordinate_system != "raw_observation":
        return _check(
            "fail",
            "Dynamics normalization coordinate_system must be 'raw_observation'.",
            coordinate_system=coordinate_system,
        )

    r_mean, r_std = _return_stats(return_model, return_payload)
    if r_mean is not None and r_std is not None:
        if r_mean.shape != r_std.shape or not np.isfinite(r_mean).all() or not np.isfinite(r_std).all():
            return _check("fail", "Return-model normalization statistics are invalid.")
        if np.any(r_std <= 0):
            return _check("fail", "Return-model state_std must be strictly positive.")
        if r_mean.shape != mean.shape:
            return _check(
                "fail",
                "Return-model normalization dimension differs from dynamics.",
                dynamics_dim=int(mean.shape[0]),
                return_dim=int(r_mean.shape[0]),
            )
    return _check(
        "pass",
        "Normalization statistics are finite, positive, and consistent.",
        state_dim=int(mean.shape[0]),
        source=source,
        return_model_stats=r_mean is not None,
    )


def _check_components(
    capper: Any | None,
    return_model: Any | None,
    return_payload: Mapping[str, Any] | None,
    ensemble: Any | None,
    dynamics_payload: Mapping[str, Any] | None,
    required_variants: Sequence[str],
) -> dict[str, Any]:
    requested = tuple(required_variants)
    unknown = [variant for variant in requested if variant not in SUPPORTED_VARIANTS]
    if unknown:
        return _check("fail", f"Unknown variants requested: {unknown}")

    has_r = _has_return_component(capper, return_model, return_payload)
    has_u = _has_dynamics_component(ensemble, dynamics_payload)
    available = ["B0"]
    if has_r:
        available.append("B3")
    if has_u:
        available.append("B4")
    if has_r and has_u:
        available.append("B5")

    missing = [variant for variant in requested if variant not in available]
    if missing:
        return _check(
            "fail",
            f"Required variants are unavailable: {missing}",
            available_variants=available,
            has_r=has_r,
            has_u=has_u,
        )
    warnings: list[str] = []
    num_models = None
    if ensemble is not None:
        num_models = getattr(ensemble, "num_models", None)
    elif dynamics_payload is not None:
        num_models = (dynamics_payload.get("architecture") or {}).get("num_models")
    if has_u and num_models is not None and int(num_models) < 2:
        warnings.append("U has a single dynamics head; disagreement penalty is identically zero.")
    return _check(
        "warn" if warnings else "pass",
        "All required R/U components are available."
        if not warnings
        else "Components exist but U cannot express ensemble disagreement.",
        available_variants=available,
        has_r=has_r,
        has_u=has_u,
        num_models=num_models,
        warnings=warnings,
    )


def _check_traceability(
    dynamics_checkpoint: str | Path | None,
    return_checkpoint: str | Path | None,
    dynamics_payload: Mapping[str, Any] | None,
    return_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    problems: list[str] = []
    hashes: dict[str, str] = {}
    for name, path, payload in (
        ("dynamics", dynamics_checkpoint, dynamics_payload),
        ("return", return_checkpoint, return_payload),
    ):
        if payload is not None and payload.get("_load_error"):
            problems.append(f"{name} checkpoint could not be loaded: {payload['_load_error']}")
            continue
        if path is None:
            problems.append(f"{name} checkpoint path not supplied")
            continue
        path = Path(path)
        if not path.exists():
            problems.append(f"{name} checkpoint missing: {path}")
            continue
        hashes[name] = compute_file_sha256(path)

    dynamics_metrics = (dynamics_payload or {}).get("metrics") or {}
    if not dynamics_metrics.get("models"):
        problems.append("dynamics checkpoint has no per-head metrics")
    if dynamics_metrics.get("base_seed") is None:
        problems.append("dynamics metrics missing base_seed")
    if not dynamics_metrics.get("training_config"):
        problems.append("dynamics metrics missing training_config")
    if dynamics_payload is not None:
        num_models = (dynamics_payload.get("architecture") or {}).get("num_models")
        init_seeds = dynamics_payload.get("init_seeds")
        bootstrap = dynamics_payload.get("bootstrap")
        if num_models is not None:
            if init_seeds is None or len(init_seeds) != int(num_models):
                problems.append("checkpoint init_seeds do not match num_models")
            if bootstrap is None or len(bootstrap) != int(num_models):
                problems.append("checkpoint bootstrap records do not match num_models")

    if return_payload is not None:
        metrics = return_payload.get("metrics") or {}
        if metrics.get("best_val_loss") is None or not np.isfinite(metrics["best_val_loss"]):
            problems.append("return checkpoint metrics missing finite best_val_loss")
        coverage = metrics.get("best_val_coverage")
        if coverage is None or not (0.0 <= float(coverage) <= 1.0):
            problems.append("return checkpoint metrics missing best_val_coverage in [0, 1]")
        if "restored_checkpoint_evaluation" not in metrics:
            problems.append("return checkpoint metrics do not re-evaluate the restored model")
        if return_payload.get("horizon") is None or return_payload.get("scale") is None:
            problems.append("return checkpoint missing horizon/scale")
    elif return_checkpoint is not None:
        problems.append("return checkpoint payload could not be inspected")

    if problems:
        return _check("fail", "Checkpoint/metric traceability is incomplete.", problems=problems)
    return _check(
        "pass",
        "Checkpoints and restored metrics are traceable by SHA-256 and seed.",
        checkpoint_sha256=hashes,
    )


def _check_consistency(
    manifest: Mapping[str, Any] | None,
    dynamics_payload: Mapping[str, Any] | None,
    return_payload: Mapping[str, Any] | None,
    ensemble: Any | None,
    return_model: Any | None,
    capper: Any | None,
) -> dict[str, Any]:
    expected: dict[str, Any] = {}
    if manifest:
        expected["horizon"] = manifest.get("horizon")
        expected["reward_scale"] = manifest.get("reward_scale")
    payload_horizon = (dynamics_payload or {}).get("horizon")
    payload_scale = (dynamics_payload or {}).get("reward_scale")
    if payload_horizon is not None:
        if expected.get("horizon") is not None and int(expected["horizon"]) != int(payload_horizon):
            return _check(
                "fail",
                "Manifest horizon differs from dynamics checkpoint horizon.",
                manifest_horizon=int(expected["horizon"]),
                dynamics_horizon=int(payload_horizon),
            )
        expected["horizon"] = payload_horizon
    if payload_scale is not None:
        if expected.get("reward_scale") is not None and not np.isclose(
            float(expected["reward_scale"]), float(payload_scale)
        ):
            return _check(
                "fail",
                "Manifest reward_scale differs from dynamics checkpoint reward_scale.",
                manifest_scale=float(expected["reward_scale"]),
                dynamics_scale=float(payload_scale),
            )
        expected["reward_scale"] = payload_scale

    for name, component in (
        ("ensemble", ensemble),
        ("return_model", return_model),
        ("capper", capper),
    ):
        if component is None:
            continue
        horizon = getattr(component, "horizon", None)
        scale = getattr(component, "reward_scale", None)
        if scale is None:
            scale = getattr(component, "scale", None)
        if horizon is not None and expected.get("horizon") is not None and int(horizon) != int(expected["horizon"]):
            return _check(
                "fail",
                f"{name} horizon differs from the configured/checkpoint horizon.",
                component_horizon=int(horizon),
                expected_horizon=int(expected["horizon"]),
            )
        if scale is not None and expected.get("reward_scale") is not None and not np.isclose(
            float(scale), float(expected["reward_scale"])
        ):
            return _check(
                "fail",
                f"{name} reward scale differs from the configured/checkpoint scale.",
                component_scale=float(scale),
                expected_scale=float(expected["reward_scale"]),
            )

    if return_payload:
        for key in ("horizon", "scale"):
            if return_payload.get(key) is None:
                continue
            mapped = "reward_scale" if key == "scale" else "horizon"
            if expected.get(mapped) is not None:
                value = float(return_payload[key]) if key == "scale" else int(return_payload[key])
                expected_value = (
                    float(expected[mapped]) if key == "scale" else int(expected[mapped])
                )
                same = np.isclose(value, expected_value) if key == "scale" else value == expected_value
                if not same:
                    return _check(
                        "fail",
                        f"Return checkpoint {key} differs from the configured/checkpoint value.",
                        return_value=value,
                        expected_value=expected_value,
                    )
                expected[mapped] = value
    return _check(
        "pass",
        "Horizon and reward scale agree across manifest, checkpoints, and components.",
        horizon=expected.get("horizon"),
        reward_scale=expected.get("reward_scale"),
    )


def check_pilot_readiness(
    *,
    manifest: Mapping[str, Any] | None = None,
    dataset_path: str | Path | None = None,
    dynamics_checkpoint: str | Path | None = None,
    return_checkpoint: str | Path | None = None,
    dynamics_payload: Mapping[str, Any] | None = None,
    return_payload: Mapping[str, Any] | None = None,
    ensemble: Any | None = None,
    return_model: Any | None = None,
    capper: Any | None = None,
    dt_parity: Mapping[str, Any] | None = None,
    required_variants: Sequence[str] = SUPPORTED_VARIANTS,
) -> dict[str, Any]:
    """Audit a pilot artifact set and return a pass/fail/blocked report.

    The function never raises for missing evidence; callers must inspect
    ``ready`` and the per-check statuses before claiming a run is reportable.
    """
    dynamics_payload = _torch_load_if_needed(dynamics_payload, dynamics_checkpoint)
    return_payload = _torch_load_if_needed(return_payload, return_checkpoint)

    checks = {
        "provenance": _check_provenance(manifest, dynamics_payload, dataset_path),
        "splits": _check_splits(manifest, dynamics_payload),
        "final_test_isolation": _check_test_isolation(manifest, dynamics_payload),
        "frozen_dt_parity": _check_parity(dt_parity),
        "bootstrap": _check_bootstrap(dynamics_payload, manifest, ensemble),
        "normalization": _check_normalization(
            ensemble, dynamics_payload, return_model, return_payload
        ),
        "components": _check_components(
            capper,
            return_model,
            return_payload,
            ensemble,
            dynamics_payload,
            required_variants,
        ),
        "traceability": _check_traceability(
            dynamics_checkpoint,
            return_checkpoint,
            dynamics_payload,
            return_payload,
        ),
        "consistency": _check_consistency(
            manifest,
            dynamics_payload,
            return_payload,
            ensemble,
            return_model,
            capper,
        ),
    }

    blockers = [
        name
        for name, result in checks.items()
        if result["status"] in {"fail", "blocked"}
    ]
    warnings = [
        name for name, result in checks.items() if result["status"] == "warn"
    ]
    return {
        "ready": not blockers,
        "checks": checks,
        "blockers": blockers,
        "warnings": warnings,
        "required_variants": list(required_variants),
    }


def format_readiness_report(report: Mapping[str, Any]) -> str:
    """Render a readiness report as stable, human-readable text."""
    lines = [
        "PILOT READINESS: " + ("READY" if report.get("ready") else "NOT READY"),
        "=" * 72,
    ]
    for name, result in report.get("checks", {}).items():
        lines.append(f"[{result['status'].upper():7s}] {name}: {result['detail']}")
        for key, value in result.get("evidence", {}).items():
            if value not in (None, [], {}, ""):
                lines.append(f"           {key}: {value}")
    if report.get("blockers"):
        lines.append("Blockers: " + ", ".join(report["blockers"]))
    if report.get("warnings"):
        lines.append("Warnings: " + ", ".join(report["warnings"]))
    return "\n".join(lines)


def readiness_json(report: Mapping[str, Any]) -> str:
    """Serialize a readiness report deterministically."""
    return json.dumps(report, indent=2, sort_keys=True, default=str)
