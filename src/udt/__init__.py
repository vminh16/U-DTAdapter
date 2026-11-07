"""Research tools for frozen Decision Transformer interventions."""

from udt.data import audit_dataset, load_dataset, prepare_dataset
from udt.download import download_dataset
from udt.dt_adapter import DTAdapter
from udt.dt_integration import (
    RealDTUnavailable,
    assert_parameters_frozen,
    load_decision_transformer,
    load_frozen_dt_adapter,
    verify_b0_parity,
    verify_n1_matches_dt_action,
)
from udt.dynamics import DynamicsEnsemble, DynamicsMLP, ModuleUController, train_dynamics_ensemble
from udt.evaluation import RolloutResult, evaluate_with_budget
from udt.experiment import VariantRun, run_variant
from udt.intervention import ReturnBudget, ensemble_scores, propose_actions
from udt.profiling import measure_controller_latency, measure_latency
from udt.readiness import check_pilot_readiness, format_readiness_report
from udt.return_cap import (
    ExpectileLoss,
    GlobalReturnCap,
    ReturnCapMLP,
    ReturnCapper,
    compute_expectile_foc_residual,
    compute_global_expectile,
    train_return_cap,
)

__version__ = "0.1.0"

__all__ = [
    "DTAdapter",
    "DynamicsEnsemble",
    "DynamicsMLP",
    "ExpectileLoss",
    "GlobalReturnCap",
    "ModuleUController",
    "RealDTUnavailable",
    "ReturnBudget",
    "ReturnCapMLP",
    "ReturnCapper",
    "RolloutResult",
    "VariantRun",
    "assert_parameters_frozen",
    "audit_dataset",
    "check_pilot_readiness",
    "compute_expectile_foc_residual",
    "compute_global_expectile",
    "download_dataset",
    "ensemble_scores",
    "evaluate_with_budget",
    "format_readiness_report",
    "load_dataset",
    "load_decision_transformer",
    "load_frozen_dt_adapter",
    "measure_controller_latency",
    "measure_latency",
    "prepare_dataset",
    "propose_actions",
    "run_variant",
    "train_dynamics_ensemble",
    "train_return_cap",
    "verify_b0_parity",
    "verify_n1_matches_dt_action",
]
