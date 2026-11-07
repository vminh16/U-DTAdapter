"""Research tools for frozen Decision Transformer interventions."""

from udt.data import audit_dataset, load_dataset, prepare_dataset
from udt.download import download_dataset
from udt.dt_adapter import DTAdapter
from udt.dynamics import DynamicsEnsemble, DynamicsMLP, ModuleUController, train_dynamics_ensemble
from udt.evaluation import RolloutResult, evaluate_with_budget
from udt.intervention import ReturnBudget, ensemble_scores, propose_actions
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
    "ReturnBudget",
    "ReturnCapMLP",
    "ReturnCapper",
    "RolloutResult",
    "audit_dataset",
    "compute_expectile_foc_residual",
    "compute_global_expectile",
    "download_dataset",
    "ensemble_scores",
    "evaluate_with_budget",
    "load_dataset",
    "prepare_dataset",
    "propose_actions",
    "train_dynamics_ensemble",
    "train_return_cap",
]
