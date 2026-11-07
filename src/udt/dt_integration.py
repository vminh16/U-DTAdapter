"""Frozen Decision Transformer integration and parity checks.

The baseline Decision Transformer code lives under
``baselines/decision_transformer`` and depends on the optional ``transformers``
package. When that dependency or a checkpoint is unavailable this module
raises an explicit error instead of silently falling back to a mock model.
"""

from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from udt.dt_adapter import DTAdapter


class RealDTUnavailable(RuntimeError):
    """Raised when real DT inference cannot be constructed."""


def load_frozen_dt_adapter(
    model: nn.Module,
    *,
    state_dim: int,
    act_dim: int,
    max_length: int = 20,
    state_mean: np.ndarray | None = None,
    state_std: np.ndarray | None = None,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
    device: str | torch.device = "cpu",
) -> DTAdapter:
    """Wrap an already-constructed (or loaded) DT model for inference."""
    return DTAdapter(
        model=model,
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=max_length,
        state_mean=state_mean,
        state_std=state_std,
        action_low=action_low,
        action_high=action_high,
        device=device,
    )


def load_decision_transformer(
    checkpoint_path: str | Path,
    *,
    state_dim: int,
    act_dim: int,
    hidden_size: int = 128,
    max_length: int = 20,
    max_ep_len: int = 4096,
    n_layer: int = 3,
    n_head: int = 1,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Construct the baseline DT architecture and load a checkpoint.

    Raises:
        RealDTUnavailable: If the baseline dependency tree (``transformers``)
            or the checkpoint cannot be loaded. This is intentionally loud.
    """
    baseline_root = Path(__file__).resolve().parents[2] / "baselines" / "decision_transformer" / "gym"
    if not baseline_root.exists():
        raise RealDTUnavailable(f"Baseline DT code not found at {baseline_root}")
    if str(baseline_root) not in sys.path:
        sys.path.insert(0, str(baseline_root))

    try:
        from decision_transformer.models.decision_transformer import DecisionTransformer
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise RealDTUnavailable(
            "Real Decision Transformer inference requires the optional baseline "
            "dependencies (notably `transformers`). Install them and retry; "
            f"import failed with: {exc}"
        ) from exc

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=hidden_size,
        max_length=max_length,
        max_ep_len=max_ep_len,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=4 * hidden_size,
        activation_function="relu",
        n_positions=1024,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict)
    model.to(torch.device(device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, payload if isinstance(payload, dict) else {"state_dict": payload}


def vanilla_dt_action(
    model: nn.Module,
    adapter: DTAdapter,
    obs: np.ndarray,
    rtg_token: float,
    step: int,
) -> np.ndarray:
    """Evaluate one vanilla-DT action without adapter-side intervention."""
    normalized = adapter.normalize_obs(obs)
    states = torch.as_tensor(normalized, dtype=torch.float32, device=adapter.device)
    states = states.reshape(1, 1, -1)
    actions = torch.zeros(1, 1, adapter.act_dim, dtype=torch.float32, device=adapter.device)
    rtgs = torch.tensor([[[float(rtg_token)]]], dtype=torch.float32, device=adapter.device)
    timesteps = torch.tensor([[int(step)]], dtype=torch.long, device=adapter.device)
    with torch.no_grad():
        action = model.get_action(states, actions, None, rtgs, timesteps)
    return action.detach().cpu().numpy().astype(np.float32).reshape(-1)


def verify_b0_parity(
    model: nn.Module,
    adapter: DTAdapter,
    *,
    observations: list[np.ndarray],
    rtg_tokens: list[float],
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Compare a fresh adapter's B0 action with direct vanilla DT inference."""
    adapter.reset()
    history_states: list[np.ndarray] = []
    history_actions: list[np.ndarray] = []
    history_rtgs: list[float] = []
    history_timesteps: list[int] = []
    k = adapter.max_length

    def direct_action(obs, rtg, step):
        history_states.append(adapter.normalize_obs(obs))
        history_actions.append(np.zeros(adapter.act_dim, dtype=np.float32))
        history_rtgs.append(float(rtg))
        history_timesteps.append(int(step))

        states = np.array(history_states[-k:], dtype=np.float32)[None]
        actions = np.array(history_actions[-k:], dtype=np.float32)[None]
        rtgs = np.array(history_rtgs[-k:], dtype=np.float32)[None, :, None]
        timesteps = np.array(history_timesteps[-k:], dtype=np.int64)[None]
        with torch.no_grad():
            action = model.get_action(
                torch.as_tensor(states, device=adapter.device),
                torch.as_tensor(actions, device=adapter.device),
                None,
                torch.as_tensor(rtgs, device=adapter.device),
                torch.as_tensor(timesteps, device=adapter.device),
            )
        action = action.detach().cpu().numpy().astype(np.float32).reshape(-1)
        if adapter.action_low is not None and adapter.action_high is not None:
            action = np.clip(action, adapter.action_low, adapter.action_high)
        history_actions[-1] = action.copy()
        return action

    differences = []
    for step, (obs, rtg) in enumerate(zip(observations, rtg_tokens)):
        direct = direct_action(obs, rtg, step)
        # B0 is exactly adapter.step with no R cap or U reranking.
        via_adapter = adapter.step(obs=obs, rtg_token=rtg, step=step)
        differences.append(float(np.max(np.abs(direct - via_adapter))))
    max_diff = max(differences) if differences else 0.0
    return {
        "max_abs_difference": max_diff,
        "tolerance": float(atol),
        "passed": bool(max_diff <= atol),
        "steps": len(differences),
    }


def assert_parameters_frozen(model: nn.Module) -> None:
    """Raise when any DT parameter is trainable or the model is in train mode."""
    if model.training:
        raise AssertionError("Decision Transformer must be in eval mode")
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if trainable:
        raise AssertionError(f"Decision Transformer parameters are not frozen: {trainable}")


def verify_n1_matches_dt_action(
    adapter: DTAdapter,
    controller: Any,
    *,
    obs: np.ndarray,
    rtg_token: float,
    step: int,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Verify N=1 returns the DT action of the corresponding R branch."""
    adapter.reset()
    expected = adapter.step(obs=obs, rtg_token=rtg_token, step=step)
    # The controller must see the same fresh episode context.
    adapter.reset()
    selected, info = controller.get_action(obs=obs, rtg_token=rtg_token, step=step)
    difference = float(np.max(np.abs(selected - expected)))
    return {
        "max_abs_difference": difference,
        "tolerance": float(atol),
        "passed": bool(difference <= atol),
        "selected_index": info.get("selected_index"),
    }
