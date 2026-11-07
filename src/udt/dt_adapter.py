"""Frozen Decision Transformer adapter for inference-time intervention.

Maintains context sequences, normalization statistics, action bounds,
and strictly keeps the underlying DT model frozen without gradient computations.
"""

from typing import Any
import numpy as np
import torch
import torch.nn as nn


class DTAdapter:
    """Inference-time adapter for frozen Decision Transformers.

    Maintains the rolling context window K (states, actions, returns-to-go, timesteps),
    normalizes observations, enforces action boundaries, and allows Module U to
    override the action stored in the transformer's conditioning history.
    """

    def __init__(
        self,
        model: nn.Module,
        state_dim: int,
        act_dim: int,
        max_length: int = 20,
        state_mean: np.ndarray | None = None,
        state_std: np.ndarray | None = None,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        device: str | torch.device = "cpu",
    ):
        self.model = model
        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)
        self.max_length = int(max_length)
        self.device = torch.device(device)

        # Normalization parameters
        self.state_mean = np.asarray(state_mean, dtype=np.float32) if state_mean is not None else None
        self.state_std = np.asarray(state_std, dtype=np.float32) if state_std is not None else None
        if self.state_std is not None:
            self.state_std = np.maximum(self.state_std, 1e-6)

        # Action bounds
        self.action_low = np.asarray(action_low, dtype=np.float32) if action_low is not None else None
        self.action_high = np.asarray(action_high, dtype=np.float32) if action_high is not None else None

        # Strictly freeze underlying model
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # History buffers
        self.history_states: list[np.ndarray] = []
        self.history_actions: list[np.ndarray] = []
        self.history_rtgs: list[float] = []
        self.history_timesteps: list[int] = []

    def reset(self) -> None:
        """Clear the context window for a new episode."""
        self.history_states.clear()
        self.history_actions.clear()
        self.history_rtgs.clear()
        self.history_timesteps.clear()

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply precomputed normalization statistics to raw observation."""
        obs_arr = np.asarray(obs, dtype=np.float32)
        if self.state_mean is not None and self.state_std is not None:
            return (obs_arr - self.state_mean) / self.state_std
        return obs_arr

    def step(self, obs: np.ndarray, rtg_token: float, step: int) -> np.ndarray:
        """Produce the deterministic baseline DT action a_{0, t}.

        Args:
            obs: Raw environment observation s_t.
            rtg_token: Clipped or unclipped conditioning token B_t.
            step: Environment timestep t.

        Returns:
            Continuous action vector a_{0, t}.
        """
        norm_obs = self.normalize_obs(obs)
        self.history_states.append(norm_obs)
        self.history_rtgs.append(float(rtg_token))
        self.history_timesteps.append(int(step))
        # Temporary zero-padding action for the current timestep
        self.history_actions.append(np.zeros(self.act_dim, dtype=np.float32))

        # Slice rolling context window of size <= K
        k = self.max_length
        states_k = np.array(self.history_states[-k:], dtype=np.float32)
        actions_k = np.array(self.history_actions[-k:], dtype=np.float32)
        rtgs_k = np.array(self.history_rtgs[-k:], dtype=np.float32)[:, None]
        timesteps_k = np.array(self.history_timesteps[-k:], dtype=np.int64)

        states_t = torch.as_tensor(states_k, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions_t = torch.as_tensor(actions_k, dtype=torch.float32, device=self.device).unsqueeze(0)
        rtgs_t = torch.as_tensor(rtgs_k, dtype=torch.float32, device=self.device).unsqueeze(0)
        timesteps_t = torch.as_tensor(timesteps_k, dtype=torch.long, device=self.device).unsqueeze(0)
        rewards_t = torch.zeros(1, states_k.shape[0], device=self.device, dtype=torch.float32)

        with torch.no_grad():
            out = self.model.get_action(
                states_t,
                actions_t,
                rewards_t,
                rtgs_t,
                timesteps_t,
            )
            action = out.detach().cpu().numpy().astype(np.float32)

        # Enforce action boundaries if configured
        if self.action_low is not None and self.action_high is not None:
            action = np.clip(action, self.action_low, self.action_high)

        # Replace temporary placeholder with predicted action
        self.history_actions[-1] = action
        return action

    def record_commanded_action(self, commanded_action: np.ndarray) -> None:
        """Update history with the commanded action chosen by Module U.

        Per RRD Section 2.4: History stores the commanded action a*,
        not the original DT proposal or actuator-corrupted execution.
        """
        if not self.history_actions:
            raise RuntimeError("Cannot record commanded action before calling step()")
        cmd_arr = np.asarray(commanded_action, dtype=np.float32)
        if self.action_low is not None and self.action_high is not None:
            cmd_arr = np.clip(cmd_arr, self.action_low, self.action_high)
        self.history_actions[-1] = cmd_arr

    def policy_fn(self, tokens_used: list[float], obs: np.ndarray, step: int) -> np.ndarray:
        """Callable interface matching evaluate_with_budget signature."""
        rtg_token = tokens_used[-1] if tokens_used else 0.0
        return self.step(obs=obs, rtg_token=rtg_token, step=step)
