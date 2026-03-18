"""Leaderboard submission policy for Homework 2 Problem 4 (DQN CartPole)."""

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Q-network architecture used during training."""

    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


def load_policy(checkpoint_path: str):
    """Load DQN policy and return a callable obs->action."""
    device = torch.device("cpu")
    model = QNetwork(state_dim=4, action_dim=2).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    def policy(obs):
        if isinstance(obs, np.ndarray):
            obs_t = torch.from_numpy(obs).float()
        elif isinstance(obs, torch.Tensor):
            obs_t = obs.float()
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32)

        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)

        with torch.no_grad():
            q_values = model(obs_t.to(device))
            actions = torch.argmax(q_values, dim=-1)

        return actions.item() if actions.numel() == 1 else actions

    return policy
