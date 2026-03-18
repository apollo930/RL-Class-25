"""Leaderboard submission policy for Homework 2 Problem 3 (PPO Pong)."""

import numpy as np
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic used during training."""

    def __init__(self, obs_dim: int = 8, act_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, act_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        return self.actor(h), self.critic(h).squeeze(-1)


def load_policy(checkpoint_path: str):
    """Load PPO policy and return a callable obs->action."""
    device = torch.device("cpu")
    model = ActorCritic(obs_dim=8, act_dim=3).to(device)
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
            logits, _ = model(obs_t.to(device))
            actions = torch.argmax(logits, dim=-1)

        return actions.item() if actions.numel() == 1 else actions

    return policy