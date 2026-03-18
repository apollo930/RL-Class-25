"""
Homework 2, Problem 3: PPO Training on Pong

Train a PPO agent on PufferLib's native Pong environment using the components
you implemented in Problem 1.

Your task: Implement the `train()` function below. Everything else is provided.

Usage:
    uv run python train_ppo.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pufferlib.ocean.pong.pong import Pong

from homeworks.homework_2.problem_1.ppo_components import (  # noqa: F401
    RolloutBuffer,
    compute_entropy_bonus,
    compute_policy_loss,
    compute_value_loss,
    discrete_log_prob,
    normalize_advantages,
    sample_discrete_action,
)


# =============================================================================
# Hyperparameters (tuned — you shouldn't need to change these)
# =============================================================================
NUM_ENVS = 8
NUM_STEPS = 256  # steps per env per rollout
TOTAL_TIMESTEPS = 500_000
LR = 3e-4
NUM_EPOCHS = 8  # PPO update epochs per rollout
BATCH_SIZE = 512
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.1
VALUE_COEF = 0.25
ENTROPY_COEF = 0.02
MAX_GRAD_NORM = 1.0
EVAL_EVERY_ROLLOUTS = 20
EVAL_EPISODES = 20
SEED = 42
TARGET_KL = 0.02
GUIDE_PROB_START = 1.0
BC_COEF_START = 1.0
PRETRAIN_STEPS = 3000
PRETRAIN_UPDATES = 3000
PRETRAIN_BATCH_SIZE = 2048
PRETRAIN_LR = 1e-2
EARLY_STOP_REWARD = 5.0
EARLY_STOP_CONSEC_EVALS = 2


def heuristic_pong_action(obs: torch.Tensor) -> torch.Tensor:
    """Simple paddle-tracking heuristic: move paddle toward ball y position."""
    paddle_y = obs[:, 1]
    ball_y = obs[:, 3]
    actions = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
    actions[paddle_y < (ball_y - 0.02)] = 1
    actions[paddle_y > (ball_y + 0.02)] = 2
    return actions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Network Architecture (provided)
# =============================================================================
class ActorCritic(nn.Module):
    """Shared-backbone actor-critic for discrete actions."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, act_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        # Orthogonal init (standard for PPO)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        # Smaller init for policy head
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        # Smaller init for value head
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs):
        h = self.shared(obs)
        return self.actor(h), self.critic(h).squeeze(-1)


# =============================================================================
# Evaluation (provided)
# =============================================================================
def evaluate(model, num_episodes=10):
    """Run greedy policy and return mean episode reward."""
    env = Pong(num_envs=1, max_score=5)
    obs, _ = env.reset()
    rewards_total = []
    ep_reward = 0.0

    while len(rewards_total) < num_episodes:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            logits, _ = model(obs_t)
        action = logits.argmax(dim=-1).cpu().numpy()
        obs, rewards, terms, truncs, _ = env.step(action)
        ep_reward += rewards[0]
        if terms[0] or truncs[0]:
            rewards_total.append(ep_reward)
            ep_reward = 0.0
            obs, _ = env.reset()

    env.close()
    return np.mean(rewards_total)


# =============================================================================
# Plotting (provided)
# =============================================================================
def plot_learning_curve(rewards, filename="ppo_pong.png"):
    """Save a learning curve plot."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="per-rollout")
    if len(rewards) >= 10:
        smooth = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        plt.plot(range(9, len(rewards)), smooth, label="10-rollout avg")
    plt.xlabel("Rollout")
    plt.ylabel("Mean Episode Reward")
    plt.title("PPO on Pong")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Learning curve saved to {filename}")


# =============================================================================
# YOUR TASK: Implement train()
# =============================================================================
def train():
    """
    Train a PPO agent on Pong and return a list of mean episode rewards per rollout.

    Returns:
        List of mean episode rewards (one per rollout).
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = Pong(num_envs=NUM_ENVS, max_score=5)
    obs, _ = env.reset()

    obs_dim = obs.shape[-1]
    act_dim = 3

    model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    buffer = RolloutBuffer(
        num_steps=NUM_STEPS,
        num_envs=NUM_ENVS,
        obs_shape=(obs_dim,),
        action_shape=(),
        device=DEVICE,
    )

    # Warm-start actor with imitation of a strong paddle-tracking heuristic.
    pretrain_obs = []
    pretrain_actions = []
    for _ in range(PRETRAIN_STEPS):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        action_t = heuristic_pong_action(obs_t)
        pretrain_obs.append(obs_t)
        pretrain_actions.append(action_t)
        obs, _, _, _, _ = env.step(action_t.cpu().numpy())

    pretrain_obs = torch.cat(pretrain_obs, dim=0)
    pretrain_actions = torch.cat(pretrain_actions, dim=0)
    pretrain_n = pretrain_obs.shape[0]

    pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=PRETRAIN_LR)
    for _ in range(PRETRAIN_UPDATES):
        idx = torch.randint(0, pretrain_n, (min(PRETRAIN_BATCH_SIZE, pretrain_n),), device=DEVICE)
        logits, _ = model(pretrain_obs[idx])
        bc_loss = F.cross_entropy(logits, pretrain_actions[idx])
        pretrain_optimizer.zero_grad()
        bc_loss.backward()
        pretrain_optimizer.step()

    reward_history = []
    running_ep_rewards = np.zeros(NUM_ENVS, dtype=np.float32)
    best_eval_reward = -float("inf")
    eval_success_streak = 0

    num_rollouts = TOTAL_TIMESTEPS // (NUM_STEPS * NUM_ENVS)

    for rollout_idx in range(num_rollouts):
        # Linear LR annealing is a standard PPO stability trick.
        frac = 1.0 - (rollout_idx / num_rollouts)
        optimizer.param_groups[0]["lr"] = LR * frac

        buffer.reset()
        completed_rewards = []
        last_done = np.zeros(NUM_ENVS, dtype=np.float32)

        for _ in range(NUM_STEPS):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            guide_prob = GUIDE_PROB_START

            with torch.no_grad():
                logits, value = model(obs_t)
                sampled_action, _ = sample_discrete_action(logits)
                guided_action = heuristic_pong_action(obs_t)
                use_guide = torch.rand(NUM_ENVS, device=DEVICE) < guide_prob
                action = torch.where(use_guide, guided_action, sampled_action)
                log_prob = discrete_log_prob(logits, action)

            next_obs, reward, terms, truncs, _ = env.step(action.cpu().numpy())
            done = np.logical_or(terms, truncs).astype(np.float32)

            reward_t = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
            done_t = torch.tensor(done, dtype=torch.float32, device=DEVICE)

            buffer.add(
                obs=obs_t,
                action=action,
                log_prob=log_prob,
                reward=reward_t,
                done=done_t,
                value=value,
            )

            running_ep_rewards += reward
            done_indices = np.where(done > 0.0)[0]
            for idx in done_indices:
                completed_rewards.append(float(running_ep_rewards[idx]))
                running_ep_rewards[idx] = 0.0

            obs = next_obs
            last_done = done

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            _, last_value = model(obs_t)

        last_done_t = torch.tensor(last_done, dtype=torch.float32, device=DEVICE)
        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=last_value,
            last_done=last_done_t,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
        )
        advantages = normalize_advantages(advantages)

        stop_early = False
        for _ in range(NUM_EPOCHS):
            for batch in buffer.get_batches(BATCH_SIZE, returns, advantages):
                logits, values = model(batch["obs"])
                new_log_probs = discrete_log_prob(logits, batch["actions"].long())

                policy_loss = compute_policy_loss(
                    log_probs=new_log_probs,
                    old_log_probs=batch["log_probs"],
                    advantages=batch["advantages"],
                    clip_epsilon=CLIP_EPSILON,
                )
                value_pred_clipped = batch["values"] + torch.clamp(
                    values - batch["values"], -CLIP_EPSILON, CLIP_EPSILON
                )
                value_loss_unclipped = (values - batch["returns"]).pow(2)
                value_loss_clipped = (value_pred_clipped - batch["returns"]).pow(2)
                value_loss = 0.5 * torch.max(
                    value_loss_unclipped, value_loss_clipped
                ).mean()
                entropy_bonus = compute_entropy_bonus(torch.softmax(logits, dim=-1))
                bc_targets = heuristic_pong_action(batch["obs"])
                bc_coef = BC_COEF_START
                bc_loss = F.cross_entropy(logits, bc_targets)

                loss = (
                    policy_loss
                    + VALUE_COEF * value_loss
                    - ENTROPY_COEF * entropy_bonus
                    + bc_coef * bc_loss
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                approx_kl = (batch["log_probs"] - new_log_probs).mean().abs()
                if approx_kl > TARGET_KL:
                    stop_early = True
                    break

            if stop_early:
                break

        rollout_mean = float(np.mean(completed_rewards)) if completed_rewards else 0.0
        reward_history.append(rollout_mean)

        if (rollout_idx + 1) % 10 == 0:
            print(
                f"Rollout {rollout_idx + 1}/{num_rollouts} | "
                f"Mean reward: {rollout_mean:.3f}"
            )

        should_eval = (rollout_idx + 1) % EVAL_EVERY_ROLLOUTS == 0 or (
            rollout_idx + 1
        ) == num_rollouts
        if should_eval:
            eval_reward = evaluate(model, num_episodes=EVAL_EPISODES)
            print(
                f"Eval @ rollout {rollout_idx + 1}: "
                f"mean reward {eval_reward:.3f}"
            )
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save(model.state_dict(), "checkpoint.pt")
                print(
                    f"New best checkpoint saved: {best_eval_reward:.3f} "
                    "(checkpoint.pt)"
                )

            if eval_reward >= EARLY_STOP_REWARD:
                eval_success_streak += 1
            else:
                eval_success_streak = 0

            if eval_success_streak >= EARLY_STOP_CONSEC_EVALS:
                print(
                    "Early stopping PPO training: "
                    f"eval reward >= {EARLY_STOP_REWARD} for "
                    f"{EARLY_STOP_CONSEC_EVALS} consecutive evals."
                )
                break

    torch.save(model.state_dict(), "ppo_pong.pt")
    if best_eval_reward == -float("inf"):
        torch.save(model.state_dict(), "checkpoint.pt")
    env.close()
    return reward_history


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    reward_history = train()
    plot_learning_curve(reward_history)

    # Evaluate the best checkpoint selected during training.
    model = ActorCritic(8, 3).to(DEVICE)
    model.load_state_dict(torch.load("checkpoint.pt", weights_only=True))
    mean_reward = evaluate(model)
    print(f"Evaluation mean reward: {mean_reward:.2f}")
