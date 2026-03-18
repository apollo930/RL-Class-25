"""
Homework 2, Problem 4: DQN Training on CartPole

Train a DQN agent on CartPole-v1 using the components you implemented in
Problem 2.

NOTE: DQN is often not a very performant algorithm compared to policy gradient
methods like PPO. We have you implement it because its *components* — replay
buffers, target networks, epsilon-greedy exploration, and TD learning — show up
throughout modern RL. Think of this as learning the building blocks, not as the
state-of-the-art way to solve environments.

Your task: Implement the `train()` function below. Everything else is provided.

Usage:
    uv run python train_dqn.py
"""

import numpy as np
import torch
import matplotlib
import gymnasium as gym
import random

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from homeworks.homework_2.problem_2.dqn_components import (  # noqa: F401
    QNetwork,
    NStepReplayBuffer,
    batch_to_tensors,
    compute_double_dqn_target,
    compute_td_loss,
    epsilon_greedy_action,
    hard_update,
    linear_epsilon_decay,
)

# =============================================================================
# Hyperparameters (tuned — you shouldn't need to change these)
# =============================================================================
TOTAL_TIMESTEPS = 200_000
LR = 3e-4
BATCH_SIZE = 256
BUFFER_CAPACITY = 100_000
GAMMA = 0.99
N_STEP = 3  # n-step returns for better credit assignment
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 50_000
TARGET_UPDATE_FREQ = 1_000  # steps between hard target updates
LEARNING_STARTS = 5_000  # fill buffer before training
TRAIN_FREQ = 1  # train every N env steps
EVAL_FREQ_STEPS = 10_000
EVAL_EPISODES = 20
SEED = 42
EARLY_STOP_REWARD = 500.0
EARLY_STOP_CONSEC_EVALS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Evaluation (provided)
# =============================================================================
def evaluate(model, num_episodes=10):
    """Run greedy policy and return mean episode reward."""
    env = gym.make("CartPole-v1")
    rewards_total = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                q_vals = model(obs_t)
            action = q_vals.argmax(dim=-1).item()
            obs, reward, term, trunc, _ = env.step(action)
            ep_reward += reward
            done = term or trunc

        rewards_total.append(ep_reward)

    env.close()
    return np.mean(rewards_total)


# =============================================================================
# Plotting (provided)
# =============================================================================
def plot_learning_curve(rewards, filename="dqn_cartpole.png"):
    """Save a learning curve plot."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="episode reward")
    if len(rewards) >= 50:
        smooth = np.convolve(rewards, np.ones(50) / 50, mode="valid")
        plt.plot(range(49, len(rewards)), smooth, label="50-episode avg")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("DQN on CartPole-v1")
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
    Train a DQN agent on CartPole-v1 and return a list of episode rewards.

    Returns:
        List of episode rewards (one per completed episode).
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make("CartPole-v1")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    online_network = QNetwork(state_dim=obs_dim, action_dim=act_dim).to(DEVICE)
    target_network = QNetwork(state_dim=obs_dim, action_dim=act_dim).to(DEVICE)
    hard_update(online_network, target_network)

    optimizer = torch.optim.Adam(online_network.parameters(), lr=LR, eps=1e-5)
    replay_buffer = NStepReplayBuffer(
        capacity=BUFFER_CAPACITY,
        n_step=N_STEP,
        gamma=GAMMA,
    )

    obs, _ = env.reset(seed=SEED)
    episode_reward = 0.0
    reward_history = []
    best_eval_reward = -float("inf")
    eval_success_streak = 0

    for step in range(TOTAL_TIMESTEPS):
        epsilon = linear_epsilon_decay(
            step=step,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            decay_steps=EPSILON_DECAY_STEPS,
        )

        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_values = online_network(obs_t)
        action = epsilon_greedy_action(q_values, epsilon=epsilon, num_actions=act_dim)

        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc

        # For TD targets, treat only true environment terminal states as done.
        # Time-limit truncations are non-terminal and should bootstrap.
        done_for_target = bool(term)

        replay_buffer.push(obs, action, reward, next_obs, done_for_target)

        obs = next_obs
        episode_reward += reward

        if (
            step >= LEARNING_STARTS
            and step % TRAIN_FREQ == 0
            and len(replay_buffer) >= BATCH_SIZE
        ):
            batch = replay_buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = batch_to_tensors(batch, DEVICE)

            td_targets = compute_double_dqn_target(
                rewards=rewards,
                next_states=next_states,
                dones=dones,
                gamma=GAMMA**N_STEP,
                online_network=online_network,
                target_network=target_network,
            )

            q_values = online_network(states)
            loss = compute_td_loss(
                q_values=q_values,
                actions=actions,
                td_targets=td_targets,
                loss_type="huber",
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_network.parameters(), max_norm=10.0)
            optimizer.step()

        if step % TARGET_UPDATE_FREQ == 0:
            hard_update(online_network, target_network)

        if done:
            reward_history.append(episode_reward)
            episode_reward = 0.0
            obs, _ = env.reset()

            if len(reward_history) % 20 == 0:
                recent_mean = float(np.mean(reward_history[-20:]))
                print(
                    f"Step {step + 1}/{TOTAL_TIMESTEPS} | "
                    f"Episodes {len(reward_history)} | "
                    f"Recent mean reward: {recent_mean:.1f}"
                )

        should_eval = ((step + 1) % EVAL_FREQ_STEPS == 0) or (
            step + 1 == TOTAL_TIMESTEPS
        )
        if should_eval:
            eval_reward = evaluate(online_network, num_episodes=EVAL_EPISODES)
            print(
                f"Eval @ step {step + 1}: "
                f"mean reward {eval_reward:.1f}"
            )
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save(online_network.state_dict(), "checkpoint.pt")
                print(
                    f"New best checkpoint saved: {best_eval_reward:.1f} "
                    "(checkpoint.pt)"
                )

            if eval_reward >= EARLY_STOP_REWARD:
                eval_success_streak += 1
            else:
                eval_success_streak = 0

            if eval_success_streak >= EARLY_STOP_CONSEC_EVALS:
                print(
                    "Early stopping DQN training: "
                    f"eval reward >= {EARLY_STOP_REWARD} for "
                    f"{EARLY_STOP_CONSEC_EVALS} consecutive evals."
                )
                break

    torch.save(online_network.state_dict(), "dqn_cartpole.pt")
    if best_eval_reward == -float("inf"):
        torch.save(online_network.state_dict(), "checkpoint.pt")
    env.close()
    return reward_history


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    reward_history = train()
    plot_learning_curve(reward_history)

    # Evaluate the best checkpoint selected during training.
    obs_dim = 4
    act_dim = 2
    model = QNetwork(state_dim=obs_dim, action_dim=act_dim).to(DEVICE)
    model.load_state_dict(torch.load("checkpoint.pt", weights_only=True))
    mean_reward = evaluate(model)
    print(f"Evaluation mean reward: {mean_reward:.1f}")
