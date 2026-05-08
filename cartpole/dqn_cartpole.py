import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from replay_buffer import ReplayBuffer


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.network(x)


def choose_action(state, q_network, epsilon, action_size):
    if np.random.random() < epsilon:
        return np.random.randint(action_size)

    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    with torch.no_grad():
        q_values = q_network(state_tensor)

    return torch.argmax(q_values).item()


def train_step(q_network, optimizer, replay_buffer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return

    batch = replay_buffer.sample(batch_size)

    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.FloatTensor(np.array(next_states))
    dones = torch.FloatTensor(dones).unsqueeze(1)

    current_q_values = q_network(states).gather(1, actions)

    with torch.no_grad():
        next_q_values = q_network(next_states).max(1, keepdim=True)[0]
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = nn.MSELoss()(current_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def evaluate_agent(env, q_network, episodes=100):
    rewards = []
    steps_list = []

    for _ in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = choose_action(
                state,
                q_network,
                epsilon=0.0,
                action_size=env.action_space.n
            )

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        steps_list.append(steps)

    return rewards, steps_list


def main():
    env = gym.make("CartPole-v1")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    q_network = DQN(state_size, action_size)
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer(capacity=10000)

    episodes = 500
    batch_size = 64
    gamma = 0.99

    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.05

    episode_rewards = []

    for episode in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = choose_action(state, q_network, epsilon, action_size)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay_buffer.add(state, action, reward, next_state, done)

            train_step(q_network, optimizer, replay_buffer, batch_size, gamma)

            state = next_state
            total_reward += reward

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        if (episode + 1) % 50 == 0:
            recent_avg = np.mean(episode_rewards[-50:])
            print(
                f"Episode {episode + 1}, "
                f"Reward: {total_reward}, "
                f"Average Reward: {recent_avg:.2f}, "
                f"Epsilon: {epsilon:.3f}"
            )

    test_rewards, test_steps = evaluate_agent(env, q_network, episodes=100)

    print("\nEvaluation Results:")
    print("Average reward:", np.mean(test_rewards))
    print("Best reward:", np.max(test_rewards))
    print("Worst reward:", np.min(test_rewards))
    print("Average steps:", np.mean(test_steps))

    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("CartPole DQN Training Rewards")
    plt.savefig("graphs/cartpole_dqn_rewards.png")
    plt.close()

    env.close()


if __name__ == "__main__":
    main()