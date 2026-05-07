import gymnasium as gym
import numpy as np
import random

env = gym.make("Taxi-v4")

num_states = env.observation_space.n
num_actions = env.action_space.n

q_table = np.zeros((num_states, num_actions))

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.9995
min_epsilon = 0.05

episodes = 5000

for episode in range(episodes):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        q_table[state, action] = old_value + alpha * (
            reward + gamma * next_max - old_value
        )

        state = next_state
        total_reward += reward

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if (episode + 1) % 500 == 0:
        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

# Evaluate trained agent
test_episodes = 100
total_test_rewards = []
total_test_steps = []

for episode in range(test_episodes):
    state, info = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        # No exploration during testing
        action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state = next_state
        total_reward += reward
        steps += 1

    total_test_rewards.append(total_reward)
    total_test_steps.append(steps)

print("\nEvaluation Results:")
print("Average reward:", np.mean(total_test_rewards))
print("Best reward:", np.max(total_test_rewards))
print("Worst reward:", np.min(total_test_rewards))
print("Average steps:", np.mean(total_test_steps))    

env.close()