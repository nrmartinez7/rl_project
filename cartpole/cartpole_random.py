import gymnasium as gym

env = gym.make("CartPole-v1")

observation, info = env.reset()
done = False
total_reward = 0
steps = 0

while not done:
    action = env.action_space.sample()

    next_observation, reward, terminated, truncated, info = env.step(action)
    print("Observation:", next_observation)
    print("Action:", action)
    print("Reward:", reward)
    print("-------------------")

    observation = next_observation
    total_reward += reward
    steps += 1
    done = terminated or truncated

print("Final reward:", total_reward)
print("Steps survived:", steps)

env.close()