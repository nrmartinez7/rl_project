import gymnasium as gym

env = gym.make("Taxi-v4")

state, info = env.reset()

done = False
total_reward = 0

while not done:
    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)

    print("State:", next_state)
    print("Action:", action)
    print("Reward:", reward)
    print("-------------------")

    total_reward += reward

    state = next_state

    done = terminated or truncated

print("Final reward:", total_reward)

env.close()