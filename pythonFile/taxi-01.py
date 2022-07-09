from argparse import Action
import gym
import time

env = gym.make("Taxi-v3")

env.reset()

env.render()

# take a look at the action spaces
print(f"Action spaces: {env.action_space}")

# random action
action = env.action_space.sample()
print(f"Random action: {action}")

# look at the observation spaces
print(f"State space: {env.observation_space}")

new_state, reward, done, info = env.step(action)

print(f"New state: {new_state} \nreward: {reward} \ndone: {done} \ninfo: {info}")

time.sleep(4)
