import gym
import time


env = gym.make("Taxi-v3")


num_steps = 100

env.reset()


for step in range(num_steps):
    # pick a random action
    action = env.action_space.sample()
    
    # apply the action to the env
    new_state, reward, done, info = env.step(action)
    
    time.sleep(0.1)
    
    env.render()
    
    
    if done:
        env.reset()
        
env.close()

