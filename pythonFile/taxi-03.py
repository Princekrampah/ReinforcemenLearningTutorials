from wsgiref.simple_server import demo_app
import gym
import random
import numpy as np


def main():
    env = gym.make("Taxi-v3")
    
    # initialize the q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))
    
    # set the number of episodes
    EPISODE = 1000
    STEPS_PER_EPISODE = 90
    
    # hyperparameters
    epsilon = 1.0
    decay_rate = 0.009
    learning_rate = 0.9
    discount_rate = 0.8
    
    for episode in range(EPISODE):
        done = False
        # reset the state to the initial state
        state = env.reset()
        
        for step in range(STEPS_PER_EPISODE):
            if random.uniform(0, 1) < epsilon:
                # explore
                action = env.action_space.sample()
            else:
                # exploit
                action = np.argmax(qtable[state, :])
                
            new_state, reward, done, info = env.step(action)
            
            qtable[state, action] = qtable[state, action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])
            
            state = new_state
            
            if done:
                break
            
        # Decrease epsilon
        epsilon = np.exp(-decay_rate*epsilon)


    state = env.reset()
    done = False
    rewards = 0
    
    
    for s in range(STEPS_PER_EPISODE):
        print("Trained agent")
        print(f"Step: {s+1}")
        
        action = np.argmax(qtable[state, :])
        
        new_state, reward, done, info = env.step(action)
        
        rewards += reward
        
        env.render()
        
        print(f"Score: {rewards}")
        state = new_state
        
        if done:
            break
        
    env.close()
        


if __name__ == "__main__":
    main()


