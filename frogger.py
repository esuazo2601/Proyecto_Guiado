import numpy as np
import gymnasium
import random

env = gymnasium.make('ALE/Frogger-v5', render_mode='human')
height, width, channels = env.observation_space.shape
actions = env.action_space.n

episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = random.randrange(0, actions, 1)
        n_state, reward, done, truncated, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()