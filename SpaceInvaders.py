from ale_py import ALEInterface
from ale_py.roms import SpaceInvaders
import gymnasium as gym
import random

#ale = ALEInterface()
env = gym.make("ALE/SpaceInvaders-v5", render_mode = "human")
env.metadata["render_fps"] = 60
height,width,channels = env.observation_space.shape
actions = env.action_space.n

print(height,width,channels,actions) 
# 210 160 3 4 
from NEAT.neat import NEAT
model: NEAT = NEAT(
  inputSize= 216,
  outputSize= 1,
  populationSize=50,
  C1=1.0,
  C2= 2.0,
  C3=3.0
)

model.train(
  env=env,
  epochs=5,
  goal=300,
  distance_t=0.5
)  

""" episodes = 2
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = random.randrange(0, actions, 1)
        n_state, reward, done, truncated, info = env.step(action)
        score += reward
        #print('n_state:{}, reward:{}, done:{}, truncated:{}, info:{}'.format(n_state, reward, done, truncated, info))
    print('Episode:{} Score:{}'.format(episode, score))
env.close() 
 """