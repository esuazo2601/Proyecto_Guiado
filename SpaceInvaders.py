from ale_py import ALEInterface
from ale_py.roms import SpaceInvaders
import gymnasium as gym
import random


#ale = ALEInterface()
env = gym.make("SpaceInvaders-v4", render_mode = "human")
env.metadata["render_fps"] = 60
height,width,channels = env.observation_space.shape
actions = env.action_space.n

#print(height,width,channels,actions) 210 160 3 6

""" for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    #print(observation,reward,terminated,truncated, info)
    env.render()

    if terminated or truncated:
        observation, info = env.reset()
        print(observation)
        print(info)
env.close()  
 """
from NEAT.neat import NEAT
model: NEAT = NEAT(
  inputSize= 370,
  outputSize= actions,
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