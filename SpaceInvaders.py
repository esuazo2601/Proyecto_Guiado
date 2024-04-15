from ale_py import ALEInterface
from ale_py.roms import SpaceInvaders
import gymnasium as gym
import signal
import time
import atexit

#ale = ALEInterface()
env = gym.make("ALE/SpaceInvaders-v5",render_mode="human")
env.metadata["render_fps"] = 60
#height,width,channels = env.observation_space.shape
#print(height,width,channels,actions) 
actions = env.action_space.n
obs_ram = env.unwrapped.ale.getRAM()

# 210 160 3 4 
from NEAT.neat import NEAT
model = NEAT(
    inputSize=len(obs_ram),
    outputSize=actions,
    populationSize=20,
    C1=1.0,
    C2=2.0,
    C3=3.0
)

# Define la función de entrenamiento y entrena el modelo
model.train(
    env=env,
    epochs=5,
    goal=500,
    distance_t=0.5
)


""" episodes = 5
for episode in range(1, episodes+1):
    state, info = env.reset()
    print(env.unwrapped.ale.getRAM())
    #print(state,info)
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