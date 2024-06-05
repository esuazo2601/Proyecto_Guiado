from NEAT.neat import NEAT
import torch
#import gymnasium as gym
#from gymnasium.wrappers import FlattenObservation
#import numpy as np

if __name__ == '__main__':
    #env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")
    #env = FlattenObservation(env)
    #actions = env.action_space.n
    
    model = NEAT(
        inputSize=216,
        outputSize=6,
        populationSize=200,
        C1=1.2,
        C2=2.3,
        C3=3.5
    )

    model.to(torch.device('cuda'))
    
    model.train(
        #env=env,
        epochs=400,
        goal=1100,
        distance_t=0.05,
        output_file="fitness_history_1.txt"
    )
